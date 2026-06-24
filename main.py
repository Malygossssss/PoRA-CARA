# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Swin Transformer (https://github.com/microsoft/Swin-Transformer)
#
# Original file:
# Copyright (c) 2021 Microsoft
# Licensed under the MIT License
# Written by Ze Liu
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import shutil
from contextlib import nullcontext
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model, build_mtl_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper

from mtl_loss_schemes import MultiTaskLoss, get_loss
from evaluation.evaluate_utils import PerformanceMeter, get_output
from evaluation.eval_edge import eval_edge_predictions
from ptflops import get_model_complexity_info
from models.lora import mark_only_lora_as_trainable, mark_prompt_as_trainable

try:
    import wandb
    wandb_available = True
except ImportError:
    print("Warning: wandb library not found. Logging is disabled.")
    wandb_available = False


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--ckpt-freq', type=int, default=5,
                        help="checkpoint saving frequency")
    parser.add_argument('--eval-freq', type=int, default=5,
                        help="model evaluation frequency")
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of epochs to train")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--seed', type=int,
                        help='Override random seed used for training')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic CUDA/cuDNN behavior for reproducibility')
    parser.add_argument('--debug-repro-steps', type=int, default=0,
                        help='Log reproducibility diagnostics for first N train steps of each epoch')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag>/run_<timestamp> (default: output)')
    parser.add_argument('--name', type=str, help='override model name')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm',
                        action='store_true', help='Use fused layernorm.')
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # MTL Config
    parser.add_argument('--tasks', type=str, default='depth',
                        help='List of tasks to run in MTL setup.')
    parser.add_argument(
        '--nyud', type=str, help='specify the path to load NYUD, replaces --data-path')
    parser.add_argument(
        '--pascal', type=str, help='specify the path to load PASCAL, replaces --data-path and --nyud')
    parser.add_argument('--eval-training-freq', type=int,
                        help='calculate performance score on the training dataset')
    parser.add_argument('--resume-backbone',
                        help='resume checkpoint into the backbone')
    parser.add_argument('--freeze-backbone',
                        action='store_true', help='Freeze encoder layers.')

    parser.add_argument('--skip_initial_validation', action='store_true',
                        help='Skip running validation at the start')
    parser.add_argument('--decoder_map', type=str,
                        help='Path to JSON file containing the type of decoder heads')
    parser.add_argument('--skip_decoder', action='store_true',
                        help='Skip loading decoder head weights')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable wandb logging.')
    parser.add_argument('--run_name', type=str,
                        help='wandb run name')
    parser.add_argument('--no_eval_50', action='store_false',
                        help='Disable the iniital eval at 50 epochs.')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)

    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"number of params: {n_parameters / 1e6} M")

    model.cuda()
    if isinstance(config.DATA.IMG_SIZE, (list, tuple)):
        input_h, input_w = config.DATA.IMG_SIZE
    else:
        input_h = input_w = config.DATA.IMG_SIZE
    macs, params = get_model_complexity_info(model, (3, input_h, input_w), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)

    logger.info(f"ptflops GMACS = {macs} and params = {params}")

    optimizer = build_optimizer(config, model)

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(
            data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.MTL:
        loss_ft = torch.nn.ModuleDict(
            {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
        all_loss_weights = {
            'depth': 1.0,
            'semseg': 1.0,
            'human_parts': 2.0,
            'sal': 5.0,
            'edge': 50.0,
            'normals': 10.0,
        }
        loss_weights = {}
        for t in config.TASKS:
            loss_weights[t] = all_loss_weights[t]

        criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model, optimizer, lr_scheduler, loss_scaler, logger)

        if not config.SKIP_INITIAL_EVAL:
            validate(config, data_loader_val, model, 0)
        if config.EVAL_MODE:
            return

    if config.MODEL.RESUME_BACKBONE:
        max_accuracy = load_checkpoint(
            config, model.backbone, optimizer, lr_scheduler, loss_scaler, logger, True)
        if config.EVAL_MODE:
            validate(config, data_loader_val, model, 0)
            return

    if config.EVAL_MODE:
        validate(config, data_loader_val, model, 0)
        return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model, logger)
        if not config.SKIP_INITIAL_EVAL:
            acc1, _, _ = validate(config, data_loader_val, model, 0)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return
    if config.MODEL.MTLORA.ENABLED:
        if config.MODEL.MTLORA.FREEZE_PRETRAINED:
            print("\nMarking LoRA params only as trainable:")
            mark_only_lora_as_trainable(model.backbone,
                                        bias=config.MODEL.MTLORA.BIAS,
                                        freeze_patch_embed=config.TRAIN.FREEZE_PATCH_EMBED,
                                        freeze_norm=config.TRAIN.FREEZE_LAYER_NORM,
                                        free_relative_bias=config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS,
                                        freeze_downsample_reduction=True if config.MODEL.MTLORA.DOWNSAMPLER_ENABLED else config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION)
            if config.MODEL.PROMPT.ENABLED:
                mark_prompt_as_trainable(model.backbone)
        else:
            print("Marking all layers as trainable")
    if config.MODEL.FREEZE_BACKBONE:
        assert (not config.MODEL.MTLORA.ENABLED)
        print("Freezing backbone.........")
        model.freeze_backbone()
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for name, p in model.named_parameters()
                      if p.requires_grad and 'lora' in name)
    total_model_params = sum(p.numel() for p in model.parameters())
    total_model_params_without_lora = total_model_params - lora_params
    decoder_params = sum(p.numel() for name, p in model.named_parameters()
                         if 'backbone' not in name)

    logger.info(
        f"\nNumber of trainable params: {trainable_params:,}\n"
        f"Decoder params:             {decoder_params:,}\n"
        f"LoRA params:                {lora_params:,}\n"
        f"Extra params:                {(trainable_params - (lora_params + decoder_params)):,}\n"
        f"Total params:               {total_model_params:,} (trainable ratio: {trainable_params/total_model_params * 100:2.2f}%)\n"
        f"Total params without LoRA:  {total_model_params_without_lora:,} (trainable ratio: {trainable_params/total_model_params_without_lora * 100:2.2f}%)"
    )
    # --- DDP wrapping (after all model setup) ---
    model_without_ddp = model
    if dist.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    logger.info("Start training")
    start_time = time.perf_counter()

    epoch = config.TRAIN.START_EPOCH
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train,
                        optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)
        if epoch % config.EVAL_FREQ == 0 or (not args.no_eval_50 and epoch == 50):
            if config.MTL:
                validate(config, data_loader_val, model, epoch)
            else:
                acc1, _, _ = validate(config, data_loader_val, model, epoch)
                max_accuracy = max(max_accuracy, acc1)

    # final eval
    validate(config, data_loader_val, model, epoch)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def _get_dist_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _prepare_edge_eval_cache(output_dir, epoch):
    cache_dir = os.path.join(output_dir, "__edge_eval_cache__", f"epoch_{int(epoch):04d}")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(os.path.join(cache_dir, "edge"), exist_ok=True)
    return cache_dir


def _save_edge_predictions(cache_dir, image_ids, edge_logits):
    edge_probs = torch.sigmoid(edge_logits.detach()).float().cpu().numpy()
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    else:
        image_ids = list(image_ids)

    for index, image_id in enumerate(image_ids):
        np.save(
            os.path.join(cache_dir, "edge", f"{image_id}.npy"),
            edge_probs[index, 0].astype(np.float32),
        )

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)

    start = time.perf_counter()
    end = time.perf_counter()
    loss_dict = None

    for idx, batch in enumerate(data_loader):
        if not config.MTL:
            samples, targets = batch
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            samples = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(
                non_blocking=True) for task in config.TASKS}

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Skip gradient sync on intermediate accumulation steps for DDP efficiency
        is_accumulating = (idx + 1) % config.TRAIN.ACCUMULATION_STEPS != 0
        sync_ctx = model.no_sync if (hasattr(model, 'no_sync') and is_accumulating) else nullcontext

        with sync_ctx():
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(samples)
                loss, loss_dict = criterion(outputs, targets)

            if config.DEBUG_REPRO_STEPS > 0 and idx < config.DEBUG_REPRO_STEPS and dist.get_rank() == 0:
                sample_mean = float(samples.mean().item())
                sample_std = float(samples.std().item())

                batch_ids = []
                if isinstance(batch, dict) and 'meta' in batch and isinstance(batch['meta'], dict) and 'image' in batch['meta']:
                    ids = batch['meta']['image']
                    if isinstance(ids, (list, tuple)):
                        batch_ids = [str(x) for x in ids[:4]]
                    else:
                        batch_ids = [str(ids)]

                target_stats = []
                if isinstance(targets, dict):
                    for task_name, task_tensor in targets.items():
                        task_tensor_f = task_tensor.float()
                        stat = f"{task_name}:mean={float(task_tensor_f.mean().item()):.6f},std={float(task_tensor_f.std().item()):.6f}"
                        if task_tensor.dim() >= 3:
                            labels = task_tensor.long().view(-1)
                            valid = labels != 255
                            valid_ratio = float(valid.float().mean().item())
                            stat += f",valid_ratio={valid_ratio:.6f}"
                            if task_name in {'semseg', 'human_parts'} and valid.any():
                                valid_labels = labels[valid]
                                uniq, counts = torch.unique(valid_labels, return_counts=True)
                                topk = min(3, uniq.numel())
                                if topk > 0:
                                    order = torch.argsort(counts, descending=True)[:topk]
                                    top_items = [f"{int(uniq[i])}:{int(counts[i])}" for i in order]
                                    stat += f",top_labels={','.join(top_items)}"
                        target_stats.append(stat)
                else:
                    target_tensor = targets.float()
                    target_stats.append(f"cls:mean={float(target_tensor.mean().item()):.6f},std={float(target_tensor.std().item()):.6f}")

                output_stats = []
                if isinstance(outputs, dict):
                    for task_name, out in outputs.items():
                        out = out.float()
                        output_stats.append(f"{task_name}:mean={float(out.mean().item()):.6f},std={float(out.std().item()):.6f}")
                else:
                    out = outputs.float()
                    output_stats.append(f"cls:mean={float(out.mean().item()):.6f},std={float(out.std().item()):.6f}")

                logger.info(
                    "[debug_repro] epoch=%d step=%d ids=%s sample(mean=%.6f,std=%.6f) target_stats=[%s] output_stats=[%s] loss=%.6f",
                    epoch,
                    idx,
                    batch_ids,
                    sample_mean,
                    sample_std,
                    "; ".join(target_stats),
                    "; ".join(output_stats),
                    float(loss.item()),
                )

            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # torch.cuda.synchronize()

        if not config.MTL:
            loss_meter.update(loss.item(), targets.size(0))
        else:
            loss_meter.update(loss.item())

        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        if wandb_available:
            metrics = {
                "train/epoch_ndx": epoch,
                "train/batch_ndx": idx,
                "train/train_loss": loss_meter.val,
                "train/train_loss_avg": loss_meter.avg,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/weight_decay": optimizer.param_groups[0]['weight_decay'],
                "train/time": batch_time.val,
                "train/time_avg": batch_time.avg,
                "train/grad_norm": norm_meter.val,
                "train/grad_norm_avg": norm_meter.avg,
                "train/loss_scale": scaler_meter.val,
                "train/loss_scale_avg": scaler_meter.avg,
                "train/memory": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            }
            if loss_dict is not None:
                for task, task_loss in loss_dict.items():
                    metrics[f"train/tasks/{task}/loss"] = task_loss.item()
            wandb.log(metrics)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    if config.EVAL_TRAINING is not None and (epoch % config.EVAL_TRAINING == 0):
        print("Training Eval:")
        performance_meter.update(
            {t: get_output(outputs[t], t) for t in config.TASKS}, targets)

        scores = performance_meter.get_score(verbose=True)
        if wandb_available:
            scores_logs = {
                "train/epoch": epoch,
            }
            if 'semseg' in scores:
                scores_logs["train/tasks/semseg/mIoU"] = scores['semseg']['mIoU']
            if 'normals' in loss_dict:
                scores_logs["train/tasks/normals/mean"] = scores['normals']['mean']
                scores_logs["train/tasks/normals/rmse"] = scores['normals']['rmse']
                scores_logs["train/tasks/normals/mean_v2"] = scores['normals']['mean_v2']
                scores_logs["train/tasks/normals/rmse_v2"] = scores['normals']['rmse_v2']
            if 'human_parts' in loss_dict:
                scores_logs["train/tasks/human_parts/mIoU"] = scores['human_parts']['mIoU']
            if 'sal' in loss_dict:
                scores_logs["train/tasks/sal/maxF"] = scores['sal']['maxF']
                scores_logs["train/tasks/sal/Beta maxF"] = scores['sal']['Beta maxF']
                scores_logs["train/tasks/sal/mIoU"] = scores['sal']['mIoU']
            if 'edge' in loss_dict:
                scores_logs["train/tasks/edge/loss"] = scores['edge']['loss']
            if 'depth' in loss_dict:
                scores_logs["train/tasks/depth/rmse"] = scores['depth']['rmse']
                scores_logs["train/tasks/depth/log_rmse"] = scores['depth']['log_rmse']

            wandb.log(scores_logs)

    epoch_time = time.perf_counter() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, epoch):
    """Evaluate the model online and save temporary edge predictions when needed."""
    tasks = config.TASKS
    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)
    loss_meter = AverageMeter()
    rank = _get_dist_rank()
    edge_eval_dir = None
    should_run_formal_edge_eval = (
        "edge" in tasks and config.DATA.DBNAME == "NYUD" and rank == 0
    )
    if should_run_formal_edge_eval:
        edge_eval_dir = _prepare_edge_eval_cache(config.OUTPUT, epoch)

    loss_ft = torch.nn.ModuleDict(
        {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
    all_loss_weights = {
        'depth': 1.0,
        'semseg': 1.0,
        'human_parts': 2.0,
        'sal': 5.0,
        'edge': 50.0,
        'normals': 10.0,
    }
    loss_weights = {}
    for t in config.TASKS:
        loss_weights[t] = all_loss_weights[t]
    criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    model.eval()
    num_val_points = 0
    logger.info("Start eval")
    start = time.perf_counter()
    loss_dict = None
    for i, batch in enumerate(data_loader):
        # Forward pass
        logger.debug(f"Image ID = {batch['meta']['image']}")
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(
            non_blocking=True) for task in tasks}

        output = model(images)

        num_val_points += 1

        # Measure performance
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            loss, loss_dict = criterion(
                output, targets)
            loss_meter.update(loss.item())
        processed_output = {t: get_output(
            output[t], t) for t in tasks}
        performance_meter.update(processed_output, targets)
        if edge_eval_dir is not None:
            _save_edge_predictions(edge_eval_dir, batch["meta"]["image"], output["edge"])
        if wandb_available:
            metrics = {
                "val/epoch_ndx": epoch,
                "val/batch_ndx": i,
                "val/val_loss": loss_meter.val,
                "val/val_loss_avg": loss_meter.avg,
            }
            if loss_dict is not None:
                for task, task_loss in loss_dict.items():
                    metrics[f"val/tasks/{task}/loss"] = task_loss.item()
            wandb.log(metrics)

    logger.info(f"val loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t")

    try:
        if edge_eval_dir is not None:
            formal_edge_results = eval_edge_predictions(
                database="NYUD",
                save_dir=edge_eval_dir,
                gt_root=config.DATA.DATA_PATH,
                write_outputs=False,
            )
            performance_meter.meters["edge"].set_formal_results(formal_edge_results)

        eval_results = performance_meter.get_score(verbose=True)
    finally:
        if edge_eval_dir is not None and os.path.isdir(edge_eval_dir):
            shutil.rmtree(edge_eval_dir, ignore_errors=True)

    epoch_time = time.perf_counter() - start
    logger.info(
        f"eval takes {datetime.timedelta(seconds=int(epoch_time))}")
    if wandb_available:
        scores_logs = {
            "val/epoch": epoch,
        }
        if 'semseg' in eval_results:
            scores_logs["val/tasks/semseg/mIoU"] = eval_results['semseg']['mIoU']
        if 'normals' in eval_results:
            scores_logs["val/tasks/normals/mean"] = eval_results['normals']['mean']
            scores_logs["val/tasks/normals/rmse"] = eval_results['normals']['rmse']
            scores_logs["val/tasks/normals/mean_v2"] = eval_results['normals']['mean_v2']
            scores_logs["val/tasks/normals/rmse_v2"] = eval_results['normals']['rmse_v2']
        if 'human_parts' in eval_results:
            scores_logs["val/tasks/human_parts/mIoU"] = eval_results['human_parts']['mIoU']
        if 'sal' in eval_results:
            scores_logs["val/tasks/sal/maxF"] = eval_results['sal']['maxF']
            scores_logs["val/tasks/sal/Beta maxF"] = eval_results['sal']['Beta maxF']
            scores_logs["val/tasks/sal/mIoU"] = eval_results['sal']['mIoU']
        if 'edge' in eval_results:
            scores_logs["val/tasks/edge/loss"] = eval_results['edge']['loss']
            if 'odsF' in eval_results['edge']:
                scores_logs["val/tasks/edge/odsF"] = eval_results['edge']['odsF']
                scores_logs["val/tasks/edge/oisF"] = eval_results['edge']['oisF']
                scores_logs["val/tasks/edge/ap"] = eval_results['edge']['ap']
        if 'depth' in eval_results:
            scores_logs["val/tasks/depth/rmse"] = eval_results['depth']['rmse']
            scores_logs["val/tasks/depth/log_rmse"] = eval_results['depth']['log_rmse']

        wandb.log(scores_logs)

    return eval_results


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        # torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.perf_counter()
        for i in range(30):
            model(images)
        # torch.cuda.synchronize()
        tic2 = time.perf_counter()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    # Enforce float32 matmul precision and disable TF32 for reproducible/strict precision behavior.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if config.DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True
        # Some CUDA ops used by dense prediction losses (e.g., nll_loss2d)
        # are not fully deterministic in current PyTorch/CUDA builds.
        # warn_only keeps reproducibility diagnostics while avoiding hard crash.
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.OUTPUT, f"run_{timestamp}")
    config.defrost()
    config.OUTPUT = output_dir
    config.freeze()
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    eval_logger = create_logger(output_dir=output_dir,
                                dist_rank=dist.get_rank(), name="eval")
    if dist.get_rank() == 0:
        path = os.path.join(output_dir, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))
    logger.info(
        "Runtime: torch=%s cuda=%s cudnn=%s device=%s amp=%s deterministic=%s tf32(matmul=%s,cudnn=%s) debug_repro_steps=%s",
        torch.__version__,
        torch.version.cuda,
        cudnn.version(),
        torch.cuda.get_device_name(torch.cuda.current_device()),
        config.AMP_ENABLE,
        config.DETERMINISTIC,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        config.DEBUG_REPRO_STEPS,
    )

    if args.disable_wandb:
        wandb_available = False
        logger.info("Wandb logging disabled.")
    elif wandb_available:
        if dist.get_rank() == 0:
            try:
                if not os.getenv("WANDB_API_KEY"):
                    wandb.login()
                else:
                    wandb.login(key=os.getenv("WANDB_API_KEY"))
                config_name = f"{os.path.basename(os.path.dirname(args.cfg))}/{os.path.basename(args.cfg)}"
                wandb.init(project='MTLoRA', config=config,
                           name=config_name if not args.run_name else args.run_name)
                wandb.config.update({'args': vars(args)})
            except wandb.exc.LaunchError:
                logger.warning("Could not initialize wandb. Logging is disabled.")
                wandb_available = False
        else:
            wandb_available = False

    main(config)
