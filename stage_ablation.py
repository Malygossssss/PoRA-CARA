"""Shared backbone-stage policy for construction, training, evaluation and checkpoints."""

import hashlib
import json


def verify_initialization_file(config, path):
    expected = getattr(config.MODEL, "INIT_CHECKPOINT_SHA256", "")
    if not expected:
        return
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("Common initialization checkpoint SHA256 mismatch.")


def resolve_stage_modes(config):
    policy = getattr(config.MODEL, "STAGE_ABLATION", None)
    if policy is None or not policy.ENABLED:
        return None
    modes = list(policy.MODES)
    if len(modes) != len(config.MODEL.SWIN.DEPTHS):
        raise ValueError("STAGE_ABLATION.MODES must match the number of backbone stages.")
    if any(mode not in {"B", "P", "G"} for mode in modes):
        raise ValueError("STAGE_ABLATION.MODES only accepts B, P, G.")
    if not config.MODEL.PROMPT.ENABLED or not config.MODEL.MTLORA.ENABLED:
        raise ValueError("Stage ablation requires task-prompt and group-LoRA.")
    ag = config.MODEL.AGMTLORA
    mtlora = config.MODEL.MTLORA
    if not ag.ENABLED or ag.GROUPING_SOURCE != "fixed_json" or not mtlora.AGMTLORA_ENABLED:
        raise ValueError("Stage ablation requires fixed_json, already resolved grouping; no search is allowed.")
    if not mtlora.AGMTLORA_GROUP_NAMES or not mtlora.AGMTLORA_GROUP_RANKS:
        raise ValueError("Stage ablation requires resolved group ranks.")
    if any(int(ranks.get(task, 0)) != 0 for ranks in mtlora.R_PER_TASK_LIST for task in config.TASKS):
        raise ValueError("Stage ablation requires all task-specific LoRA ranks to be 0.")
    if mtlora.DOWNSAMPLER_ENABLED:
        raise ValueError("This ablation baseline requires backbone downsampler LoRA disabled.")
    return modes


def configure_stage_ablation(model, config):
    modes = resolve_stage_modes(config)
    if modes is None:
        return
    backbone = getattr(model, "backbone", model)
    backbone.stage_ablation_modes = tuple(modes)
    backbone.stage_prompt_enabled = tuple(mode != "G" for mode in modes)
    for stage, mode in enumerate(modes):
        backbone.set_shared_lora_enabled(stage, mode != "P")


def disabled_adapter_parameters(model):
    backbone = getattr(model, "backbone", model)
    modes = getattr(backbone, "stage_ablation_modes", None)
    if modes is None:
        return
    for stage, mode in enumerate(modes):
        if mode == "G":
            if stage == 0:
                yield from backbone.prompt_embeddings.parameters()
            yield from backbone.deep_prompt_embeddings[stage].parameters()
        elif mode == "P":
            # layers[i] includes its own PatchMerging; decoder downsamplers are excluded.
            for name, parameter in backbone.layers[stage].named_parameters():
                if "lora_shared_" in name:
                    yield parameter


def freeze_stage_ablation(model):
    for parameter in disabled_adapter_parameters(model):
        parameter.requires_grad_(False)
        parameter.grad = None


def prepare_trainable_parameters(model, config):
    from models.lora import mark_only_lora_as_trainable, mark_prompt_as_trainable

    backbone = getattr(model, "backbone", model)
    configure_stage_ablation(model, config)
    if config.MODEL.MTLORA.ENABLED and config.MODEL.MTLORA.FREEZE_PRETRAINED:
        mark_only_lora_as_trainable(
            backbone, bias=config.MODEL.MTLORA.BIAS,
            freeze_patch_embed=config.TRAIN.FREEZE_PATCH_EMBED,
            freeze_norm=config.TRAIN.FREEZE_LAYER_NORM,
            free_relative_bias=config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS,
            freeze_downsample_reduction=(config.MODEL.MTLORA.DOWNSAMPLER_ENABLED
                                         or config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION),
        )
        if config.MODEL.PROMPT.ENABLED:
            mark_prompt_as_trainable(backbone)
    if config.MODEL.FREEZE_BACKBONE:
        if config.MODEL.MTLORA.ENABLED:
            raise ValueError("FREEZE_BACKBONE cannot be combined with MTLoRA.")
        model.freeze_backbone()
    freeze_stage_ablation(model)


def stage_ablation_metadata(config):
    modes = resolve_stage_modes(config)
    if modes is None:
        return None
    m = config.MODEL.MTLORA
    grouping = {
        "granularity": m.AGMTLORA_PARTITION_GRANULARITY,
        "names": list(m.AGMTLORA_GROUP_NAMES),
        "ranks": list(m.AGMTLORA_GROUP_RANKS),
        "task_to_group": dict(m.AGMTLORA_TASK_TO_GROUP),
        "task_to_group_by_stage": {k: dict(v) for k, v in m.AGMTLORA_TASK_TO_GROUP_BY_STAGE.items()},
    }
    identity = hashlib.sha256(json.dumps(grouping, sort_keys=True).encode()).hexdigest()
    world_size = getattr(config.MODEL.STAGE_ABLATION, "WORLD_SIZE", 0)
    lr_factor = 1.0 if config.TRAIN.LR_SCALED else (
        config.DATA.BATCH_SIZE * (world_size or 1) * config.TRAIN.ACCUMULATION_STEPS / 512.0)
    protocol = {
        "seed": config.SEED, "batch_size": config.DATA.BATCH_SIZE,
        "world_size": world_size, "epochs": config.TRAIN.EPOCHS,
        "accumulation_steps": config.TRAIN.ACCUMULATION_STEPS,
        "effective_lrs": {key: float(config.TRAIN[key]) * lr_factor
                          for key in ("BASE_LR", "WARMUP_LR", "MIN_LR")},
        "weight_decay": config.TRAIN.WEIGHT_DECAY,
        "warmup_epochs": config.TRAIN.WARMUP_EPOCHS,
        "optimizer": dict(config.TRAIN.OPTIMIZER),
        "scheduler": dict(config.TRAIN.LR_SCHEDULER),
        "freeze": {key: bool(config.TRAIN[key]) for key in (
            "FREEZE_PATCH_EMBED", "FREEZE_LAYER_NORM", "FREEZE_DOWNSAMPLE_REDUCTION", "FREEZE_RELATIVE_POSITION_BIAS")},
        "amp": config.AMP_ENABLE, "deterministic": config.DETERMINISTIC,
        "eval_freq": config.EVAL_FREQ, "dataset": config.DATA.DBNAME,
    }
    return {"version": 1, "modes": modes, "grouping_id": identity,
            "grouping": grouping, "tasks": list(config.TASKS),
            "prompt_tokens": int(config.MODEL.PROMPT.NUM_TOKENS),
            "depths": list(config.MODEL.SWIN.DEPTHS),
            "initialization": getattr(config.MODEL, "INIT_CHECKPOINT_SHA256", "")
                              or getattr(config.MODEL, "INIT_CHECKPOINT", ""),
            "protocol": protocol}


def optimizer_parameter_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]


def validate_ablation_checkpoint(config, checkpoint, model=None, optimizer=None):
    expected = stage_ablation_metadata(config)
    actual = checkpoint.get("stage_ablation")
    if expected is None and actual is None:
        return
    # Evaluation may use a different batch size; adapter identity remains mandatory.
    compared_expected, compared_actual = expected, actual
    if optimizer is None and expected is not None and isinstance(actual, dict):
        compared_expected = {key: value for key, value in expected.items() if key != "protocol"}
        compared_actual = {key: value for key, value in actual.items() if key != "protocol"}
    if compared_expected != compared_actual:
        raise ValueError("Checkpoint stage ablation/grouping/initialization mismatch. "
                         "Use MODEL.INIT_CHECKPOINT for common weight-only initialization; "
                         "--resume is only for the same experiment.")
    if optimizer is not None:
        if checkpoint.get("optimizer_parameter_names") != optimizer_parameter_names(model, optimizer):
            raise ValueError("Checkpoint optimizer parameter groups are incompatible with this experiment.")
        if any(key not in checkpoint for key in ("optimizer", "lr_scheduler", "epoch", "scaler")):
            raise ValueError("Same-experiment resume requires complete optimizer/scheduler/epoch/scaler state.")


def adapter_parameter_summary(model):
    backbone = getattr(model, "backbone", model)
    disabled = {id(p) for p in disabled_adapter_parameters(model)}
    adapter = [(name, p) for name, p in model.named_parameters()
               if "lora_" in name or "prompt_embeddings" in name]
    modes = getattr(backbone, "stage_ablation_modes", None)
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "stored_adapter_parameters": sum(p.numel() for _, p in adapter),
        "enabled_adapter_parameters": sum(p.numel() for _, p in adapter if id(p) not in disabled),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stages": [{"stage": i + 1, "mode": mode,
                    "blocks": len(backbone.layers[i].blocks),
                    "prompt_tokens": 0 if mode == "G" else backbone.num_tokens,
                    "group_lora_enabled": mode != "P"}
                   for i, mode in enumerate(modes)] if modes else [],
    }
