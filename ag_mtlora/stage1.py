import ast
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Subset
from yacs.config import CfgNode as CN

from ag_mtlora.config_utils import (
    DEFAULT_PARTITION_GRANULARITY,
    STAGE1_RUNTIME_SCHEMA_VERSION,
    build_task_to_group_by_stage,
    build_task_to_group,
    canonicalize_groups,
    enumerate_candidate_groups,
    enumerate_partitions,
    group_display_name,
    normalize_partition_granularity,
    resolve_group_shared_ranks,
    resolve_stagewise_group_shared_ranks,
    select_predictor_train_groups,
)
from data import build_loader
from data.mtl_ds import (
    get_mtl_dataset,
    get_mtl_split_sample_ids,
    get_mtl_train_dataloader,
    get_mtl_val_dataloader,
    get_tasks_config,
    get_transformations,
)
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model, build_mtl_model
from models.lora import mark_only_lora_as_trainable, mark_prompt_as_trainable
from mtl_loss_schemes import MultiTaskLoss, get_loss
from optimizer import build_optimizer
from utils import (
    NativeScalerWithGradNormCount,
    load_checkpoint,
    load_pretrained,
    mkdir_if_missing,
)


DEFAULT_TASK_LOSS_WEIGHTS = {
    "depth": 1.0,
    "semseg": 1.0,
    "human_parts": 2.0,
    "sal": 5.0,
    "edge": 50.0,
    "normals": 10.0,
}

PREDICTOR_PROGRESS_FILENAME = "predictor_progress.json"
DEFAULT_SEARCH_SCORE_SOURCE = "final_predictions"
SUPPORTED_SEARCH_SCORE_SOURCES = {"final_predictions", "group_proxy"}
SHARED_PARAM_STAGE_PATTERN = re.compile(r"backbone\.layers\.(\d+)\.")


@dataclass
class Stage1TrainRuntime:
    optimizer: object
    lr_scheduler: object
    loss_scaler: object
    amp_enabled: bool
    clip_grad: Optional[float]
    global_step: int = 0
    amp_overflow_count: int = 0
    consecutive_amp_overflows: int = 0

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def loss_scale(self) -> float:
        state = self.loss_scaler.state_dict()
        return float(state.get("scale", 1.0))


@dataclass(frozen=True)
class Stage1StepResult:
    grad_norm: Optional[float]
    optimizer_step_skipped: bool
    loss_scale_before: float
    loss_scale_after: float


class Stage1NumericalError(RuntimeError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.details = dict(details or {})
        super().__init__(message)


def update_stage1_execution_context(execution_context: Optional[Dict], **updates) -> Dict:
    if execution_context is None:
        execution_context = {}
    execution_context.update({key: value for key, value in updates.items() if value is not None})
    return execution_context


def get_batch_sample_ids(batch) -> List[str]:
    if not isinstance(batch, dict):
        return []
    meta = batch.get("meta")
    if not isinstance(meta, dict) or "image" not in meta:
        return []
    image_ids = meta["image"]
    if isinstance(image_ids, (list, tuple)):
        return [str(value) for value in image_ids]
    return [str(image_ids)]


def get_target_valid_ratios(targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    ratios = {}
    for task, target in targets.items():
        if not torch.is_tensor(target) or target.numel() == 0:
            continue
        if task in {"semseg", "human_parts", "normals", "depth"}:
            ratios[task] = float((target != 255).float().mean().item())
        else:
            ratios[task] = 1.0
    return ratios


def _tensor_failure_stats(tensor: torch.Tensor) -> Dict:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_values = detached[finite_mask]
    stats = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "nonfinite_count": int((~finite_mask).sum().item()),
    }
    if finite_values.numel() > 0:
        stats.update({
            "finite_min": float(finite_values.min().item()),
            "finite_max": float(finite_values.max().item()),
            "finite_mean": float(finite_values.float().mean().item()),
        })
    return stats


def ensure_finite_tensor(tensor: torch.Tensor, tensor_name: str, execution_context: Optional[Dict] = None):
    if not torch.is_tensor(tensor) or not (tensor.is_floating_point() or tensor.is_complex()):
        return
    if bool(torch.isfinite(tensor).all().item()):
        return
    details = dict(execution_context or {})
    details.update({
        "tensor_name": tensor_name,
        "tensor_stats": _tensor_failure_stats(tensor),
    })
    raise Stage1NumericalError(
        f"Non-finite tensor detected during Stage-1: {tensor_name}",
        details=details,
    )


def ensure_finite_number(value, value_name: str, execution_context: Optional[Dict] = None, positive=False):
    numeric_value = float(value)
    if np.isfinite(numeric_value) and (not positive or numeric_value > 0.0):
        return
    details = dict(execution_context or {})
    details.update({"value_name": value_name, "value": numeric_value})
    qualifier = "positive finite" if positive else "finite"
    raise Stage1NumericalError(
        f"Stage-1 expected {value_name} to be {qualifier}, got {numeric_value}",
        details=details,
    )


def ensure_finite_outputs(outputs: Dict[str, torch.Tensor], execution_context: Optional[Dict] = None):
    for task, output in outputs.items():
        task_context = update_stage1_execution_context(dict(execution_context or {}), task=task)
        ensure_finite_tensor(output, f"outputs.{task}", task_context)


def find_first_nonfinite_state(model_or_state) -> Optional[Dict]:
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        if not bool(torch.isfinite(tensor).all().item()):
            return {"tensor_name": name, "tensor_stats": _tensor_failure_stats(tensor)}
    return None


def ensure_finite_model_state(model_or_state, execution_context: Optional[Dict] = None):
    failure = find_first_nonfinite_state(model_or_state)
    if failure is None:
        return
    details = dict(execution_context or {})
    details.update(failure)
    raise Stage1NumericalError(
        f"Non-finite model state detected during Stage-1: {failure['tensor_name']}",
        details=details,
    )


def atomic_torch_save(payload: Dict, output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    temporary_path = output_path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)


def save_stage1_last_good_checkpoint(
    working_dir: str,
    model: nn.Module,
    runtime: Stage1TrainRuntime,
    phase: str,
    epoch: int,
) -> str:
    execution_context = {"phase": phase, "epoch": int(epoch)}
    ensure_finite_model_state(model, execution_context)
    output_path = os.path.join(working_dir, "last_good_checkpoint.pth")
    atomic_torch_save(
        {
            "model": clone_state_dict_to_cpu(model),
            "optimizer": runtime.optimizer.state_dict(),
            "lr_scheduler": runtime.lr_scheduler.state_dict(),
            "scaler": runtime.loss_scaler.state_dict(),
            "global_step": int(runtime.global_step),
            "amp_overflow_count": int(runtime.amp_overflow_count),
            "consecutive_amp_overflows": int(runtime.consecutive_amp_overflows),
            "phase": phase,
            "epoch": int(epoch),
            "stage1_runtime_schema_version": STAGE1_RUNTIME_SCHEMA_VERSION,
        },
        output_path,
    )
    return output_path


def build_stage1_scheduler_config(config: CN, warmup_epochs: int, total_epochs: int) -> CN:
    scheduler_config = config.clone()
    scheduler_config.defrost()
    scheduler_config.TRAIN.EPOCHS = max(int(total_epochs), 1)
    scheduler_config.TRAIN.WARMUP_EPOCHS = max(0, min(int(warmup_epochs), int(total_epochs)))
    scheduler_config.freeze()
    return scheduler_config


def build_stage1_train_runtime(
    config: CN,
    model: nn.Module,
    steps_per_epoch: int,
    warmup_epochs: int,
    total_epochs: int,
) -> Stage1TrainRuntime:
    optimizer = build_optimizer(config, model)
    scheduler_config = build_stage1_scheduler_config(config, warmup_epochs, total_epochs)
    lr_scheduler = build_scheduler(scheduler_config, optimizer, max(int(steps_per_epoch), 1))
    clip_grad = float(config.TRAIN.CLIP_GRAD)
    if clip_grad <= 0:
        clip_grad = None
    return Stage1TrainRuntime(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_scaler=NativeScalerWithGradNormCount(),
        amp_enabled=bool(config.AMP_ENABLE and torch.cuda.is_available()),
        clip_grad=clip_grad,
    )


def stage1_optimizer_step(
    runtime: Stage1TrainRuntime,
    loss: torch.Tensor,
    model: nn.Module,
    execution_context: Optional[Dict] = None,
):
    ensure_finite_tensor(loss, "total_loss", execution_context)
    loss_scale_before = runtime.loss_scale
    grad_norm = runtime.loss_scaler(
        loss,
        runtime.optimizer,
        clip_grad=runtime.clip_grad,
        parameters=model.parameters(),
    )
    loss_scale_after = runtime.loss_scale
    ensure_finite_number(loss_scale_after, "loss_scale", execution_context, positive=True)

    optimizer_step_skipped = False
    finite_grad_norm = None
    if grad_norm is not None:
        grad_norm_is_finite = bool(torch.isfinite(grad_norm).all().item())
        amp_scale_backed_off = bool(
            runtime.amp_enabled and loss_scale_after < loss_scale_before
        )
        if not grad_norm_is_finite and amp_scale_backed_off:
            optimizer_step_skipped = True
            runtime.amp_overflow_count += 1
            runtime.consecutive_amp_overflows += 1
            overflow_context = update_stage1_execution_context(
                dict(execution_context or {}),
                loss_scale_before=loss_scale_before,
                loss_scale_after=loss_scale_after,
                amp_overflow_count=runtime.amp_overflow_count,
                consecutive_amp_overflows=runtime.consecutive_amp_overflows,
            )
            if runtime.consecutive_amp_overflows > 16 or loss_scale_after < 1.0:
                raise Stage1NumericalError(
                    "Repeated AMP gradient overflow did not stabilize during Stage-1.",
                    details=overflow_context,
                )
        else:
            ensure_finite_tensor(grad_norm, "grad_norm", execution_context)
            finite_grad_norm = float(grad_norm.detach().float().item())

    if not optimizer_step_skipped:
        runtime.consecutive_amp_overflows = 0
    runtime.lr_scheduler.step_update(runtime.global_step)
    runtime.global_step += 1
    runtime.optimizer.zero_grad()
    return Stage1StepResult(
        grad_norm=finite_grad_norm,
        optimizer_step_skipped=optimizer_step_skipped,
        loss_scale_before=loss_scale_before,
        loss_scale_after=loss_scale_after,
    )


def log_stage1_step_result(logger, result: Stage1StepResult, execution_context: Dict) -> str:
    if result.optimizer_step_skipped:
        logger.warning(
            "Recoverable AMP overflow; optimizer step skipped and loss scale backed off | phase=%s | epoch=%s | batch=%s | scale=%.6g->%.6g",
            execution_context.get("phase"),
            execution_context.get("epoch"),
            execution_context.get("batch"),
            result.loss_scale_before,
            result.loss_scale_after,
        )
        return "skipped_amp_overflow"
    if result.grad_norm is None:
        return "unavailable"
    return f"{result.grad_norm:.6f}"


def group_to_key(group: Sequence[str]) -> str:
    return "|".join(group)


def normalize_search_score_source(search_score_source: str) -> str:
    source = str(search_score_source or DEFAULT_SEARCH_SCORE_SOURCE)
    if source not in SUPPORTED_SEARCH_SCORE_SOURCES:
        raise ValueError(
            f"Unsupported SEARCH_SCORE_SOURCE: {source}. "
            f"Expected one of {sorted(SUPPORTED_SEARCH_SCORE_SOURCES)}."
        )
    return source


def get_search_score_source(config: CN) -> str:
    return normalize_search_score_source(
        getattr(config.MODEL.AGMTLORA, "SEARCH_SCORE_SOURCE", DEFAULT_SEARCH_SCORE_SOURCE)
    )


def get_partition_granularity(config: CN) -> str:
    return normalize_partition_granularity(
        getattr(config.MODEL.AGMTLORA, "PARTITION_GRANULARITY", DEFAULT_PARTITION_GRANULARITY)
    )


def append_search_score_suffix(path: str, search_score_source: str) -> str:
    source = normalize_search_score_source(search_score_source)
    if source == DEFAULT_SEARCH_SCORE_SOURCE:
        return path
    root, ext = os.path.splitext(path)
    source_suffix = f"__{source}"
    if root.endswith(source_suffix):
        return path
    return f"{root}__{source}{ext}"


def build_search_artifact_paths(output_root: str, grouping_save_path: str, search_score_source: str) -> Dict[str, str]:
    return {
        "partition_results_path": append_search_score_suffix(
            os.path.join(output_root, "partition_search_results.json"),
            search_score_source,
        ),
        "partition_results_csv": append_search_score_suffix(
            os.path.join(output_root, "partition_search_results.csv"),
            search_score_source,
        ),
        "grouping_json_path": append_search_score_suffix(grouping_save_path, search_score_source),
        "resolved_config_path": append_search_score_suffix(
            os.path.join(output_root, "resolved_agmtlora_config.yaml"),
            search_score_source,
        ),
        "resolved_runtime_snapshot_path": append_search_score_suffix(
            os.path.join(output_root, "resolved_agmtlora_runtime_snapshot.yaml"),
            search_score_source,
        ),
    }


def partition_to_key(groups: Sequence[Sequence[str]]) -> str:
    return ";".join(group_to_key(group) for group in groups)


def safe_num_batches(data_loader) -> int:
    try:
        return int(len(data_loader))
    except TypeError:
        return 0


def resolve_log_interval(total_steps: int, target_updates: int = 5) -> int:
    if total_steps <= 0:
        return 50
    return max(1, total_steps // max(int(target_updates), 1))


def maybe_log_progress(logger, phase: str, step_idx: int, total_steps: int, log_interval: int, extra: str = "") -> None:
    should_log = (
        step_idx == 1
        or step_idx == total_steps
        or (log_interval > 0 and step_idx % log_interval == 0)
    )
    if not should_log:
        return
    suffix = f" | {extra}" if extra else ""
    if total_steps > 0:
        logger.info("%s: %d/%d%s", phase, step_idx, total_steps, suffix)
    else:
        logger.info("%s: %d%s", phase, step_idx, suffix)


def round_value_map(values: Dict[str, float], digits: int = 6) -> Dict[str, float]:
    return {key: round(float(value), digits) for key, value in values.items()}


def average_search_scores_by_stage(
    search_scores_by_stage: Sequence[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    aggregated = {}
    counts = {}
    for stage_scores in search_scores_by_stage:
        for group_key, group_values in stage_scores.items():
            aggregated.setdefault(group_key, {})
            counts.setdefault(group_key, {})
            for task, value in group_values.items():
                aggregated[group_key][task] = aggregated[group_key].get(task, 0.0) + float(value)
                counts[group_key][task] = counts[group_key].get(task, 0) + 1
    return {
        group_key: {
            task: float(total / max(counts[group_key][task], 1))
            for task, total in group_values.items()
        }
        for group_key, group_values in aggregated.items()
    }


def get_affinity_warmup_epochs(config: CN) -> int:
    warmup_epochs = int(getattr(config.MODEL.AGMTLORA, "AFFINITY_WARMUP_EPOCHS", -1))
    if warmup_epochs >= 0:
        return warmup_epochs
    return int(config.MODEL.AGMTLORA.AFFINITY_COLLECT_EPOCHS)


def get_affinity_score_epochs(config: CN) -> int:
    return int(getattr(config.MODEL.AGMTLORA, "AFFINITY_SCORE_EPOCHS", 50))


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def save_json(payload: Dict, output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(output_path: str) -> Dict:
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_matrix_csv(tasks: Sequence[str], matrix: np.ndarray, output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task"] + list(tasks))
        for task, row in zip(tasks, matrix.tolist()):
            writer.writerow([task] + row)


def save_affinity_epoch_history_csv(tasks: Sequence[str], epoch_history: Sequence[np.ndarray], output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "source_task", "target_task", "value"])
        for epoch_idx, epoch_matrix in enumerate(epoch_history, start=1):
            epoch_matrix = np.asarray(epoch_matrix)
            for src_idx, src_task in enumerate(tasks):
                for dst_idx, dst_task in enumerate(tasks):
                    writer.writerow([epoch_idx, src_task, dst_task, float(epoch_matrix[src_idx, dst_idx])])


def save_group_task_rows(rows: Iterable[Dict], output_path: str, value_key: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "task", value_key])
        for row in rows:
            writer.writerow([row["group"], row["task"], row[value_key]])


def save_predictor_value_csv(value_by_group: Dict[str, Dict[str, float]], output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "task", "value"])
        for group_key, group_values in value_by_group.items():
            for task, value in group_values.items():
                writer.writerow([group_key, task, float(value)])


def save_partition_csv(ranked_partitions: List[Dict], output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "partition_score", "num_groups", "groups", "per_task_scores"])
        for rank, item in enumerate(ranked_partitions, start=1):
            writer.writerow(
                [
                    rank,
                    float(item["partition_score"]),
                    int(item["num_groups"]),
                    json.dumps(item["groups"], ensure_ascii=False),
                    json.dumps(item["per_task_scores"], ensure_ascii=False),
                ]
            )


def save_stage_partition_csv(ranked_partitions_by_stage: Sequence[Sequence[Dict]], output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "rank", "partition_score", "num_groups", "groups", "per_task_scores"])
        for stage_idx, ranked_partitions in enumerate(ranked_partitions_by_stage):
            for rank, item in enumerate(ranked_partitions, start=1):
                writer.writerow(
                    [
                        stage_idx,
                        rank,
                        float(item["partition_score"]),
                        int(item["num_groups"]),
                        json.dumps(item["groups"], ensure_ascii=False),
                        json.dumps(item["per_task_scores"], ensure_ascii=False),
                    ]
                )


def get_selection_data_split_mode(config: CN, data_split_manifest: Dict = None) -> str:
    if data_split_manifest is not None and data_split_manifest.get("selection_data_split_mode"):
        return str(data_split_manifest["selection_data_split_mode"])
    return str(getattr(config.MODEL.AGMTLORA, "DATA_SPLIT_MODE", "official_val"))


def get_selection_train_label(config: CN, data_split_manifest: Dict = None) -> str:
    if get_selection_data_split_mode(config, data_split_manifest) == "train_meta_strict":
        return "meta-train"
    return "train"


def get_selection_eval_label(config: CN, data_split_manifest: Dict = None) -> str:
    if get_selection_data_split_mode(config, data_split_manifest) == "train_meta_strict":
        return "meta-val"
    return "official val"


def get_effective_meta_split_seed(config: CN) -> int:
    return int(getattr(config.MODEL.AGMTLORA, "RESOLVED_META_SPLIT_SEED", config.SEED))


def get_dataset_sample_ids(dataset) -> List[str]:
    sample_ids = getattr(dataset, "im_ids", None)
    if sample_ids is None:
        raise ValueError("Stage-1 meta split requires datasets to expose stable sample IDs via `im_ids`.")
    return [str(sample_id) for sample_id in sample_ids]


def get_predictor_progress_path(working_dir: str) -> str:
    return os.path.join(working_dir, PREDICTOR_PROGRESS_FILENAME)


def parse_predictor_progress_from_log(log_path: str):
    singleton_losses = {}
    predictor_targets = {}
    if not os.path.exists(log_path):
        return predictor_targets, singleton_losses

    singleton_pattern = re.compile(r"Predictor target collected \| singleton task=(?P<task>[^|]+) \| val_loss=(?P<loss>[-+0-9.eE]+)")
    group_pattern = re.compile(r"Predictor target collected \| group=(?P<group>\[.*?\]) \| gains=(?P<gains>\{.*\})")

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            singleton_match = singleton_pattern.search(line)
            if singleton_match:
                task = singleton_match.group("task").strip()
                singleton_losses[task] = float(singleton_match.group("loss"))
                predictor_targets[group_to_key([task])] = {task: 0.0}
                continue

            group_match = group_pattern.search(line)
            if not group_match:
                continue
            try:
                group = list(ast.literal_eval(group_match.group("group")))
                gains = dict(ast.literal_eval(group_match.group("gains")))
            except (SyntaxError, ValueError):
                continue
            predictor_targets[group_to_key(group)] = {
                str(task): float(value) for task, value in gains.items()
            }

    return predictor_targets, singleton_losses


def load_predictor_progress(working_dir: str, logger):
    progress_path = get_predictor_progress_path(working_dir)
    if os.path.exists(progress_path):
        payload = load_json(progress_path)
        predictor_targets = {
            str(group_key): {str(task): float(value) for task, value in group_values.items()}
            for group_key, group_values in payload.get("predictor_targets", {}).items()
        }
        singleton_losses = {
            str(task): float(value) for task, value in payload.get("singleton_losses", {}).items()
        }
        logger.info(
            "Loaded predictor progress from %s | singleton_groups=%d | total_groups=%d",
            progress_path,
            len(singleton_losses),
            len(predictor_targets),
        )
        return predictor_targets, singleton_losses

    log_path = os.path.join(working_dir, "log_rank0.txt")
    predictor_targets, singleton_losses = parse_predictor_progress_from_log(log_path)
    if predictor_targets or singleton_losses:
        logger.info(
            "Recovered predictor progress from %s | singleton_groups=%d | total_groups=%d",
            log_path,
            len(singleton_losses),
            len(predictor_targets),
        )
    return predictor_targets, singleton_losses


def save_predictor_progress(
    working_dir: str,
    tasks: Sequence[str],
    train_groups: Sequence[Sequence[str]],
    singleton_losses: Dict[str, float],
    predictor_targets: Dict[str, Dict[str, float]],
    selection_data_split_mode: str,
    selection_eval_label: str,
    meta_split_path: str,
) -> None:
    progress_payload = {
        "tasks": list(tasks),
        "selection_data_split_mode": str(selection_data_split_mode),
        "selection_eval_label": str(selection_eval_label),
        "meta_split_path": meta_split_path,
        "train_groups": [list(group) for group in train_groups],
        "singleton_losses": round_value_map(singleton_losses),
        "predictor_targets": {
            group_key: round_value_map(group_values)
            for group_key, group_values in predictor_targets.items()
        },
    }
    save_json(progress_payload, get_predictor_progress_path(working_dir))


def maybe_load_affinity_result_from_existing_artifacts(config: CN, output_root: str, logger):
    partition_granularity = get_partition_granularity(config)
    affinity_json_path = config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH
    affinity_epoch_history_json = os.path.join(os.path.dirname(affinity_json_path), "affinity_epoch_history.json")
    warmup_checkpoint_path = os.path.join(output_root, "warmup_checkpoint.pth")
    post_affinity_checkpoint_path = os.path.join(output_root, "post_affinity_checkpoint.pth")
    required_paths = [
        affinity_json_path,
        affinity_epoch_history_json,
        warmup_checkpoint_path,
        post_affinity_checkpoint_path,
    ]
    if not all(os.path.exists(path) for path in required_paths):
        return None

    affinity_payload = load_json(affinity_json_path)
    epoch_history_payload = load_json(affinity_epoch_history_json)
    directed_affinity = np.asarray(affinity_payload["directed_affinity"], dtype=np.float32)
    directed_affinity_by_stage_payload = affinity_payload.get("directed_affinity_by_stage", [])
    if partition_granularity == "stage" and not directed_affinity_by_stage_payload:
        return None
    directed_affinity_by_stage = [
        np.asarray(matrix, dtype=np.float32)
        for matrix in directed_affinity_by_stage_payload
    ]
    affinity_epoch_history = [
        np.asarray(matrix, dtype=np.float32)
        for matrix in epoch_history_payload.get("epoch_directed_affinity", [])
    ]
    warmup_checkpoint = torch.load(warmup_checkpoint_path, map_location="cpu")
    post_affinity_checkpoint = torch.load(post_affinity_checkpoint_path, map_location="cpu")
    post_extra_state = post_affinity_checkpoint.get("extra_state", {})
    runtime_schema_version = int(post_extra_state.get("stage1_runtime_schema_version", 0))
    if runtime_schema_version != STAGE1_RUNTIME_SCHEMA_VERSION:
        raise RuntimeError(
            "Refusing to resume legacy Stage-1 affinity artifacts without the "
            f"numerical-stability contract (found schema={runtime_schema_version}, "
            f"required={STAGE1_RUNTIME_SCHEMA_VERSION}). Start a clean Stage-1 run "
            "from the original backbone checkpoint."
        )
    if not isinstance(post_extra_state.get("amp_smoke_test"), dict):
        raise RuntimeError(
            "Refusing to resume Stage-1 artifacts that do not record a passed AMP smoke test."
        )
    ensure_finite_model_state(
        warmup_checkpoint["model"],
        {"phase": "resume_artifact_validation", "checkpoint": warmup_checkpoint_path},
    )
    ensure_finite_model_state(
        post_affinity_checkpoint["model"],
        {"phase": "resume_artifact_validation", "checkpoint": post_affinity_checkpoint_path},
    )
    matrices_to_validate = [("directed_affinity", directed_affinity)]
    matrices_to_validate.extend(
        (f"directed_affinity_by_stage.{stage_idx}", matrix)
        for stage_idx, matrix in enumerate(directed_affinity_by_stage)
    )
    matrices_to_validate.extend(
        (f"affinity_epoch_history.{epoch_idx}", matrix)
        for epoch_idx, matrix in enumerate(affinity_epoch_history)
    )
    for matrix_name, matrix in matrices_to_validate:
        if not bool(np.isfinite(matrix).all()):
            raise Stage1NumericalError(
                f"Non-finite resumed Stage-1 artifact detected: {matrix_name}",
                details={
                    "phase": "resume_artifact_validation",
                    "artifact_name": matrix_name,
                },
            )
    for loss_group in ["warmup_validation_losses", "post_affinity_validation_losses"]:
        for task, value in affinity_payload.get(loss_group, {}).items():
            ensure_finite_number(
                value,
                f"{loss_group}.{task}",
                {"phase": "resume_artifact_validation", "task": task},
            )
    logger.info(
        "Resuming validated Stage-1 affinity artifacts | runtime_schema=%d | affinity_json=%s | post_affinity_checkpoint=%s",
        runtime_schema_version,
        affinity_json_path,
        post_affinity_checkpoint_path,
    )
    return {
        "warmup_checkpoint_path": warmup_checkpoint_path,
        "warmup_state_dict": warmup_checkpoint["model"],
        "warmup_validation_losses": affinity_payload.get("warmup_validation_losses", {}),
        "post_affinity_checkpoint_path": post_affinity_checkpoint_path,
        "post_affinity_state_dict": post_affinity_checkpoint["model"],
        "post_affinity_validation_losses": affinity_payload.get("post_affinity_validation_losses", {}),
        "directed_affinity": directed_affinity,
        "directed_affinity_by_stage": directed_affinity_by_stage,
        "symmetric_affinity": 0.5 * (directed_affinity + directed_affinity.T),
        "affinity_epoch_history": affinity_epoch_history,
        "num_batches_per_epoch": list(affinity_payload.get("num_batches_per_epoch", [])),
        "num_affinity_epochs": int(affinity_payload.get("num_affinity_epochs", len(affinity_epoch_history))),
        "validation_losses": affinity_payload.get("post_affinity_validation_losses", {}),
        "selection_data_split_mode": str(affinity_payload.get("selection_data_split_mode", get_selection_data_split_mode(config))),
        "selection_eval_label": str(affinity_payload.get("selection_eval_label", get_selection_eval_label(config))),
        "meta_split_path": affinity_payload.get("meta_split_path"),
        "num_batches": int(sum(affinity_payload.get("num_batches_per_epoch", []))),
    }


def build_stage1_data_split_manifest(config: CN, logger) -> Dict:
    selection_data_split_mode = get_selection_data_split_mode(config)
    meta_split_path = None
    if selection_data_split_mode != "train_meta_strict":
        logger.info(
            "Stage-1 selection data split | mode=%s | train_split=%s | selection_eval=%s",
            selection_data_split_mode,
            get_selection_train_label(config),
            get_selection_eval_label(config),
        )
        return {
            "selection_data_split_mode": selection_data_split_mode,
            "selection_train_label": get_selection_train_label(config),
            "selection_eval_label": get_selection_eval_label(config),
            "meta_split_path": meta_split_path,
        }

    sample_ids = get_mtl_split_sample_ids(config.DATA.DBNAME, config.DATA.DATA_PATH, split="train")
    num_samples = len(sample_ids)
    if num_samples < 2:
        raise ValueError("AG-MTLoRA Stage-1 strict meta split requires at least 2 training samples.")

    meta_val_ratio = float(config.MODEL.AGMTLORA.META_VAL_RATIO)
    effective_meta_split_seed = get_effective_meta_split_seed(config)
    shuffled_indices = list(range(num_samples))
    random.Random(effective_meta_split_seed).shuffle(shuffled_indices)
    meta_val_count = int(round(num_samples * meta_val_ratio))
    meta_val_count = min(max(meta_val_count, 1), num_samples - 1)
    meta_val_indices = sorted(shuffled_indices[:meta_val_count])
    meta_train_indices = sorted(shuffled_indices[meta_val_count:])
    meta_split_path = config.MODEL.AGMTLORA.META_SPLIT_SAVE_PATH

    manifest = {
        "dataset_name": str(config.DATA.DBNAME),
        "selection_data_split_mode": selection_data_split_mode,
        "selection_train_label": get_selection_train_label(config),
        "selection_eval_label": get_selection_eval_label(config),
        "meta_val_ratio": meta_val_ratio,
        "effective_meta_split_seed": effective_meta_split_seed,
        "num_samples": int(num_samples),
        "meta_train_indices": meta_train_indices,
        "meta_val_indices": meta_val_indices,
        "meta_train_ids": [sample_ids[index] for index in meta_train_indices],
        "meta_val_ids": [sample_ids[index] for index in meta_val_indices],
        "meta_split_path": meta_split_path,
    }
    save_json(manifest, meta_split_path)
    logger.info(
        "Stage-1 selection data split prepared | mode=%s | dataset=%s | num_samples=%d | meta_train=%d | meta_val=%d | meta_split_path=%s",
        selection_data_split_mode,
        str(config.DATA.DBNAME),
        int(num_samples),
        len(meta_train_indices),
        len(meta_val_indices),
        meta_split_path,
    )
    return manifest


def build_stage1_data_loaders(config: CN, data_split_manifest: Dict = None):
    selection_data_split_mode = get_selection_data_split_mode(config, data_split_manifest)
    if selection_data_split_mode != "train_meta_strict":
        return build_loader(config)

    if data_split_manifest is None:
        raise ValueError("Strict Stage-1 data loading requires a prepared meta split manifest.")

    train_transforms, eval_transforms = get_transformations(config.DATA.DBNAME, config.TASKS_CONFIG)
    meta_train_dataset_full = get_mtl_dataset(config.DATA.DBNAME, config, train_transforms, split="train")
    meta_val_dataset_full = get_mtl_dataset(config.DATA.DBNAME, config, eval_transforms, split="train")
    meta_train_sample_ids = get_dataset_sample_ids(meta_train_dataset_full)
    meta_val_sample_ids = get_dataset_sample_ids(meta_val_dataset_full)
    if meta_train_sample_ids != meta_val_sample_ids:
        raise ValueError("Train-transform and eval-transform views of the train split must preserve sample ordering.")

    expected_meta_train_ids = [str(sample_id) for sample_id in data_split_manifest.get("meta_train_ids", [])]
    expected_meta_val_ids = [str(sample_id) for sample_id in data_split_manifest.get("meta_val_ids", [])]
    index_by_sample_id = {sample_id: idx for idx, sample_id in enumerate(meta_train_sample_ids)}
    meta_train_indices = [index_by_sample_id[sample_id] for sample_id in expected_meta_train_ids if sample_id in index_by_sample_id]
    meta_val_indices = [index_by_sample_id[sample_id] for sample_id in expected_meta_val_ids if sample_id in index_by_sample_id]
    resolved_meta_train_ids = [meta_train_sample_ids[index] for index in meta_train_indices]
    resolved_meta_val_ids = [meta_train_sample_ids[index] for index in meta_val_indices]
    filtered_expected_meta_train_ids = [sample_id for sample_id in expected_meta_train_ids if sample_id in index_by_sample_id]
    filtered_expected_meta_val_ids = [sample_id for sample_id in expected_meta_val_ids if sample_id in index_by_sample_id]
    if filtered_expected_meta_train_ids != resolved_meta_train_ids:
        raise ValueError("Meta-train split manifest does not match the current dataset sample ordering.")
    if filtered_expected_meta_val_ids != resolved_meta_val_ids:
        raise ValueError("Meta-val split manifest does not match the current dataset sample ordering.")
    if len(meta_train_indices) == 0 or len(meta_val_indices) == 0:
        raise ValueError(
            "Stage-1 strict meta split projected to an empty subset for the current task configuration. "
            "Adjust META_VAL_RATIO or META_SPLIT_SEED."
        )

    dataset_train = Subset(meta_train_dataset_full, meta_train_indices)
    dataset_val = Subset(meta_val_dataset_full, meta_val_indices)
    data_loader_train = get_mtl_train_dataloader(config, dataset_train)
    data_loader_val = get_mtl_val_dataloader(config, dataset_val)
    return dataset_train, dataset_val, data_loader_train, data_loader_val, None


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def rebuild_mtlora_task_lists(config: CN) -> None:
    if not config.MODEL.MTLORA.ENABLED:
        return

    num_stages = len(config.MODEL.SWIN.DEPTHS)
    config.MODEL.MTLORA.R_PER_TASK_LIST = []
    config.MODEL.MTLORA.SCALE_PER_TASK_LIST = []
    for stage_idx in range(num_stages):
        layer_task_r = {
            "shared": (
                config.MODEL.MTLORA.R_PER_TASK["shared"][stage_idx]
                if "shared" in config.MODEL.MTLORA.R_PER_TASK
                else config.MODEL.MTLORA.R[stage_idx]
            )
        }
        layer_task_scale = {}
        for task in config.TASKS:
            layer_task_r[task] = config.MODEL.MTLORA.R_PER_TASK[task][stage_idx]
            layer_task_scale[task] = config.MODEL.MTLORA.SCALE_PER_TASK[task][stage_idx]
        config.MODEL.MTLORA.R_PER_TASK_LIST.append(layer_task_r)
        config.MODEL.MTLORA.SCALE_PER_TASK_LIST.append(layer_task_scale)


def rebuild_task_config(base_config: CN, tasks: Sequence[str], output_dir: str = None) -> CN:
    config = base_config.clone()
    config.defrost()
    config.TASKS = list(tasks)
    config.MTL = True
    task_cfg, _ = get_tasks_config(config.DATA.DBNAME, config.TASKS, config.DATA.IMG_SIZE)
    task_cfg = dict(task_cfg)
    config.TASKS_CONFIG = CN(task_cfg)
    config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT = CN(dict(config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT))
    config.TASKS_CONFIG.ALL_TASKS.FLAGVALS = CN(dict(config.TASKS_CONFIG.ALL_TASKS.FLAGVALS))
    config.TASKS_CONFIG.ALL_TASKS.INFER_FLAGVALS = CN(dict(config.TASKS_CONFIG.ALL_TASKS.INFER_FLAGVALS))

    if config.MODEL.MTLORA.ENABLED:
        filtered_r_per_task = CN(new_allowed=True)
        filtered_scale_per_task = CN(new_allowed=True)
        for task in config.TASKS:
            filtered_r_per_task[task] = list(base_config.MODEL.MTLORA.R_PER_TASK[task])
            filtered_scale_per_task[task] = list(base_config.MODEL.MTLORA.SCALE_PER_TASK[task])
        if "shared" in base_config.MODEL.MTLORA.R_PER_TASK:
            filtered_r_per_task["shared"] = list(base_config.MODEL.MTLORA.R_PER_TASK["shared"])
        config.MODEL.MTLORA.R_PER_TASK = filtered_r_per_task
        config.MODEL.MTLORA.SCALE_PER_TASK = filtered_scale_per_task
        rebuild_mtlora_task_lists(config)

    config.MODEL.AGMTLORA.ENABLED = False
    config.MODEL.MTLORA.AGMTLORA_ENABLED = False
    config.MODEL.MTLORA.AGMTLORA_STAGE = 0
    config.MODEL.MTLORA.AGMTLORA_PARTITION_GRANULARITY = "global"
    config.MODEL.MTLORA.AGMTLORA_GROUPS = []
    config.MODEL.MTLORA.AGMTLORA_GROUP_NAMES = []
    config.MODEL.MTLORA.AGMTLORA_GROUP_RANKS = []
    config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP = CN(new_allowed=True)
    config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)

    if output_dir is not None:
        config.OUTPUT = output_dir
    config.freeze()
    return config


def build_loss_bundle(config: CN):
    loss_ft = nn.ModuleDict(
        {task: get_loss(config["TASKS_CONFIG"], task, config) for task in config.TASKS}
    )
    loss_weights = {task: DEFAULT_TASK_LOSS_WEIGHTS[task] for task in config.TASKS}
    criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)
    return criterion, loss_ft, loss_weights


def move_batch_to_device(batch: Dict, tasks: Sequence[str], device: torch.device):
    samples = batch["image"].to(device, non_blocking=True)
    targets = {task: batch[task].to(device, non_blocking=True) for task in tasks}
    return samples, targets


def maybe_load_initial_weights(config: CN, model: nn.Module, logger) -> None:
    old_eval_mode = bool(config.EVAL_MODE)
    config.defrost()
    config.EVAL_MODE = True
    config.freeze()
    try:
        if config.MODEL.RESUME:
            load_checkpoint(config, model, None, None, None, logger, quiet=True)
        elif config.MODEL.RESUME_BACKBONE:
            target_model = model.backbone if hasattr(model, "backbone") else model
            load_checkpoint(config, target_model, None, None, None, logger, backbone=True, quiet=True)
        elif config.MODEL.PRETRAINED:
            load_pretrained(config, model, logger)
    finally:
        config.defrost()
        config.EVAL_MODE = old_eval_mode
        config.freeze()


def build_task_model(config: CN, device: torch.device, logger, init_state_dict: Dict[str, torch.Tensor] = None):
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)
    else:
        maybe_load_initial_weights(config, model, logger)

    model.to(device)
    if config.MODEL.MTLORA.ENABLED and config.MODEL.MTLORA.FREEZE_PRETRAINED:
        mark_only_lora_as_trainable(
            model.backbone,
            bias=config.MODEL.MTLORA.BIAS,
            freeze_patch_embed=config.TRAIN.FREEZE_PATCH_EMBED,
            freeze_norm=config.TRAIN.FREEZE_LAYER_NORM,
            free_relative_bias=config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS,
            freeze_downsample_reduction=(
                True if config.MODEL.MTLORA.DOWNSAMPLER_ENABLED else config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION
            ),
        )
        if bool(getattr(getattr(config.MODEL, "PROMPT", None), "ENABLED", False)):
            mark_prompt_as_trainable(model.backbone)
    return model


def evaluate_task_losses(
    model,
    data_loader,
    loss_ft,
    tasks,
    device: torch.device,
    max_batches: int = None,
    amp_enabled: bool = False,
    execution_context: Optional[Dict] = None,
):
    model.eval()
    aggregated = {task: 0.0 for task in tasks}
    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            samples, targets = move_batch_to_device(batch, tasks, device)
            batch_context = update_stage1_execution_context(
                execution_context,
                batch=batch_idx,
                sample_ids=get_batch_sample_ids(batch),
                valid_label_ratios=get_target_valid_ratios(targets),
            )
            with torch.cuda.amp.autocast(enabled=bool(amp_enabled)):
                outputs = model(samples)
                task_losses = {
                    task: loss_ft[task](outputs[task], targets[task])
                    for task in tasks
                }
            ensure_finite_outputs(outputs, batch_context)
            for task, task_loss in task_losses.items():
                ensure_finite_tensor(
                    task_loss,
                    f"task_losses.{task}",
                    update_stage1_execution_context(dict(batch_context), task=task),
                )
                aggregated[task] += float(task_loss.item())
            num_batches += 1
    if num_batches == 0:
        return {task: 0.0 for task in tasks}
    return {task: value / float(num_batches) for task, value in aggregated.items()}


def run_stage1_amp_smoke_test(
    model: nn.Module,
    batch,
    criterion,
    tasks: Sequence[str],
    device: torch.device,
    amp_enabled: bool,
    execution_context: Optional[Dict] = None,
) -> Dict:
    """Exercise a train-mode forward/backward without mutating the saved state."""
    snapshot = clone_state_dict_to_cpu(model)
    was_training = model.training
    context = update_stage1_execution_context(
        execution_context,
        phase="post_affinity_amp_smoke",
        sample_ids=get_batch_sample_ids(batch),
    )
    try:
        model.train()
        samples, targets = move_batch_to_device(batch, tasks, device)
        context = update_stage1_execution_context(
            context,
            valid_label_ratios=get_target_valid_ratios(targets),
        )
        with torch.cuda.amp.autocast(enabled=bool(amp_enabled)):
            outputs = model(samples)
            loss, loss_dict = criterion(outputs, targets)

        ensure_finite_outputs(outputs, context)
        for task in tasks:
            ensure_finite_tensor(
                loss_dict[task],
                f"task_losses.{task}",
                update_stage1_execution_context(dict(context), task=task),
            )
        ensure_finite_tensor(loss, "total_loss", context)

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(loss, trainable, allow_unused=True)
        checked_gradients = 0
        for parameter_idx, gradient in enumerate(gradients):
            if gradient is None:
                continue
            ensure_finite_tensor(gradient, f"gradients.{parameter_idx}", context)
            checked_gradients += 1
        ensure_finite_model_state(model, context)
        return {
            "amp_enabled": bool(amp_enabled),
            "loss": float(loss.item()),
            "task_losses": {
                task: float(loss_dict[task].item())
                for task in tasks
            },
            "checked_gradients": checked_gradients,
            "sample_ids": list(context.get("sample_ids", [])),
            "valid_label_ratios": dict(context.get("valid_label_ratios", {})),
        }
    finally:
        model.load_state_dict(snapshot, strict=True)
        model.train(was_training)
        model.zero_grad(set_to_none=True)


def short_train_model(
    config: CN,
    logger,
    init_state_dict: Dict[str, torch.Tensor],
    num_epochs: int,
    device: torch.device,
    data_split_manifest: Dict = None,
):
    dataset_train, dataset_val, data_loader_train, data_loader_val, _ = build_stage1_data_loaders(
        config,
        data_split_manifest=data_split_manifest,
    )
    model = build_task_model(config, device, logger, init_state_dict=init_state_dict)
    criterion, loss_ft, _ = build_loss_bundle(config)
    optimizer = build_optimizer(config, model)
    num_train_batches = safe_num_batches(data_loader_train)
    num_val_batches = safe_num_batches(data_loader_val)
    train_log_interval = resolve_log_interval(num_train_batches)
    selection_train_label = get_selection_train_label(config, data_split_manifest)
    selection_eval_label = get_selection_eval_label(config, data_split_manifest)
    selection_loss_label = selection_eval_label.replace(" ", "_").replace("-", "_")

    logger.info(
        "Predictor short-train start | tasks=%s | epochs=%d | train_split=%s | selection_eval=%s | train_batches=%d | selection_batches=%d | output=%s",
        list(config.TASKS),
        int(num_epochs),
        selection_train_label,
        selection_eval_label,
        num_train_batches,
        num_val_batches,
        config.OUTPUT,
    )

    model.train()
    for epoch_idx in range(int(num_epochs)):
        logger.info(
            "Predictor short-train epoch %d/%d | tasks=%s",
            epoch_idx + 1,
            int(num_epochs),
            list(config.TASKS),
        )
        for batch_idx, batch in enumerate(data_loader_train, start=1):
            samples, targets = move_batch_to_device(batch, config.TASKS, device)
            outputs = model(samples)
            loss, _ = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            maybe_log_progress(
                logger,
                f"Predictor short-train epoch {epoch_idx + 1}/{int(num_epochs)}",
                batch_idx,
                num_train_batches,
                train_log_interval,
                extra=f"loss={float(loss.item()):.6f}",
            )

    selection_losses = evaluate_task_losses(model, data_loader_val, loss_ft, config.TASKS, device)
    logger.info(
        "Predictor short-train finished | tasks=%s | %s_losses=%s",
        list(config.TASKS),
        selection_loss_label,
        round_value_map(selection_losses),
    )
    trained_state = clone_state_dict_to_cpu(model)
    del dataset_train, dataset_val, data_loader_train, data_loader_val
    del optimizer, criterion, loss_ft, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trained_state, selection_losses


def get_shared_ta_parameters(model: nn.Module):
    params = []
    for name, param in model.named_parameters():
        if "backbone" not in name:
            continue
        if "lora_shared_" not in name:
            continue
        if "groups" in name:
            continue
        params.append(param)
    return params


def get_shared_ta_parameters_and_stage_slices(model: nn.Module, num_stages: int):
    params = []
    stage_slices = [[] for _ in range(int(num_stages))]
    flat_offset = 0
    for name, param in model.named_parameters():
        if "backbone" not in name:
            continue
        if "lora_shared_" not in name:
            continue
        if "groups" in name:
            continue
        match = SHARED_PARAM_STAGE_PATTERN.search(name)
        if match is None:
            raise ValueError(f"Could not infer the Swin stage for shared TA-LoRA parameter: {name}")
        stage_idx = int(match.group(1))
        if stage_idx >= int(num_stages):
            raise ValueError(
                f"Shared TA-LoRA parameter {name} resolved to stage {stage_idx}, "
                f"which exceeds the configured stage count {num_stages}."
            )
        params.append(param)
        stage_slices[stage_idx].append(slice(flat_offset, flat_offset + int(param.numel())))
        flat_offset += int(param.numel())
    return params, stage_slices


def flatten_gradient_list(params, grads):
    flattened = []
    for param, grad in zip(params, grads):
        if grad is None:
            flattened.append(torch.zeros_like(param).reshape(-1))
        else:
            flattened.append(grad.detach().reshape(-1))
    return torch.cat(flattened, dim=0)


def dot_on_flat_slices(lhs: torch.Tensor, rhs: torch.Tensor, slices: Sequence[slice]) -> torch.Tensor:
    total = lhs.new_zeros(())
    for flat_slice in slices:
        total = total + torch.dot(lhs[flat_slice], rhs[flat_slice])
    return total


def build_pseudo_update(flat_grad: torch.Tensor, optimizer) -> torch.Tensor:
    lr = float(optimizer.param_groups[0]["lr"])
    momentum = float(optimizer.param_groups[0].get("momentum", 0.0))
    if momentum <= 0.0:
        return lr * flat_grad

    buffers = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            state = optimizer.state.get(param, {})
            if "momentum_buffer" in state:
                buffers.append(state["momentum_buffer"].detach().reshape(-1))
            else:
                buffers.append(torch.zeros_like(param).reshape(-1))
    if len(buffers) == 0:
        return lr * flat_grad
    momentum_buffer = torch.cat(buffers, dim=0).to(flat_grad.device)
    if momentum_buffer.shape[0] != flat_grad.shape[0]:
        return lr * flat_grad
    return lr * (flat_grad + momentum * momentum_buffer)


def warmup_and_collect_affinity(
    config: CN,
    logger,
    working_dir: str,
    data_split_manifest: Dict = None,
    execution_context: Optional[Dict] = None,
):
    execution_context = update_stage1_execution_context(
        execution_context,
        phase="stage1_initialization",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partition_granularity = get_partition_granularity(config)
    num_stages = len(config.MODEL.SWIN.DEPTHS)
    dataset_train, dataset_val, data_loader_train, data_loader_val, _ = build_stage1_data_loaders(
        config,
        data_split_manifest=data_split_manifest,
    )
    model = build_task_model(config, device, logger)
    criterion, loss_ft, _ = build_loss_bundle(config)
    num_train_batches = safe_num_batches(data_loader_train)
    num_val_batches = safe_num_batches(data_loader_val)
    train_log_interval = resolve_log_interval(num_train_batches)
    warmup_epochs = get_affinity_warmup_epochs(config)
    affinity_score_epochs = get_affinity_score_epochs(config)
    runtime = build_stage1_train_runtime(
        config,
        model,
        steps_per_epoch=num_train_batches,
        warmup_epochs=warmup_epochs,
        total_epochs=warmup_epochs + affinity_score_epochs,
    )
    optimizer = runtime.optimizer
    selection_train_label = get_selection_train_label(config, data_split_manifest)
    selection_eval_label = get_selection_eval_label(config, data_split_manifest)
    selection_loss_label = selection_eval_label.replace(" ", "_").replace("-", "_")
    selection_data_split_mode = get_selection_data_split_mode(config, data_split_manifest)
    meta_split_path = None if data_split_manifest is None else data_split_manifest.get("meta_split_path")

    logger.info(
        "Stage-1 Step A start | tasks=%s | warmup_epochs=%d | affinity_score_epochs=%d | train_split=%s | selection_eval=%s | train_batches=%d | selection_batches=%d | base_lr=%.10g | amp=%s | clip_grad=%s | output=%s",
        list(config.TASKS),
        warmup_epochs,
        affinity_score_epochs,
        selection_train_label,
        selection_eval_label,
        num_train_batches,
        num_val_batches,
        float(config.TRAIN.BASE_LR),
        runtime.amp_enabled,
        runtime.clip_grad,
        working_dir,
    )

    if warmup_epochs > 0:
        for epoch_idx in range(warmup_epochs):
            execution_context = update_stage1_execution_context(
                execution_context,
                phase="warmup",
                epoch=epoch_idx,
            )
            logger.info("Warmup epoch %d/%d started", epoch_idx + 1, warmup_epochs)
            model.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(data_loader_train, start=1):
                samples, targets = move_batch_to_device(batch, config.TASKS, device)
                execution_context = update_stage1_execution_context(
                    execution_context,
                    batch=batch_idx - 1,
                    sample_ids=get_batch_sample_ids(batch),
                    valid_label_ratios=get_target_valid_ratios(targets),
                    learning_rate=runtime.learning_rate,
                    loss_scale=runtime.loss_scale,
                )
                with torch.cuda.amp.autocast(enabled=runtime.amp_enabled):
                    outputs = model(samples)
                    loss, loss_dict = criterion(outputs, targets)
                execution_context = update_stage1_execution_context(
                    execution_context,
                    task_losses={
                        task: float(loss_dict[task].detach().float().item())
                        for task in config.TASKS
                    },
                    total_loss=float(loss.detach().float().item()),
                )
                ensure_finite_outputs(outputs, execution_context)
                for task in config.TASKS:
                    ensure_finite_tensor(
                        loss_dict[task],
                        f"task_losses.{task}",
                        update_stage1_execution_context(dict(execution_context), task=task),
                    )
                step_result = stage1_optimizer_step(runtime, loss, model, execution_context)
                execution_context = update_stage1_execution_context(
                    execution_context,
                    loss_scale=step_result.loss_scale_after,
                    amp_overflow_count=runtime.amp_overflow_count,
                    optimizer_step_skipped=step_result.optimizer_step_skipped,
                )
                grad_norm_text = log_stage1_step_result(
                    logger,
                    step_result,
                    execution_context,
                )
                maybe_log_progress(
                    logger,
                    f"Warmup epoch {epoch_idx + 1}/{warmup_epochs}",
                    batch_idx,
                    num_train_batches,
                    train_log_interval,
                    extra=(
                        f"loss={float(loss.item()):.6f} | "
                        f"task_losses={round_value_map({task: value.item() for task, value in loss_dict.items() if task != 'total'})} | "
                        f"lr={runtime.learning_rate:.10g} | grad_norm={grad_norm_text} | "
                        f"loss_scale={runtime.loss_scale:.6f}"
                    ),
                )
            last_good_path = save_stage1_last_good_checkpoint(
                working_dir,
                model,
                runtime,
                phase="warmup",
                epoch=epoch_idx,
            )
            logger.info("Stage-1 last-good checkpoint updated: %s", last_good_path)
    else:
        logger.info("Warmup is skipped because AFFINITY_WARMUP_EPOCHS=%d", warmup_epochs)

    execution_context = update_stage1_execution_context(
        execution_context,
        phase="warmup_validation",
        epoch=max(warmup_epochs - 1, 0),
    )
    warmup_validation_losses = evaluate_task_losses(
        model,
        data_loader_val,
        loss_ft,
        config.TASKS,
        device,
        amp_enabled=runtime.amp_enabled,
        execution_context=execution_context,
    )
    for task, value in warmup_validation_losses.items():
        ensure_finite_number(
            value,
            f"warmup_validation_losses.{task}",
            update_stage1_execution_context(dict(execution_context), task=task),
        )
    ensure_finite_model_state(model, execution_context)
    warmup_state_dict = clone_state_dict_to_cpu(model)

    warmup_checkpoint_path = os.path.join(working_dir, "warmup_checkpoint.pth")
    atomic_torch_save(
        {
            "model": warmup_state_dict,
            "epoch": max(0, warmup_epochs - 1),
            "extra_state": {
                "stage": "ag_mtlora_stage1_warmup",
                "tasks": list(config.TASKS),
                "affinity_warmup_epochs": warmup_epochs,
                "affinity_score_epochs": affinity_score_epochs,
                "partition_granularity": partition_granularity,
                "selection_data_split_mode": selection_data_split_mode,
                "meta_split_path": meta_split_path,
                "stage1_runtime_schema_version": STAGE1_RUNTIME_SCHEMA_VERSION,
            },
        },
        warmup_checkpoint_path,
    )
    logger.info(
        "Warmup checkpoint saved to %s | %s_losses=%s",
        warmup_checkpoint_path,
        selection_loss_label,
        round_value_map(warmup_validation_losses),
    )

    if partition_granularity == "stage":
        shared_params, shared_param_stage_slices = get_shared_ta_parameters_and_stage_slices(
            model,
            num_stages,
        )
    else:
        shared_params = get_shared_ta_parameters(model)
        shared_param_stage_slices = None
    task_index = {task: idx for idx, task in enumerate(config.TASKS)}
    affinity_epoch_history = []
    affinity_epoch_history_by_stage = []
    num_batches_per_epoch = []

    logger.info(
        "Collecting directed affinity on shared TA-LoRA parameters | tasks=%s | affinity_score_epochs=%d | train_batches=%d",
        list(config.TASKS),
        affinity_score_epochs,
        num_train_batches,
    )
    if affinity_score_epochs == 0:
        logger.info("Affinity score collection is skipped because AFFINITY_SCORE_EPOCHS=0")

    for affinity_epoch_idx in range(affinity_score_epochs):
        execution_context = update_stage1_execution_context(
            execution_context,
            phase="affinity",
            epoch=affinity_epoch_idx,
        )
        logger.info("Affinity epoch %d/%d started", affinity_epoch_idx + 1, affinity_score_epochs)
        epoch_directed_sum = torch.zeros((len(config.TASKS), len(config.TASKS)), dtype=torch.float32, device=device)
        epoch_directed_sum_by_stage = None
        if partition_granularity == "stage":
            epoch_directed_sum_by_stage = [
                torch.zeros((len(config.TASKS), len(config.TASKS)), dtype=torch.float32, device=device)
                for _ in range(num_stages)
            ]
        epoch_batches = 0
        model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(data_loader_train, start=1):
            samples, targets = move_batch_to_device(batch, config.TASKS, device)
            execution_context = update_stage1_execution_context(
                execution_context,
                batch=batch_idx - 1,
                sample_ids=get_batch_sample_ids(batch),
                valid_label_ratios=get_target_valid_ratios(targets),
                learning_rate=runtime.learning_rate,
                loss_scale=runtime.loss_scale,
            )
            with torch.cuda.amp.autocast(enabled=runtime.amp_enabled):
                outputs = model(samples)
                loss, loss_dict = criterion(outputs, targets)
            execution_context = update_stage1_execution_context(
                execution_context,
                task_losses={
                    task: float(loss_dict[task].detach().float().item())
                    for task in config.TASKS
                },
                total_loss=float(loss.detach().float().item()),
            )
            ensure_finite_outputs(outputs, execution_context)

            flat_gradients = {}
            pseudo_updates = {}
            for task in config.TASKS:
                task_loss = loss_dict[task]
                task_context = update_stage1_execution_context(dict(execution_context), task=task)
                ensure_finite_tensor(task_loss, f"task_losses.{task}", task_context)
                grads = torch.autograd.grad(
                    task_loss,
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                flat_grad = flatten_gradient_list(shared_params, grads)
                ensure_finite_tensor(flat_grad, f"flat_gradients.{task}", task_context)
                flat_gradients[task] = flat_grad
                pseudo_updates[task] = build_pseudo_update(flat_grad, optimizer)
                ensure_finite_tensor(pseudo_updates[task], f"pseudo_updates.{task}", task_context)

            lr = float(optimizer.param_groups[0]["lr"])
            for src_task in config.TASKS:
                src_idx = task_index[src_task]
                for dst_task in config.TASKS:
                    dst_idx = task_index[dst_task]
                    dot_value = torch.dot(flat_gradients[dst_task], pseudo_updates[src_task])
                    ensure_finite_tensor(
                        dot_value,
                        f"affinity_dot.{src_task}.{dst_task}",
                        execution_context,
                    )
                    epoch_directed_sum[src_idx, dst_idx] += dot_value / max(lr, 1e-12)
                    if epoch_directed_sum_by_stage is not None:
                        for stage_idx, flat_slices in enumerate(shared_param_stage_slices):
                            epoch_directed_sum_by_stage[stage_idx][src_idx, dst_idx] += (
                                dot_on_flat_slices(
                                    flat_gradients[dst_task],
                                    pseudo_updates[src_task],
                                    flat_slices,
                                )
                                / max(lr, 1e-12)
                            )
            epoch_batches += 1

            step_result = stage1_optimizer_step(runtime, loss, model, execution_context)
            execution_context = update_stage1_execution_context(
                execution_context,
                loss_scale=step_result.loss_scale_after,
                amp_overflow_count=runtime.amp_overflow_count,
                optimizer_step_skipped=step_result.optimizer_step_skipped,
            )
            grad_norm_text = log_stage1_step_result(
                logger,
                step_result,
                execution_context,
            )
            maybe_log_progress(
                logger,
                f"Affinity epoch {affinity_epoch_idx + 1}/{affinity_score_epochs}",
                batch_idx,
                num_train_batches,
                train_log_interval,
                extra=(
                    f"loss={float(loss.item()):.6f} | "
                    f"task_losses={round_value_map({task: value.item() for task, value in loss_dict.items() if task != 'total'})} | "
                    f"lr={runtime.learning_rate:.10g} | grad_norm={grad_norm_text} | "
                    f"loss_scale={runtime.loss_scale:.6f}"
                ),
            )

        epoch_directed_affinity = (epoch_directed_sum / max(float(epoch_batches), 1.0)).detach().cpu().numpy()
        if epoch_directed_sum_by_stage is not None:
            epoch_directed_affinity_by_stage = [
                (stage_sum / max(float(epoch_batches), 1.0)).detach().cpu().numpy()
                for stage_sum in epoch_directed_sum_by_stage
            ]
            affinity_epoch_history_by_stage.append(epoch_directed_affinity_by_stage)
        affinity_epoch_history.append(epoch_directed_affinity)
        num_batches_per_epoch.append(int(epoch_batches))
        epoch_matrix_stats = {
            "mean": float(np.mean(epoch_directed_affinity)),
            "std": float(np.std(epoch_directed_affinity)),
            "min": float(np.min(epoch_directed_affinity)),
            "max": float(np.max(epoch_directed_affinity)),
        }
        logger.info(
            "Affinity epoch %d/%d finished | epoch_batches=%d | epoch_directed_matrix_stats=%s",
            affinity_epoch_idx + 1,
            affinity_score_epochs,
            epoch_batches,
            round_value_map(epoch_matrix_stats),
        )
        ensure_finite_tensor(epoch_directed_sum, "epoch_directed_sum", execution_context)
        last_good_path = save_stage1_last_good_checkpoint(
            working_dir,
            model,
            runtime,
            phase="affinity",
            epoch=affinity_epoch_idx,
        )
        logger.info("Stage-1 last-good checkpoint updated: %s", last_good_path)

    if affinity_epoch_history:
        directed_affinity = np.mean(np.stack(affinity_epoch_history, axis=0), axis=0)
    else:
        directed_affinity = np.zeros((len(config.TASKS), len(config.TASKS)), dtype=np.float32)
    if affinity_epoch_history_by_stage:
        directed_affinity_by_stage = [
            np.mean(
                np.stack([epoch_stage_affinity[stage_idx] for epoch_stage_affinity in affinity_epoch_history_by_stage], axis=0),
                axis=0,
            )
            for stage_idx in range(num_stages)
        ]
    elif partition_granularity == "stage":
        directed_affinity_by_stage = [
            np.zeros((len(config.TASKS), len(config.TASKS)), dtype=np.float32)
            for _ in range(num_stages)
        ]
    else:
        directed_affinity_by_stage = []
    symmetric_affinity = 0.5 * (directed_affinity + directed_affinity.T)
    execution_context = update_stage1_execution_context(
        execution_context,
        phase="post_affinity_validation",
        epoch=max(affinity_score_epochs - 1, 0),
    )
    post_affinity_validation_losses = evaluate_task_losses(
        model,
        data_loader_val,
        loss_ft,
        config.TASKS,
        device,
        amp_enabled=runtime.amp_enabled,
        execution_context=execution_context,
    )
    for task, value in post_affinity_validation_losses.items():
        ensure_finite_number(
            value,
            f"post_affinity_validation_losses.{task}",
            update_stage1_execution_context(dict(execution_context), task=task),
        )
    ensure_finite_model_state(model, execution_context)
    post_affinity_state_dict = clone_state_dict_to_cpu(model)
    try:
        smoke_batch = next(iter(data_loader_train))
    except StopIteration as exc:
        raise Stage1NumericalError(
            "Stage-1 AMP smoke test requires at least one training batch.",
            details=dict(execution_context),
        ) from exc
    smoke_result = run_stage1_amp_smoke_test(
        model,
        smoke_batch,
        criterion,
        config.TASKS,
        device,
        runtime.amp_enabled,
        execution_context=execution_context,
    )
    logger.info("Stage-1 post-affinity AMP smoke test passed: %s", smoke_result)
    ensure_finite_model_state(post_affinity_state_dict, execution_context)
    post_affinity_checkpoint_path = os.path.join(working_dir, "post_affinity_checkpoint.pth")
    atomic_torch_save(
        {
            "model": post_affinity_state_dict,
            "epoch": max(0, warmup_epochs + affinity_score_epochs - 1) if (warmup_epochs + affinity_score_epochs) > 0 else 0,
            "extra_state": {
                "stage": "ag_mtlora_stage1_post_affinity",
                "tasks": list(config.TASKS),
                "affinity_warmup_epochs": warmup_epochs,
                "affinity_score_epochs": affinity_score_epochs,
                "partition_granularity": partition_granularity,
                "selection_data_split_mode": selection_data_split_mode,
                "meta_split_path": meta_split_path,
                "amp_smoke_test": smoke_result,
                "stage1_runtime_schema_version": STAGE1_RUNTIME_SCHEMA_VERSION,
            },
        },
        post_affinity_checkpoint_path,
    )
    logger.info(
        "Stage-1 Step A finished | num_affinity_epochs=%d | num_batches_per_epoch=%s | %s_losses=%s | post_affinity_checkpoint=%s",
        affinity_score_epochs,
        num_batches_per_epoch,
        selection_loss_label,
        round_value_map(post_affinity_validation_losses),
        post_affinity_checkpoint_path,
    )

    del dataset_train, dataset_val, data_loader_train, data_loader_val
    del optimizer, criterion, loss_ft, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "warmup_checkpoint_path": warmup_checkpoint_path,
        "warmup_state_dict": warmup_state_dict,
        "warmup_validation_losses": warmup_validation_losses,
        "post_affinity_checkpoint_path": post_affinity_checkpoint_path,
        "post_affinity_state_dict": post_affinity_state_dict,
        "post_affinity_validation_losses": post_affinity_validation_losses,
        "directed_affinity": directed_affinity,
        "directed_affinity_by_stage": directed_affinity_by_stage,
        "symmetric_affinity": symmetric_affinity,
        "affinity_epoch_history": affinity_epoch_history,
        "num_batches_per_epoch": num_batches_per_epoch,
        "num_affinity_epochs": affinity_score_epochs,
        "validation_losses": post_affinity_validation_losses,
        "selection_data_split_mode": selection_data_split_mode,
        "selection_eval_label": selection_eval_label,
        "meta_split_path": meta_split_path,
        "num_batches": int(sum(num_batches_per_epoch)),
    }


def build_group_proxy(directed_affinity: np.ndarray, tasks: Sequence[str], candidate_groups: Sequence[Sequence[str]]):
    task_index = {task: idx for idx, task in enumerate(tasks)}
    group_proxy = {}
    csv_rows = []
    for group in candidate_groups:
        group = list(group)
        group_key = group_to_key(group)
        group_proxy[group_key] = {}
        for task in group:
            if len(group) == 1:
                proxy_value = 0.0
            else:
                incoming = [
                    float(directed_affinity[task_index[other_task], task_index[task]])
                    for other_task in group if other_task != task
                ]
                proxy_value = float(np.mean(incoming)) if len(incoming) > 0 else 0.0
            group_proxy[group_key][task] = proxy_value
            csv_rows.append({"group": group_key, "task": task, "proxy": proxy_value})
    return group_proxy, csv_rows


def collect_predictor_training_targets(
    base_config: CN,
    logger,
    baseline_state_dict: Dict[str, torch.Tensor],
    train_groups: Sequence[Sequence[str]],
    working_dir: str,
    data_split_manifest: Dict = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor_targets, singleton_losses = load_predictor_progress(working_dir, logger)
    selection_data_split_mode = get_selection_data_split_mode(base_config, data_split_manifest)
    selection_eval_label = get_selection_eval_label(base_config, data_split_manifest)
    meta_split_path = None if data_split_manifest is None else data_split_manifest.get("meta_split_path")

    singleton_groups = canonicalize_groups(
        [group for group in train_groups if len(group) == 1],
        base_config.TASKS,
    )
    multitask_groups = [group for group in train_groups if len(group) > 1]
    logger.info(
        "Stage-1 Step B start | predictor_train_groups=%d | singleton_groups=%d | multitask_groups=%d",
        len(train_groups),
        len(singleton_groups),
        len(multitask_groups),
    )
    for singleton_group in singleton_groups:
        task = singleton_group[0]
        if task in singleton_losses:
            predictor_targets[group_to_key([task])] = {task: 0.0}
            logger.info(
                "Predictor target collection skipped | singleton task=%s already cached | val_loss=%.6f",
                task,
                singleton_losses[task],
            )
            continue
        logger.info("Predictor target collection | singleton group=%s", singleton_group)
        task_output_dir = os.path.join(working_dir, "predictor_runs", f"singleton__{task}")
        task_config = rebuild_task_config(base_config, [task], output_dir=task_output_dir)
        _, val_losses = short_train_model(
            task_config,
            logger,
            baseline_state_dict,
            task_config.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS,
            device,
            data_split_manifest=data_split_manifest,
        )
        singleton_losses[task] = float(val_losses[task])
        predictor_targets[group_to_key([task])] = {task: 0.0}
        logger.info(
            "Predictor target collected | singleton task=%s | val_loss=%.6f",
            task,
            singleton_losses[task],
        )
        save_predictor_progress(
            working_dir,
            base_config.TASKS,
            train_groups,
            singleton_losses,
            predictor_targets,
            selection_data_split_mode,
            selection_eval_label,
            meta_split_path,
        )

    for group in train_groups:
        group = list(group)
        if len(group) == 1:
            continue
        group_key = group_to_key(group)
        cached_group_values = predictor_targets.get(group_key, {})
        if all(task in cached_group_values for task in group):
            logger.info(
                "Predictor target collection skipped | group=%s already cached | gains=%s",
                group,
                round_value_map(cached_group_values),
            )
            continue
        logger.info("Predictor target collection | group=%s", group)
        group_output_dir = os.path.join(working_dir, "predictor_runs", f"group__{group_key.replace('|', '__')}")
        group_config = rebuild_task_config(base_config, group, output_dir=group_output_dir)
        _, val_losses = short_train_model(
            group_config,
            logger,
            baseline_state_dict,
            group_config.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS,
            device,
            data_split_manifest=data_split_manifest,
        )
        predictor_targets[group_key] = {}
        for task in group:
            predictor_targets[group_key][task] = float(singleton_losses[task] - val_losses[task])
        logger.info(
            "Predictor target collected | group=%s | gains=%s",
            group,
            round_value_map(predictor_targets[group_key]),
        )
        save_predictor_progress(
            working_dir,
            base_config.TASKS,
            train_groups,
            singleton_losses,
            predictor_targets,
            selection_data_split_mode,
            selection_eval_label,
            meta_split_path,
        )

    return predictor_targets, singleton_losses


def fit_base_predictor(train_pairs: List[Dict], predictor_name: str, predictor_kwargs: Dict):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import SplineTransformer
    except ImportError as exc:
        raise ImportError("AG-MTLoRA Stage-1 requires scikit-learn for the predictor chain.") from exc

    x = np.array([row["proxy"] for row in train_pairs], dtype=np.float64).reshape(-1, 1)
    y = np.array([row["gain"] for row in train_pairs], dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError("No training pairs were collected for the base predictor.")

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x.shape[0] >= 2 and x_std > 1e-12:
        correlation = float(np.corrcoef(x[:, 0], y)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    if x_std <= 1e-12:
        affine_a = 1.0
    else:
        affine_a = correlation * (y_std / max(x_std, 1e-12))
    affine_b = y_mean - affine_a * x_mean
    calibrated_x = affine_a * x + affine_b

    predictor_name = str(predictor_name)
    predictor_kwargs = dict(predictor_kwargs or {})
    if predictor_name == "spline_ridge":
        model = Pipeline(
            steps=[
                (
                    "spline",
                    SplineTransformer(
                        degree=int(predictor_kwargs.get("degree", 3)),
                        n_knots=int(predictor_kwargs.get("n_knots", 4)),
                        include_bias=False,
                    ),
                ),
                ("ridge", Ridge(alpha=float(predictor_kwargs.get("alpha", 1.0)))),
            ]
        )
    elif predictor_name == "knn":
        model = KNeighborsRegressor(n_neighbors=int(predictor_kwargs.get("n_neighbors", 3)))
    elif predictor_name == "rf":
        model = RandomForestRegressor(
            n_estimators=int(predictor_kwargs.get("n_estimators", 200)),
            random_state=int(predictor_kwargs.get("random_state", 0)),
        )
    else:
        raise ValueError(f"Unsupported BASE_PREDICTOR: {predictor_name}")

    model.fit(calibrated_x, y)
    return {
        "predictor_name": predictor_name,
        "affine_a": affine_a,
        "affine_b": affine_b,
        "model": model,
        "num_samples": int(x.shape[0]),
    }


def predict_base_gain(base_predictor, proxy_value: float) -> float:
    calibrated_x = np.array([[base_predictor["affine_a"] * proxy_value + base_predictor["affine_b"]]], dtype=np.float64)
    prediction = base_predictor["model"].predict(calibrated_x)
    return float(np.asarray(prediction).reshape(-1)[0])


def fit_residual_predictor(tasks: Sequence[str], train_groups: Sequence[Sequence[str]], predictor_targets: Dict[str, Dict[str, float]], initial_predictions: Dict[str, Dict[str, float]], alpha: float):
    try:
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise ImportError("AG-MTLoRA Stage-1 requires scikit-learn for the residual predictor.") from exc

    x_rows = []
    y_rows = []
    for group in train_groups:
        group = list(group)
        group_key = group_to_key(group)
        mask = np.array([1.0 if task in group else 0.0 for task in tasks], dtype=np.float64)
        residual = np.zeros(len(tasks), dtype=np.float64)
        for task_idx, task in enumerate(tasks):
            if task in predictor_targets[group_key]:
                residual[task_idx] = predictor_targets[group_key][task] - initial_predictions[group_key].get(task, 0.0)
        x_rows.append(mask)
        y_rows.append(residual)

    model = Ridge(alpha=float(alpha))
    model.fit(np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64))
    return model


def predict_all_candidate_groups(
    tasks: Sequence[str],
    candidate_groups: Sequence[Sequence[str]],
    group_proxy: Dict[str, Dict[str, float]],
    base_predictor,
    residual_model,
):
    initial_predictions = {}
    residual_predictions = {}
    final_predictions = {}

    for group in candidate_groups:
        group = list(group)
        group_key = group_to_key(group)
        initial_predictions[group_key] = {task: 0.0 for task in group}
        if len(group) == 1:
            task = group[0]
            initial_predictions[group_key][task] = 0.0
            residual_predictions[group_key] = {task: 0.0 for task in group}
            final_predictions[group_key] = {task: 0.0 for task in group}
            continue

        for task in group:
            initial_predictions[group_key][task] = predict_base_gain(
                base_predictor,
                group_proxy[group_key][task],
            )

        mask = np.array([[1.0 if task in group else 0.0 for task in tasks]], dtype=np.float64)
        residual_vector = residual_model.predict(mask)[0]
        residual_predictions[group_key] = {
            task: float(residual_vector[tasks.index(task)]) for task in group
        }
        final_predictions[group_key] = {
            task: float(initial_predictions[group_key][task] + residual_predictions[group_key][task])
            for task in group
        }

    return initial_predictions, residual_predictions, final_predictions


def run_partition_search(tasks: Sequence[str], search_scores: Dict[str, Dict[str, float]], max_groups: int):
    partitions = enumerate_partitions(tasks, max_groups)
    ranked_results = []
    for partition in partitions:
        partition = canonicalize_groups(partition, tasks)
        per_task_scores = {}
        for group in partition:
            group_key = group_to_key(group)
            for task in group:
                per_task_scores[task] = float(search_scores[group_key][task])
        score = float(np.mean([per_task_scores[task] for task in tasks]))
        ranked_results.append(
            {
                "groups": partition,
                "partition_score": score,
                "per_task_scores": per_task_scores,
                "num_groups": len(partition),
            }
        )

    ranked_results.sort(
        key=lambda item: (
            -item["partition_score"],
            item["num_groups"],
            [group_to_key(group) for group in item["groups"]],
        )
    )
    return ranked_results


def run_stagewise_partition_search(
    tasks: Sequence[str],
    search_scores_by_stage: Sequence[Dict[str, Dict[str, float]]],
    max_groups: int,
) -> List[List[Dict]]:
    return [
        run_partition_search(tasks, stage_search_scores, max_groups)
        for stage_search_scores in search_scores_by_stage
    ]


def create_resolved_training_config(
    base_cfg_path: str,
    grouping_json_path: str,
    resolved_group_ranks: List[List[int]],
    partition_granularity: str = DEFAULT_PARTITION_GRANULARITY,
) -> CN:
    if not base_cfg_path:
        raise ValueError("AG-MTLoRA Stage-1 requires the original --cfg path to build a reusable training override.")

    resolved_config = CN(new_allowed=True)
    resolved_config.BASE = [os.path.abspath(base_cfg_path)]
    resolved_config.MODEL = CN(new_allowed=True)
    resolved_config.MODEL.AGMTLORA = CN(new_allowed=True)
    resolved_config.MODEL.AGMTLORA.ENABLED = True
    resolved_config.MODEL.AGMTLORA.STAGE = 1
    resolved_config.MODEL.AGMTLORA.PARTITION_GRANULARITY = normalize_partition_granularity(
        partition_granularity
    )
    resolved_config.MODEL.AGMTLORA.GROUPING_SOURCE = "fixed_json"
    resolved_config.MODEL.AGMTLORA.GROUPING_JSON = os.path.abspath(grouping_json_path)
    resolved_config.MODEL.AGMTLORA.GROUP_SHARED_RANKS = [
        [int(rank) for rank in stage_ranks]
        for stage_ranks in resolved_group_ranks
    ]
    resolved_config.freeze()
    return resolved_config


def write_search_artifacts(
    *,
    tasks: Sequence[str],
    search_scores,
    search_score_source: str,
    search_score_path: str,
    output_root: str,
    grouping_save_path: str,
    max_groups: int,
    group_shared_ranks,
    total_shared_rank_budget: int,
    num_stages: int,
    group_rank_allocation: str,
    search_objective: str,
    selection_data_split_mode: str,
    selection_eval_label: str,
    meta_split_path: Optional[str],
    affinity_path: Optional[str],
    final_predictions_path: Optional[str],
    warmup_checkpoint_path: Optional[str],
    post_affinity_checkpoint_path: Optional[str],
    affinity_warmup_epochs: int,
    affinity_score_epochs: int,
    base_cfg_path: str,
    runtime_snapshot_text: str,
    logger,
    partition_granularity: str = DEFAULT_PARTITION_GRANULARITY,
) -> Dict[str, object]:
    search_score_source = normalize_search_score_source(search_score_source)
    partition_granularity = normalize_partition_granularity(partition_granularity)
    artifact_paths = build_search_artifact_paths(output_root, grouping_save_path, search_score_source)

    if partition_granularity == "stage":
        if search_score_source != "group_proxy":
            raise ValueError("Stage-wise partition artifacts currently require search_score_source='group_proxy'.")
        if not isinstance(search_scores, (list, tuple)):
            raise ValueError("Stage-wise partition search expects a per-stage list of search score tables.")

        ranked_partitions_by_stage = run_stagewise_partition_search(tasks, search_scores, int(max_groups))
        best_partition_by_stage = [ranked_partitions[0] for ranked_partitions in ranked_partitions_by_stage]
        group_slot_names = [group_display_name(group_idx) for group_idx in range(int(max_groups))]
        groups_by_stage = [best_partition["groups"] for best_partition in best_partition_by_stage]
        task_to_group_by_stage = build_task_to_group_by_stage(groups_by_stage, group_slot_names)
        num_groups_by_stage = [len(groups) for groups in groups_by_stage]
        resolved_group_ranks, rank_source = resolve_stagewise_group_shared_ranks(
            group_shared_ranks,
            int(total_shared_rank_budget),
            int(max_groups),
            num_groups_by_stage,
            int(num_stages),
            str(group_rank_allocation),
        )
        logger.info(
            "Stage-wise partition search finished | search_score_source=%s | max_groups=%d | best_groups_by_stage=%s | best_scores_by_stage=%s",
            search_score_source,
            int(max_groups),
            groups_by_stage,
            [float(best_partition["partition_score"]) for best_partition in best_partition_by_stage],
        )

        save_json(
            {
                "tasks": list(tasks),
                "partition_granularity": partition_granularity,
                "group_slot_names": group_slot_names,
                "selection_data_split_mode": selection_data_split_mode,
                "selection_eval_label": selection_eval_label,
                "meta_split_path": meta_split_path,
                "search_objective": str(search_objective),
                "search_score_source": search_score_source,
                "search_score_path": search_score_path,
                "ranked_partitions_by_stage": ranked_partitions_by_stage,
            },
            artifact_paths["partition_results_path"],
        )
        save_stage_partition_csv(ranked_partitions_by_stage, artifact_paths["partition_results_csv"])

        grouping_payload = {
            "stage1_runtime_schema_version": STAGE1_RUNTIME_SCHEMA_VERSION,
            "tasks": list(tasks),
            "partition_granularity": partition_granularity,
            "group_slot_names": group_slot_names,
            "groups_by_stage": groups_by_stage,
            "task_to_group_by_stage": task_to_group_by_stage,
            "num_groups_by_stage": num_groups_by_stage,
            "max_groups": int(max_groups),
            "group_shared_ranks": resolved_group_ranks,
            "search_objective": str(search_objective),
            "search_score_source": search_score_source,
            "search_score_path": search_score_path,
            "selection_data_split_mode": selection_data_split_mode,
            "selection_eval_label": selection_eval_label,
            "meta_split_path": meta_split_path,
            "affinity_path": affinity_path,
            "final_predictions_path": final_predictions_path,
            "warmup_checkpoint": warmup_checkpoint_path,
            "post_affinity_checkpoint": post_affinity_checkpoint_path,
            "affinity_warmup_epochs": int(affinity_warmup_epochs),
            "affinity_score_epochs": int(affinity_score_epochs),
            "partition_search_results": artifact_paths["partition_results_path"],
            "group_rank_source": rank_source,
        }
        save_json(grouping_payload, artifact_paths["grouping_json_path"])
        logger.info("Grouping saved to %s", artifact_paths["grouping_json_path"])

        resolved_config = create_resolved_training_config(
            base_cfg_path,
            artifact_paths["grouping_json_path"],
            resolved_group_ranks,
            partition_granularity=partition_granularity,
        )
    else:
        ranked_partitions = run_partition_search(tasks, search_scores, int(max_groups))
        logger.info(
            "Partition search finished | search_score_source=%s | max_groups=%d | num_partitions=%d | best_groups=%s | best_score=%.6f",
            search_score_source,
            int(max_groups),
            len(ranked_partitions),
            ranked_partitions[0]["groups"],
            float(ranked_partitions[0]["partition_score"]),
        )

        save_json(
            {
                "tasks": list(tasks),
                "partition_granularity": partition_granularity,
                "selection_data_split_mode": selection_data_split_mode,
                "selection_eval_label": selection_eval_label,
                "meta_split_path": meta_split_path,
                "search_objective": str(search_objective),
                "search_score_source": search_score_source,
                "search_score_path": search_score_path,
                "ranked_partitions": ranked_partitions,
            },
            artifact_paths["partition_results_path"],
        )
        save_partition_csv(ranked_partitions, artifact_paths["partition_results_csv"])

        best_partition = ranked_partitions[0]
        resolved_group_ranks, rank_source = resolve_group_shared_ranks(
            group_shared_ranks,
            int(total_shared_rank_budget),
            len(best_partition["groups"]),
            int(num_stages),
            str(group_rank_allocation),
        )
        task_to_group = build_task_to_group(best_partition["groups"])
        grouping_payload = {
            "stage1_runtime_schema_version": STAGE1_RUNTIME_SCHEMA_VERSION,
            "tasks": list(tasks),
            "partition_granularity": partition_granularity,
            "groups": best_partition["groups"],
            "task_to_group": task_to_group,
            "max_groups": int(max_groups),
            "partition_score": float(best_partition["partition_score"]),
            "group_shared_ranks": resolved_group_ranks,
            "search_objective": str(search_objective),
            "search_score_source": search_score_source,
            "search_score_path": search_score_path,
            "selection_data_split_mode": selection_data_split_mode,
            "selection_eval_label": selection_eval_label,
            "meta_split_path": meta_split_path,
            "affinity_path": affinity_path,
            "final_predictions_path": final_predictions_path,
            "warmup_checkpoint": warmup_checkpoint_path,
            "post_affinity_checkpoint": post_affinity_checkpoint_path,
            "affinity_warmup_epochs": int(affinity_warmup_epochs),
            "affinity_score_epochs": int(affinity_score_epochs),
            "partition_search_results": artifact_paths["partition_results_path"],
            "group_rank_source": rank_source,
        }
        save_json(grouping_payload, artifact_paths["grouping_json_path"])
        logger.info("Grouping saved to %s", artifact_paths["grouping_json_path"])

        resolved_config = create_resolved_training_config(
            base_cfg_path,
            artifact_paths["grouping_json_path"],
            resolved_group_ranks,
            partition_granularity=partition_granularity,
        )

    mkdir_if_missing(os.path.dirname(artifact_paths["resolved_config_path"]))
    with open(artifact_paths["resolved_config_path"], "w", encoding="utf-8") as handle:
        handle.write(resolved_config.dump())
    with open(artifact_paths["resolved_runtime_snapshot_path"], "w", encoding="utf-8") as handle:
        handle.write(runtime_snapshot_text)
    logger.info("Resolved AG-MTLoRA training config saved to %s", artifact_paths["resolved_config_path"])
    logger.info(
        "Resolved AG-MTLoRA runtime snapshot saved to %s",
        artifact_paths["resolved_runtime_snapshot_path"],
    )

    result = {
        "partition_granularity": partition_granularity,
        "group_rank_source": rank_source,
        "resolved_group_ranks": resolved_group_ranks,
        **artifact_paths,
    }
    if partition_granularity == "stage":
        result["ranked_partitions_by_stage"] = ranked_partitions_by_stage
        result["best_partition_by_stage"] = best_partition_by_stage
    else:
        result["ranked_partitions"] = ranked_partitions
        result["best_partition"] = best_partition
    return result


def compare_partition_rankings(
    original_ranked_partitions: Sequence[Dict],
    replay_ranked_partitions: Sequence[Dict],
) -> List[Dict[str, object]]:
    original_ranks = {
        partition_to_key(item["groups"]): rank
        for rank, item in enumerate(original_ranked_partitions, start=1)
    }
    replay_ranks = {
        partition_to_key(item["groups"]): rank
        for rank, item in enumerate(replay_ranked_partitions, start=1)
    }
    ranking_changes = []
    for partition_key in sorted(set(original_ranks) | set(replay_ranks)):
        ranking_changes.append(
            {
                "partition_key": partition_key,
                "old_rank": original_ranks.get(partition_key),
                "new_rank": replay_ranks.get(partition_key),
            }
        )
    ranking_changes.sort(
        key=lambda item: (
            item["new_rank"] if item["new_rank"] is not None else 10**9,
            item["old_rank"] if item["old_rank"] is not None else 10**9,
            item["partition_key"],
        )
    )
    return ranking_changes


def compare_partition_rankings_by_stage(
    original_ranked_partitions_by_stage: Sequence[Sequence[Dict]],
    replay_ranked_partitions_by_stage: Sequence[Sequence[Dict]],
) -> List[List[Dict[str, object]]]:
    num_stages = max(len(original_ranked_partitions_by_stage), len(replay_ranked_partitions_by_stage))
    ranking_changes_by_stage = []
    for stage_idx in range(num_stages):
        original_ranked_partitions = (
            original_ranked_partitions_by_stage[stage_idx]
            if stage_idx < len(original_ranked_partitions_by_stage)
            else []
        )
        replay_ranked_partitions = (
            replay_ranked_partitions_by_stage[stage_idx]
            if stage_idx < len(replay_ranked_partitions_by_stage)
            else []
        )
        ranking_changes_by_stage.append(
            compare_partition_rankings(original_ranked_partitions, replay_ranked_partitions)
        )
    return ranking_changes_by_stage


def replay_stage1_partition_search(
    *,
    base_cfg_path: str,
    stage1_dir: str,
    search_score_source: str = "group_proxy",
    logger=None,
) -> Dict[str, object]:
    stage1_dir = os.path.abspath(stage1_dir)
    search_score_source = normalize_search_score_source(search_score_source)
    if logger is None:
        logger = create_stage1_logger(stage1_dir)

    grouping_json_path = os.path.join(stage1_dir, "grouping.json")
    runtime_snapshot_path = os.path.join(stage1_dir, "resolved_agmtlora_runtime_snapshot.yaml")
    partition_results_path = os.path.join(stage1_dir, "partition_search_results.json")
    group_proxy_json_path = os.path.join(stage1_dir, "group_proxy.json")
    final_predictions_json_path = os.path.join(stage1_dir, "final_predictions.json")

    if search_score_source != DEFAULT_SEARCH_SCORE_SOURCE:
        candidate_grouping_path = append_search_score_suffix(grouping_json_path, search_score_source)
        candidate_runtime_snapshot_path = append_search_score_suffix(runtime_snapshot_path, search_score_source)
        candidate_partition_results_path = append_search_score_suffix(partition_results_path, search_score_source)
        if os.path.exists(candidate_grouping_path):
            grouping_json_path = candidate_grouping_path
        if os.path.exists(candidate_runtime_snapshot_path):
            runtime_snapshot_path = candidate_runtime_snapshot_path
        if os.path.exists(candidate_partition_results_path):
            partition_results_path = candidate_partition_results_path

    if not os.path.exists(grouping_json_path):
        raise FileNotFoundError(f"Missing Stage-1 grouping artifact: {grouping_json_path}")
    if not os.path.exists(runtime_snapshot_path):
        raise FileNotFoundError(f"Missing Stage-1 runtime snapshot artifact: {runtime_snapshot_path}")

    grouping_payload = load_json(grouping_json_path)
    partition_granularity = normalize_partition_granularity(
        grouping_payload.get("partition_granularity", DEFAULT_PARTITION_GRANULARITY)
    )
    if partition_granularity == "stage" and search_score_source != "group_proxy":
        raise ValueError("Stage-wise replay search only supports search_score_source='group_proxy'.")
    runtime_schema_version = int(grouping_payload.get("stage1_runtime_schema_version", 0))
    if runtime_schema_version != STAGE1_RUNTIME_SCHEMA_VERSION:
        raise RuntimeError(
            "Refusing to replay legacy Stage-1 search artifacts without the "
            f"numerical-stability contract (found schema={runtime_schema_version}, "
            f"required={STAGE1_RUNTIME_SCHEMA_VERSION})."
        )

    score_file_path = (
        group_proxy_json_path
        if search_score_source == "group_proxy"
        else final_predictions_json_path
    )
    if not os.path.exists(score_file_path):
        raise FileNotFoundError(
            f"Missing Stage-1 score artifact for source '{search_score_source}': {score_file_path}"
        )

    score_payload = load_json(score_file_path)
    original_partition_payload = (
        load_json(partition_results_path) if os.path.exists(partition_results_path) else {}
    )
    with open(runtime_snapshot_path, "r", encoding="utf-8") as handle:
        runtime_payload = yaml.safe_load(handle) or {}

    tasks = list(score_payload.get("tasks", grouping_payload.get("tasks", [])))
    if not tasks:
        raise ValueError("Replay search could not determine the Stage-1 task list.")

    if partition_granularity == "stage":
        search_scores = score_payload.get("group_proxy_by_stage")
        if not isinstance(search_scores, list):
            raise ValueError(
                f"Replay search expected 'group_proxy_by_stage' in {score_file_path}, but it was not found."
            )
    else:
        score_key = "group_proxy" if search_score_source == "group_proxy" else "final_predictions"
        search_scores = score_payload.get(score_key)
        if not isinstance(search_scores, dict):
            raise ValueError(
                f"Replay search expected '{score_key}' in {score_file_path}, but it was not found."
            )

    model_payload = runtime_payload.get("MODEL", {})
    agmtlora_payload = model_payload.get("AGMTLORA", {})
    swin_payload = model_payload.get("SWIN", {})
    agmtlora_payload["SEARCH_SCORE_SOURCE"] = search_score_source
    agmtlora_payload["PARTITION_GRANULARITY"] = partition_granularity
    model_payload["AGMTLORA"] = agmtlora_payload
    runtime_payload["MODEL"] = model_payload

    search_result = write_search_artifacts(
        tasks=tasks,
        search_scores=search_scores,
        search_score_source=search_score_source,
        search_score_path=score_file_path,
        output_root=stage1_dir,
        grouping_save_path=grouping_json_path,
        max_groups=int(agmtlora_payload.get("MAX_GROUPS", grouping_payload.get("max_groups", len(tasks)))),
        group_shared_ranks=agmtlora_payload.get("GROUP_SHARED_RANKS", []),
        total_shared_rank_budget=int(agmtlora_payload.get("TOTAL_SHARED_RANK_BUDGET", 0)),
        num_stages=len(swin_payload.get("DEPTHS", [])),
        group_rank_allocation=str(agmtlora_payload.get("GROUP_RANK_ALLOCATION", "equal_split")),
        search_objective=str(
            grouping_payload.get(
                "search_objective",
                agmtlora_payload.get("SEARCH_OBJECTIVE", "mean_final_predicted_gain"),
            )
        ),
        selection_data_split_mode=str(
            score_payload.get(
                "selection_data_split_mode",
                grouping_payload.get("selection_data_split_mode", "official_val"),
            )
        ),
        selection_eval_label=str(
            score_payload.get(
                "selection_eval_label",
                grouping_payload.get("selection_eval_label", "official val"),
            )
        ),
        meta_split_path=score_payload.get("meta_split_path", grouping_payload.get("meta_split_path")),
        affinity_path=grouping_payload.get("affinity_path"),
        final_predictions_path=(
            final_predictions_json_path if search_score_source == "final_predictions" else None
        ),
        warmup_checkpoint_path=grouping_payload.get("warmup_checkpoint"),
        post_affinity_checkpoint_path=grouping_payload.get("post_affinity_checkpoint"),
        affinity_warmup_epochs=int(grouping_payload.get("affinity_warmup_epochs", 0)),
        affinity_score_epochs=int(grouping_payload.get("affinity_score_epochs", 0)),
        base_cfg_path=base_cfg_path,
        runtime_snapshot_text=yaml.safe_dump(runtime_payload, sort_keys=False),
        logger=logger,
        partition_granularity=partition_granularity,
    )

    if partition_granularity == "stage":
        original_ranked_partitions_by_stage = original_partition_payload.get("ranked_partitions_by_stage", [])
        ranking_changes_by_stage = compare_partition_rankings_by_stage(
            original_ranked_partitions_by_stage,
            search_result["ranked_partitions_by_stage"],
        )
        if original_ranked_partitions_by_stage:
            logger.info(
                "Replay stage-wise search comparison | original_best_by_stage=%s | replay_best_by_stage=%s",
                [stage_partitions[0]["groups"] for stage_partitions in original_ranked_partitions_by_stage if stage_partitions],
                [best_partition["groups"] for best_partition in search_result["best_partition_by_stage"]],
            )
        ranking_changes = None
        original_best_partition = (
            [stage_partitions[0] for stage_partitions in original_ranked_partitions_by_stage]
            if original_ranked_partitions_by_stage
            else None
        )
    else:
        original_ranked_partitions = original_partition_payload.get("ranked_partitions", [])
        ranking_changes = compare_partition_rankings(
            original_ranked_partitions,
            search_result["ranked_partitions"],
        )
        if original_ranked_partitions:
            logger.info(
                "Replay search comparison | original_best=%s | original_score=%.6f | replay_best=%s | replay_score=%.6f",
                original_ranked_partitions[0]["groups"],
                float(original_ranked_partitions[0]["partition_score"]),
                search_result["best_partition"]["groups"],
                float(search_result["best_partition"]["partition_score"]),
            )
        ranking_changes_by_stage = None
        original_best_partition = original_ranked_partitions[0] if original_ranked_partitions else None

    return {
        "stage1_dir": stage1_dir,
        "partition_granularity": partition_granularity,
        "search_score_source": search_score_source,
        "search_score_path": score_file_path,
        "original_partition_results_path": partition_results_path if os.path.exists(partition_results_path) else None,
        "original_best_partition": original_best_partition,
        "ranking_changes": ranking_changes,
        "ranking_changes_by_stage": ranking_changes_by_stage,
        **search_result,
    }


def run_stage1_pipeline(
    config: CN,
    output_root: str,
    logger,
    base_cfg_path: str,
    execution_context: Optional[Dict] = None,
):
    execution_context = update_stage1_execution_context(
        execution_context,
        phase="pipeline_initialization",
        output_root=os.path.abspath(output_root),
    )
    mkdir_if_missing(output_root)
    logger.info("AG-MTLoRA Stage-1 pipeline started | output_root=%s", output_root)
    data_split_manifest = build_stage1_data_split_manifest(config, logger)
    selection_data_split_mode = get_selection_data_split_mode(config, data_split_manifest)
    selection_eval_label = get_selection_eval_label(config, data_split_manifest)
    meta_split_path = None if data_split_manifest is None else data_split_manifest.get("meta_split_path")
    partition_granularity = get_partition_granularity(config)

    affinity_result = maybe_load_affinity_result_from_existing_artifacts(config, output_root, logger)
    if affinity_result is None:
        affinity_result = warmup_and_collect_affinity(
            config,
            logger,
            output_root,
            data_split_manifest=data_split_manifest,
            execution_context=execution_context,
        )
    directed_affinity = affinity_result["directed_affinity"]
    directed_affinity_by_stage = affinity_result.get("directed_affinity_by_stage", [])
    symmetric_affinity = affinity_result["symmetric_affinity"]
    affinity_warmup_epochs = get_affinity_warmup_epochs(config)
    affinity_score_epochs = get_affinity_score_epochs(config)

    affinity_json_path = config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH
    affinity_csv_path = os.path.splitext(affinity_json_path)[0] + ".csv"
    affinity_sym_json_path = os.path.join(os.path.dirname(affinity_json_path), "affinity_symmetric.json")
    affinity_sym_csv_path = os.path.join(os.path.dirname(affinity_json_path), "affinity_symmetric.csv")
    affinity_epoch_history_json = os.path.join(os.path.dirname(affinity_json_path), "affinity_epoch_history.json")
    affinity_epoch_history_csv = os.path.join(os.path.dirname(affinity_json_path), "affinity_epoch_history.csv")

    save_json(
        {
            "stage1_runtime_schema_version": STAGE1_RUNTIME_SCHEMA_VERSION,
            "tasks": list(config.TASKS),
            "partition_granularity": partition_granularity,
            "selection_data_split_mode": selection_data_split_mode,
            "selection_eval_label": selection_eval_label,
            "meta_split_path": meta_split_path,
            "directed_affinity": directed_affinity.tolist(),
            "directed_affinity_by_stage": [matrix.tolist() for matrix in directed_affinity_by_stage],
            "num_affinity_epochs": int(affinity_result["num_affinity_epochs"]),
            "num_batches_per_epoch": list(affinity_result["num_batches_per_epoch"]),
            "affinity_epoch_history_path": affinity_epoch_history_json,
            "warmup_checkpoint": affinity_result["warmup_checkpoint_path"],
            "post_affinity_checkpoint": affinity_result["post_affinity_checkpoint_path"],
            "warmup_validation_losses": affinity_result["warmup_validation_losses"],
            "post_affinity_validation_losses": affinity_result["post_affinity_validation_losses"],
        },
        affinity_json_path,
    )
    save_matrix_csv(config.TASKS, directed_affinity, affinity_csv_path)
    save_json(
        {
            "tasks": list(config.TASKS),
            "partition_granularity": partition_granularity,
            "selection_data_split_mode": selection_data_split_mode,
            "selection_eval_label": selection_eval_label,
            "meta_split_path": meta_split_path,
            "affinity_warmup_epochs": affinity_warmup_epochs,
            "affinity_score_epochs": affinity_score_epochs,
            "num_batches_per_epoch": list(affinity_result["num_batches_per_epoch"]),
            "epoch_directed_affinity": [matrix.tolist() for matrix in affinity_result["affinity_epoch_history"]],
        },
        affinity_epoch_history_json,
    )
    save_affinity_epoch_history_csv(config.TASKS, affinity_result["affinity_epoch_history"], affinity_epoch_history_csv)

    if config.MODEL.AGMTLORA.VISUALIZE_SYMMETRIC_AFFINITY:
        save_json(
            {
                "tasks": list(config.TASKS),
                "partition_granularity": partition_granularity,
                "selection_data_split_mode": selection_data_split_mode,
                "selection_eval_label": selection_eval_label,
                "meta_split_path": meta_split_path,
                "symmetric_affinity": symmetric_affinity.tolist(),
            },
            affinity_sym_json_path,
        )
        save_matrix_csv(config.TASKS, symmetric_affinity, affinity_sym_csv_path)

    candidate_groups = enumerate_candidate_groups(config.TASKS)
    logger.info("Enumerated candidate groups | num_tasks=%d | num_candidate_groups=%d", len(config.TASKS), len(candidate_groups))
    if partition_granularity == "stage":
        group_proxy_by_stage = [
            build_group_proxy(stage_affinity, config.TASKS, candidate_groups)[0]
            for stage_affinity in directed_affinity_by_stage
        ]
        group_proxy = average_search_scores_by_stage(group_proxy_by_stage)
        group_proxy_rows = [
            {"group": group_key, "task": task, "proxy": float(value)}
            for group_key, group_values in group_proxy.items()
            for task, value in group_values.items()
        ]
    else:
        group_proxy, group_proxy_rows = build_group_proxy(directed_affinity, config.TASKS, candidate_groups)
        group_proxy_by_stage = []
    group_proxy_json_path = os.path.join(output_root, "group_proxy.json")
    group_proxy_csv_path = os.path.join(output_root, "group_proxy.csv")
    save_json(
        {
            "tasks": list(config.TASKS),
            "partition_granularity": partition_granularity,
            "selection_data_split_mode": selection_data_split_mode,
            "selection_eval_label": selection_eval_label,
            "meta_split_path": meta_split_path,
            "group_proxy": group_proxy,
            "group_proxy_by_stage": group_proxy_by_stage,
        },
        group_proxy_json_path,
    )
    save_group_task_rows(group_proxy_rows, group_proxy_csv_path, "proxy")

    search_score_source = get_search_score_source(config)
    if partition_granularity == "stage" and search_score_source != "group_proxy":
        raise ValueError(
            "Stage-wise partition search currently requires MODEL.AGMTLORA.SEARCH_SCORE_SOURCE='group_proxy'."
        )
    predictor_train_groups_json = None
    predictor_train_groups_csv = None
    initial_predictions_json = None
    residual_predictions_json = None
    final_predictions_json = None

    if search_score_source == "group_proxy":
        logger.info(
            "Stage-1 Step C skipped | search_score_source=%s | using group_proxy directly for partition search",
            search_score_source,
        )
        search_scores = group_proxy_by_stage if partition_granularity == "stage" else group_proxy
        search_score_path = group_proxy_json_path
    else:
        predictor_train_groups = select_predictor_train_groups(
            config.TASKS,
            int(config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_BUDGET),
            config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_STRATEGY,
            int(config.SEED),
        )
        logger.info(
            "Selected predictor train groups | budget=%d | selected=%d | groups=%s",
            int(config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_BUDGET),
            len(predictor_train_groups),
            predictor_train_groups,
        )
        predictor_train_groups_json = os.path.join(output_root, "predictor_train_groups.json")
        predictor_train_groups_csv = os.path.join(output_root, "predictor_train_groups.csv")
        save_json(
            {
                "tasks": list(config.TASKS),
                "selection_data_split_mode": selection_data_split_mode,
                "selection_eval_label": selection_eval_label,
                "meta_split_path": meta_split_path,
                "train_groups": predictor_train_groups,
                "singleton_val_losses": {},
                "gt_predicted_gains": {},
            },
            predictor_train_groups_json,
        )
        predictor_targets, singleton_losses = collect_predictor_training_targets(
            config,
            logger,
            affinity_result["post_affinity_state_dict"],
            predictor_train_groups,
            output_root,
            data_split_manifest=data_split_manifest,
        )
        save_json(
            {
                "tasks": list(config.TASKS),
                "selection_data_split_mode": selection_data_split_mode,
                "selection_eval_label": selection_eval_label,
                "meta_split_path": meta_split_path,
                "train_groups": predictor_train_groups,
                "singleton_val_losses": singleton_losses,
                "gt_predicted_gains": predictor_targets,
            },
            predictor_train_groups_json,
        )
        save_predictor_value_csv(predictor_targets, predictor_train_groups_csv)

        base_train_pairs = []
        for group in predictor_train_groups:
            group_key = group_to_key(group)
            for task in group:
                base_train_pairs.append(
                    {
                        "group": group_key,
                        "task": task,
                        "proxy": float(group_proxy[group_key][task]),
                        "gain": float(predictor_targets[group_key][task]),
                    }
                )

        base_predictor = fit_base_predictor(
            base_train_pairs,
            config.MODEL.AGMTLORA.BASE_PREDICTOR,
            dict(config.MODEL.AGMTLORA.BASE_PREDICTOR_KWARGS),
        )
        logger.info(
            "Base predictor fitted | type=%s | samples=%d | affine_a=%.6f | affine_b=%.6f",
            base_predictor["predictor_name"],
            base_predictor["num_samples"],
            float(base_predictor["affine_a"]),
            float(base_predictor["affine_b"]),
        )

        initial_train_predictions = {}
        for group in predictor_train_groups:
            group_key = group_to_key(group)
            initial_train_predictions[group_key] = {}
            for task in group:
                initial_train_predictions[group_key][task] = predict_base_gain(
                    base_predictor,
                    group_proxy[group_key][task],
                )

        residual_model = fit_residual_predictor(
            config.TASKS,
            predictor_train_groups,
            predictor_targets,
            initial_train_predictions,
            float(config.MODEL.AGMTLORA.RESIDUAL_ALPHA),
        )
        logger.info(
            "Residual predictor fitted | type=%s | alpha=%.6f",
            str(config.MODEL.AGMTLORA.RESIDUAL_PREDICTOR),
            float(config.MODEL.AGMTLORA.RESIDUAL_ALPHA),
        )

        initial_predictions, residual_predictions, final_predictions = predict_all_candidate_groups(
            config.TASKS,
            candidate_groups,
            group_proxy,
            base_predictor,
            residual_model,
        )

        initial_predictions_json = os.path.join(output_root, "initial_predictions.json")
        residual_predictions_json = os.path.join(output_root, "residual_predictions.json")
        final_predictions_json = os.path.join(output_root, "final_predictions.json")
        save_json(
            {
                "tasks": list(config.TASKS),
                "selection_data_split_mode": selection_data_split_mode,
                "selection_eval_label": selection_eval_label,
                "meta_split_path": meta_split_path,
                "initial_predictions": initial_predictions,
            },
            initial_predictions_json,
        )
        save_json(
            {
                "tasks": list(config.TASKS),
                "selection_data_split_mode": selection_data_split_mode,
                "selection_eval_label": selection_eval_label,
                "meta_split_path": meta_split_path,
                "residual_predictions": residual_predictions,
            },
            residual_predictions_json,
        )
        save_json(
            {
                "tasks": list(config.TASKS),
                "selection_data_split_mode": selection_data_split_mode,
                "selection_eval_label": selection_eval_label,
                "meta_split_path": meta_split_path,
                "final_predictions": final_predictions,
            },
            final_predictions_json,
        )
        save_predictor_value_csv(initial_predictions, os.path.join(output_root, "initial_predictions.csv"))
        save_predictor_value_csv(residual_predictions, os.path.join(output_root, "residual_predictions.csv"))
        save_predictor_value_csv(final_predictions, os.path.join(output_root, "final_predictions.csv"))
        search_scores = final_predictions
        search_score_path = final_predictions_json

    search_artifacts = write_search_artifacts(
        tasks=config.TASKS,
        search_scores=search_scores,
        search_score_source=search_score_source,
        search_score_path=search_score_path,
        output_root=output_root,
        grouping_save_path=config.MODEL.AGMTLORA.GROUPING_SAVE_PATH,
        max_groups=int(config.MODEL.AGMTLORA.MAX_GROUPS),
        group_shared_ranks=config.MODEL.AGMTLORA.GROUP_SHARED_RANKS,
        total_shared_rank_budget=int(config.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET),
        num_stages=len(config.MODEL.SWIN.DEPTHS),
        group_rank_allocation=str(config.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION),
        search_objective=str(config.MODEL.AGMTLORA.SEARCH_OBJECTIVE),
        selection_data_split_mode=selection_data_split_mode,
        selection_eval_label=selection_eval_label,
        meta_split_path=meta_split_path,
        affinity_path=affinity_json_path,
        final_predictions_path=final_predictions_json,
        warmup_checkpoint_path=affinity_result["warmup_checkpoint_path"],
        post_affinity_checkpoint_path=affinity_result["post_affinity_checkpoint_path"],
        affinity_warmup_epochs=affinity_warmup_epochs,
        affinity_score_epochs=affinity_score_epochs,
        base_cfg_path=base_cfg_path,
        runtime_snapshot_text=config.dump(),
        logger=logger,
        partition_granularity=partition_granularity,
    )

    return {
        "affinity_json_path": affinity_json_path,
        "affinity_epoch_history_json": affinity_epoch_history_json,
        "affinity_epoch_history_csv": affinity_epoch_history_csv,
        "group_proxy_json_path": group_proxy_json_path,
        "search_score_source": search_score_source,
        "search_score_path": search_score_path,
        "predictor_train_groups_json": predictor_train_groups_json,
        "predictor_train_groups_csv": predictor_train_groups_csv,
        "initial_predictions_json": initial_predictions_json,
        "residual_predictions_json": residual_predictions_json,
        "final_predictions_json": final_predictions_json,
        "partition_results_path": search_artifacts["partition_results_path"],
        "partition_results_csv": search_artifacts["partition_results_csv"],
        "grouping_json_path": search_artifacts["grouping_json_path"],
        "resolved_config_path": search_artifacts["resolved_config_path"],
        "resolved_runtime_snapshot_path": search_artifacts["resolved_runtime_snapshot_path"],
        "warmup_checkpoint_path": affinity_result["warmup_checkpoint_path"],
        "post_affinity_checkpoint_path": affinity_result["post_affinity_checkpoint_path"],
        "meta_split_path": meta_split_path,
    }


def create_stage1_logger(output_root: str, dist_rank: int = 0):
    mkdir_if_missing(output_root)
    return create_logger(output_dir=output_root, dist_rank=dist_rank, name="ag_mtlora_stage1")
