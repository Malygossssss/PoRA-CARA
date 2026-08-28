import itertools
import json
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_PARTITION_GRANULARITY = "global"
STAGE1_RUNTIME_SCHEMA_VERSION = 2
SUPPORTED_PARTITION_GRANULARITIES = {"global", "stage"}


def group_display_name(group_index: int) -> str:
    return f"group_{int(group_index)}"


def stage_index_to_key(stage_idx: int) -> str:
    return f"stage_{int(stage_idx)}"


def normalize_partition_granularity(partition_granularity: str) -> str:
    granularity = str(partition_granularity or DEFAULT_PARTITION_GRANULARITY)
    if granularity not in SUPPORTED_PARTITION_GRANULARITIES:
        raise ValueError(
            f"Unsupported PARTITION_GRANULARITY: {granularity}. "
            f"Expected one of {sorted(SUPPORTED_PARTITION_GRANULARITIES)}."
        )
    return granularity


def _task_order_map(tasks: Sequence[str]) -> Dict[str, int]:
    return {task: idx for idx, task in enumerate(tasks)}


def canonicalize_groups(groups: Iterable[Iterable[str]], tasks: Sequence[str]) -> List[List[str]]:
    task_order = _task_order_map(tasks)
    normalized = []
    for group in groups:
        unique_group = []
        seen = set()
        for task in group:
            if task in seen:
                continue
            seen.add(task)
            unique_group.append(task)
        normalized.append(sorted(unique_group, key=lambda task: task_order[task]))
    normalized.sort(key=lambda group: [task_order[task] for task in group])
    return normalized


def build_task_to_group(groups: Sequence[Sequence[str]], group_names: Sequence[str] = None) -> Dict[str, str]:
    task_to_group = {}
    if group_names is None:
        group_names = [group_display_name(group_idx) for group_idx in range(len(groups))]
    if len(group_names) < len(groups):
        raise ValueError("Not enough group names were provided for the requested groups.")
    for group_idx, group in enumerate(groups):
        group_name = str(group_names[group_idx])
        for task in group:
            if task in task_to_group:
                raise ValueError(f"Task '{task}' appears in multiple groups.")
            task_to_group[task] = group_name
    return task_to_group


def build_task_to_group_by_stage(
    groups_by_stage: Sequence[Sequence[Sequence[str]]],
    group_slot_names: Sequence[str],
) -> List[Dict[str, str]]:
    return [
        build_task_to_group(stage_groups, group_slot_names)
        for stage_groups in groups_by_stage
    ]


def _validate_groups_cover_tasks(
    groups: Sequence[Sequence[str]],
    expected_tasks: Sequence[str],
    context_label: str,
) -> None:
    covered_tasks = sorted(
        itertools.chain.from_iterable(groups),
        key=lambda task: _task_order_map(expected_tasks)[task],
    )
    if list(expected_tasks) != covered_tasks:
        raise ValueError(
            f"{context_label} tasks do not match config tasks. "
            f"Expected {list(expected_tasks)}, got {covered_tasks}."
        )


def load_grouping_json(grouping_json_path: str, expected_tasks: Sequence[str]) -> Dict:
    with open(grouping_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("post_affinity_checkpoint"):
        runtime_schema_version = int(payload.get("stage1_runtime_schema_version", 0))
        if runtime_schema_version != STAGE1_RUNTIME_SCHEMA_VERSION:
            raise ValueError(
                "Refusing to use a legacy Stage-1 grouping without the numerical-stability "
                f"contract (found schema={runtime_schema_version}, "
                f"required={STAGE1_RUNTIME_SCHEMA_VERSION}). Rerun Stage-1 from the original "
                "backbone checkpoint."
            )

    partition_granularity = normalize_partition_granularity(
        payload.get("partition_granularity", DEFAULT_PARTITION_GRANULARITY)
    )
    payload["partition_granularity"] = partition_granularity

    if partition_granularity == "stage":
        groups_by_stage = payload.get("groups_by_stage", [])
        if not isinstance(groups_by_stage, list) or len(groups_by_stage) == 0:
            raise ValueError("Stage-wise grouping JSON must provide a non-empty 'groups_by_stage' list.")

        normalized_groups_by_stage = []
        for stage_idx, stage_groups in enumerate(groups_by_stage):
            groups = canonicalize_groups(stage_groups, expected_tasks)
            _validate_groups_cover_tasks(groups, expected_tasks, f"Grouping JSON stage_{stage_idx}")
            normalized_groups_by_stage.append(groups)

        max_num_groups = max(len(groups) for groups in normalized_groups_by_stage)
        group_slot_names = payload.get("group_slot_names", [])
        if group_slot_names:
            group_slot_names = [str(group_name) for group_name in group_slot_names]
            if len(group_slot_names) < max_num_groups:
                raise ValueError(
                    "Grouping JSON group_slot_names must cover the maximum number of groups across stages."
                )
        else:
            group_slot_names = [group_display_name(group_idx) for group_idx in range(max_num_groups)]

        payload["groups_by_stage"] = normalized_groups_by_stage
        payload["group_slot_names"] = group_slot_names
        payload["task_to_group_by_stage"] = build_task_to_group_by_stage(
            normalized_groups_by_stage,
            group_slot_names,
        )
        payload["num_groups_by_stage"] = [len(groups) for groups in normalized_groups_by_stage]
        payload["num_groups"] = max_num_groups
        return payload

    groups = payload.get("groups", [])
    groups = canonicalize_groups(groups, expected_tasks)
    _validate_groups_cover_tasks(groups, expected_tasks, "Grouping JSON")
    payload["groups"] = groups
    payload["task_to_group"] = build_task_to_group(groups)
    payload["num_groups"] = len(groups)
    return payload


def resolve_group_shared_ranks(
    group_shared_ranks,
    total_shared_rank_budget: int,
    num_groups: int,
    num_stages: int,
    allocation: str = "equal_split",
) -> Tuple[List[List[int]], str]:
    if num_groups <= 0:
        return [], "manual"

    if group_shared_ranks:
        if all(isinstance(rank, int) for rank in group_shared_ranks):
            return [
                [int(rank)] * num_stages for rank in group_shared_ranks
            ], "manual"
        if all(isinstance(rank, (list, tuple)) for rank in group_shared_ranks):
            resolved = []
            for rank in group_shared_ranks:
                if len(rank) == 1:
                    resolved.append([int(rank[0])] * num_stages)
                elif len(rank) == num_stages:
                    resolved.append([int(v) for v in rank])
                else:
                    raise ValueError(
                        "Each group rank override must have length 1 or match the number of stages."
                    )
            return resolved, "manual"
        raise ValueError("GROUP_SHARED_RANKS must be a flat list or a nested list.")

    if allocation != "equal_split":
        raise ValueError(f"Unsupported GROUP_RANK_ALLOCATION: {allocation}")

    base_rank = int(total_shared_rank_budget) // int(num_groups)
    remainder = int(total_shared_rank_budget) % int(num_groups)
    per_group = [base_rank + (1 if idx < remainder else 0) for idx in range(num_groups)]
    return [[rank] * num_stages for rank in per_group], "auto_equal_split"


def resolve_stagewise_group_shared_ranks(
    group_shared_ranks,
    total_shared_rank_budget: int,
    max_group_slots: int,
    num_groups_by_stage: Sequence[int],
    num_stages: int,
    allocation: str = "equal_split",
) -> Tuple[List[List[int]], str]:
    if int(max_group_slots) <= 0:
        return [], "manual"
    if len(num_groups_by_stage) != int(num_stages):
        raise ValueError("num_groups_by_stage must match the number of stages.")

    max_group_slots = int(max_group_slots)
    num_stages = int(num_stages)
    if group_shared_ranks:
        resolved, rank_source = resolve_group_shared_ranks(
            group_shared_ranks,
            total_shared_rank_budget,
            max_group_slots,
            num_stages,
            allocation,
        )
        if len(resolved) != max_group_slots:
            raise ValueError(
                "Stage-wise GROUP_SHARED_RANKS must provide one rank entry per fixed group slot."
            )
        for stage_idx, num_active_groups in enumerate(num_groups_by_stage):
            for slot_idx in range(int(num_active_groups), max_group_slots):
                resolved[slot_idx][stage_idx] = 0
        return resolved, rank_source

    if allocation != "equal_split":
        raise ValueError(f"Unsupported GROUP_RANK_ALLOCATION: {allocation}")

    resolved = [[0] * num_stages for _ in range(max_group_slots)]
    total_shared_rank_budget = int(total_shared_rank_budget)
    for stage_idx, num_active_groups in enumerate(num_groups_by_stage):
        num_active_groups = int(num_active_groups)
        if num_active_groups <= 0:
            continue
        base_rank = total_shared_rank_budget // num_active_groups
        remainder = total_shared_rank_budget % num_active_groups
        for slot_idx in range(num_active_groups):
            resolved[slot_idx][stage_idx] = base_rank + (1 if slot_idx < remainder else 0)
    return resolved, "auto_equal_split_stage"


def enumerate_candidate_groups(tasks: Sequence[str]) -> List[List[str]]:
    candidates = []
    for group_size in range(1, len(tasks) + 1):
        for group in itertools.combinations(tasks, group_size):
            candidates.append(list(group))
    return candidates


def enumerate_partitions(tasks: Sequence[str], max_groups: int) -> List[List[List[str]]]:
    partitions: List[List[List[str]]] = []

    def _helper(task_index: int, current_partition: List[List[str]]) -> None:
        if task_index == len(tasks):
            partitions.append([group[:] for group in current_partition])
            return

        task = tasks[task_index]
        for group in current_partition:
            group.append(task)
            _helper(task_index + 1, current_partition)
            group.pop()

        if len(current_partition) < max_groups:
            current_partition.append([task])
            _helper(task_index + 1, current_partition)
            current_partition.pop()

    _helper(0, [])
    return [canonicalize_groups(partition, tasks) for partition in partitions]


def select_predictor_train_groups(
    tasks: Sequence[str],
    budget: int,
    strategy: str,
    seed: int,
) -> List[List[str]]:
    candidates = enumerate_candidate_groups(tasks)
    if budget <= 0 or budget >= len(candidates):
        return candidates

    strategy = str(strategy)
    rng = random.Random(int(seed))
    singletons = [group for group in candidates if len(group) == 1]
    pairs = [group for group in candidates if len(group) == 2]
    higher_order = [group for group in candidates if len(group) >= 3]
    effective_budget = max(int(budget), len(singletons))
    if effective_budget >= len(candidates):
        return candidates

    selected = []
    selected.extend(singletons)

    if strategy in {"all_singletons+all_pairs+random_higher_order", "default"}:
        selected.extend(pairs)
        remaining = max(0, effective_budget - len(selected))
        if remaining > 0:
            rng.shuffle(higher_order)
            selected.extend(higher_order[:remaining])
    elif strategy == "random":
        pool = [group for group in candidates if group not in singletons]
        rng.shuffle(pool)
        remaining = max(0, effective_budget - len(selected))
        selected.extend(pool[:remaining])
    else:
        raise ValueError(f"Unsupported predictor train group strategy: {strategy}")

    return canonicalize_groups(selected[:effective_budget], tasks)


def resolve_artifact_path(base_output_dir: str, path_value: str, default_name: str) -> str:
    path_value = str(path_value or "").strip()
    if not path_value:
        path_value = os.path.join(base_output_dir, default_name)
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_output_dir, path_value))
