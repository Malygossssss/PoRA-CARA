import json
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional


MULTI_PROCESS_MODE = "unipora_independent_processes"
RANK_ARTIFACT_MANIFEST = "stage1_artifacts.json"
ROOT_MULTI_PROCESS_MANIFEST = "stage1_multi_process_manifest.json"
FAILURE_REPORT = "failure_report.json"


@dataclass(frozen=True)
class Stage1ProcessContext:
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_multi_process(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def effective_seed(self, base_seed: int) -> int:
        return int(base_seed) + self.rank


@dataclass(frozen=True)
class Stage1OutputPaths:
    run_root: str
    rank_output_root: str
    rank_artifact_manifest_path: str
    root_manifest_path: str


def _parse_environment_int(name: str, value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def resolve_process_context(
    cli_local_rank: int = 0,
    environ: Optional[Mapping[str, str]] = None,
) -> Stage1ProcessContext:
    env = os.environ if environ is None else environ
    world_size = _parse_environment_int("WORLD_SIZE", env.get("WORLD_SIZE", 1))
    rank = _parse_environment_int("RANK", env.get("RANK", 0))
    local_rank = _parse_environment_int("LOCAL_RANK", env.get("LOCAL_RANK", cli_local_rank))

    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must satisfy 0 <= RANK < WORLD_SIZE, got rank={rank}, world_size={world_size}.")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}.")
    if world_size > 1 and "RANK" not in env:
        raise ValueError("RANK must be provided when WORLD_SIZE > 1.")

    return Stage1ProcessContext(rank=rank, world_size=world_size, local_rank=local_rank)


def resolve_stage1_output_paths(
    config_output: str,
    context: Stage1ProcessContext,
    timestamp: Optional[str] = None,
    resume_stage1_dir: Optional[str] = None,
    shared_run_root: Optional[str] = None,
) -> Stage1OutputPaths:
    if resume_stage1_dir and shared_run_root:
        raise ValueError("resume_stage1_dir and shared_run_root cannot be used together.")

    if resume_stage1_dir:
        run_root = os.path.abspath(resume_stage1_dir)
        if not os.path.isdir(run_root):
            raise FileNotFoundError(f"Stage-1 resume root does not exist: {run_root}")
    elif shared_run_root:
        run_root = os.path.abspath(shared_run_root)
    else:
        if not timestamp:
            raise ValueError("timestamp is required when starting a new Stage-1 run.")
        run_root = os.path.abspath(
            os.path.join(config_output, "ag_mtlora_stage1_prepare", f"run_{timestamp}")
        )

    if context.is_multi_process:
        rank_output_root = os.path.join(run_root, f"rank_{context.rank}")
        if resume_stage1_dir and not os.path.isdir(rank_output_root):
            raise FileNotFoundError(
                f"Stage-1 rank resume directory does not exist for rank_{context.rank}: {rank_output_root}"
            )
    else:
        rank_output_root = run_root

    return Stage1OutputPaths(
        run_root=run_root,
        rank_output_root=rank_output_root,
        rank_artifact_manifest_path=os.path.join(rank_output_root, RANK_ARTIFACT_MANIFEST),
        root_manifest_path=os.path.join(run_root, ROOT_MULTI_PROCESS_MANIFEST),
    )


def build_rank_artifact_manifest(
    context: Stage1ProcessContext,
    paths: Stage1OutputPaths,
    effective_seed: int,
    artifacts: Dict,
) -> Dict:
    return {
        "schema_version": 1,
        "mode": MULTI_PROCESS_MODE if context.is_multi_process else "single_process",
        "status": "complete",
        "rank": context.rank,
        "world_size": context.world_size,
        "local_rank": context.local_rank,
        "effective_seed": int(effective_seed),
        "run_root": paths.run_root,
        "rank_output_root": paths.rank_output_root,
        "artifacts": dict(artifacts),
    }


def get_failure_report_path(context: Stage1ProcessContext, paths: Stage1OutputPaths) -> str:
    filename = FAILURE_REPORT if not context.is_multi_process else f"failure_report_rank{context.rank}.json"
    return os.path.join(paths.rank_output_root, filename)


def build_rank_failure_manifest(
    context: Stage1ProcessContext,
    paths: Stage1OutputPaths,
    effective_seed: int,
    failure_report_path: str,
    error_type: str,
    error_message: str,
) -> Dict:
    return {
        "schema_version": 1,
        "mode": MULTI_PROCESS_MODE if context.is_multi_process else "single_process",
        "status": "failed",
        "rank": context.rank,
        "world_size": context.world_size,
        "local_rank": context.local_rank,
        "effective_seed": int(effective_seed),
        "run_root": paths.run_root,
        "rank_output_root": paths.rank_output_root,
        "failure_report_path": failure_report_path,
        "error_type": str(error_type),
        "error_message": str(error_message),
        "artifacts": {},
    }


def build_multi_process_failure_manifest(
    context: Stage1ProcessContext,
    paths: Stage1OutputPaths,
    failure_report_path: str,
    error_type: str,
    error_message: str,
) -> Dict:
    if not context.is_primary or not context.is_multi_process:
        raise ValueError("Only rank 0 may build the root Stage-1 failure manifest.")
    return {
        "schema_version": 1,
        "mode": MULTI_PROCESS_MODE,
        "status": "failed",
        "world_size": context.world_size,
        "failed_rank": context.rank,
        "failure_report_path": failure_report_path,
        "error_type": str(error_type),
        "error_message": str(error_message),
    }


def build_multi_process_manifest(
    context: Stage1ProcessContext,
    paths: Stage1OutputPaths,
    canonical_artifacts: Dict,
) -> Dict:
    if not context.is_primary:
        raise ValueError("Only rank 0 may build the root Stage-1 multi-process manifest.")
    if not context.is_multi_process:
        raise ValueError("The root Stage-1 multi-process manifest requires WORLD_SIZE > 1.")

    ranks = []
    for rank in range(context.world_size):
        rank_dir = os.path.join(paths.run_root, f"rank_{rank}")
        ranks.append(
            {
                "rank": rank,
                "stage1_dir": rank_dir,
                "artifact_manifest_path": os.path.join(rank_dir, RANK_ARTIFACT_MANIFEST),
            }
        )

    return {
        "schema_version": 1,
        "mode": MULTI_PROCESS_MODE,
        "world_size": context.world_size,
        "canonical_rank": 0,
        "canonical_stage1_dir": os.path.join(paths.run_root, "rank_0"),
        "canonical_resolved_config_path": canonical_artifacts.get("resolved_config_path"),
        "canonical_grouping_json_path": canonical_artifacts.get("grouping_json_path"),
        "canonical_warmup_checkpoint_path": canonical_artifacts.get("warmup_checkpoint_path"),
        "canonical_post_affinity_checkpoint_path": canonical_artifacts.get(
            "post_affinity_checkpoint_path"
        ),
        "ranks": ranks,
    }


def write_manifest(payload: Dict, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary_path, output_path)


def write_failure_report(payload: Dict, output_path: str) -> None:
    write_manifest(payload, output_path)
