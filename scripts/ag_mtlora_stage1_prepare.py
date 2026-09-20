import argparse
import datetime
import json
import math
import os
import sys
import traceback

import torch
import torch.distributed as dist

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ag_mtlora.stage1 import (
    Stage1NumericalError,
    create_stage1_logger,
    run_stage1_pipeline,
    set_random_seed,
)
from ag_mtlora.stage1_multiprocess import (
    build_multi_process_failure_manifest,
    build_multi_process_manifest,
    build_rank_failure_manifest,
    build_rank_artifact_manifest,
    get_failure_report_path,
    resolve_process_context,
    resolve_stage1_output_paths,
    write_failure_report,
    write_manifest,
)
from config import get_config
from utils import scale_learning_rates


def parse_args():
    parser = argparse.ArgumentParser("AG-MTLoRA Stage-1 preparation")
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding KEY VALUE pairs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--resume-backbone", type=str)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="output", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--fused_window_process", action="store_true")
    parser.add_argument("--fused_layernorm", action="store_true")
    parser.add_argument("--optim", type=str)
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task list.")
    parser.add_argument("--nyud", type=str)
    parser.add_argument("--pascal", type=str)
    parser.add_argument("--decoder_map", type=str)
    parser.add_argument("--skip_decoder", action="store_true")
    parser.add_argument("--resume-stage1-dir", type=str, help="Resume a previous Stage-1 output directory in-place.")
    return parser.parse_args()


def initialize_process_group(context):
    if not context.is_multi_process:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 multi-process launch requires CUDA and the NCCL backend.")

    torch.cuda.set_device(context.local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=context.world_size,
        rank=context.rank,
    )
    return True


def broadcast_primary_value(context, primary_value):
    if not context.is_multi_process:
        return primary_value
    payload = [primary_value if context.is_primary else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def configure_stage1_output(config, output_root, effective_seed):
    config.defrost()
    config.SEED = effective_seed
    config.OUTPUT = output_root
    config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH = os.path.join(output_root, "affinity.json")
    config.MODEL.AGMTLORA.GROUPING_SAVE_PATH = os.path.join(output_root, "grouping.json")
    config.MODEL.AGMTLORA.META_SPLIT_SAVE_PATH = os.path.join(output_root, "meta_split.json")
    config.freeze()


def make_json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return str(value)


def collect_runtime_diagnostics(context):
    diagnostics = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "rank": context.rank,
        "world_size": context.world_size,
        "local_rank": context.local_rank,
    }
    if not torch.cuda.is_available():
        return diagnostics

    try:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        diagnostics.update({
            "device_index": int(device_index),
            "device_name": properties.name,
            "device_total_memory": int(properties.total_memory),
            "memory_allocated": int(torch.cuda.memory_allocated(device_index)),
            "memory_reserved": int(torch.cuda.memory_reserved(device_index)),
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(device_index)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(device_index)),
        })
    except Exception as diagnostic_error:
        diagnostics["cuda_diagnostics_error"] = repr(diagnostic_error)
    return diagnostics


def flush_logger(logger):
    if logger is None:
        return
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def main():
    args = parse_args()
    context = resolve_process_context(cli_local_rank=args.local_rank)
    args.local_rank = context.local_rank
    process_group_initialized = False
    output_paths = None
    logger = None
    effective_seed = int(args.seed) if args.seed is not None else 0
    execution_context = {
        "phase": "launcher_initialization",
        "cfg": os.path.abspath(args.cfg),
        "rank": context.rank,
        "world_size": context.world_size,
        "local_rank": context.local_rank,
    }

    try:
        process_group_initialized = initialize_process_group(context)
        config = get_config(args)
        lr_scale_info = scale_learning_rates(config, context.world_size)
        effective_seed = context.effective_seed(int(config.SEED))

        if args.resume_stage1_dir:
            primary_resume_root = os.path.abspath(args.resume_stage1_dir) if context.is_primary else None
            resume_root = broadcast_primary_value(context, primary_resume_root)
            output_paths = resolve_stage1_output_paths(
                config_output=config.OUTPUT,
                context=context,
                resume_stage1_dir=resume_root,
            )
        else:
            if context.is_primary:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                primary_run_root = os.path.abspath(
                    os.path.join(
                        config.OUTPUT,
                        "ag_mtlora_stage1_prepare",
                        f"run_{timestamp}",
                    )
                )
            else:
                primary_run_root = None
            shared_run_root = broadcast_primary_value(context, primary_run_root)
            output_paths = resolve_stage1_output_paths(
                config_output=config.OUTPUT,
                context=context,
                shared_run_root=shared_run_root,
            )

        output_root = output_paths.rank_output_root
        configure_stage1_output(config, output_root, effective_seed)
        logger = create_stage1_logger(output_root, dist_rank=context.rank)

        set_random_seed(effective_seed)
        logger.info(
            "Stage-1 process context | mode=%s | rank=%d | world_size=%d | local_rank=%d | effective_seed=%d | run_root=%s | rank_output_root=%s",
            "unipora_independent_processes" if context.is_multi_process else "single_process",
            context.rank,
            context.world_size,
            context.local_rank,
            effective_seed,
            output_paths.run_root,
            output_paths.rank_output_root,
        )
        logger.info("Running AG-MTLoRA Stage-1 preparation with config:\n%s", config.dump())
        logger.info("CLI args: %s", json.dumps(vars(args), ensure_ascii=False))
        logger.info("Runtime learning-rate scaling: %s", json.dumps(lr_scale_info, sort_keys=True))

        artifacts = run_stage1_pipeline(
            config,
            output_root,
            logger,
            base_cfg_path=os.path.abspath(args.cfg),
            execution_context=execution_context,
        )
        rank_manifest = build_rank_artifact_manifest(
            context=context,
            paths=output_paths,
            effective_seed=effective_seed,
            artifacts=artifacts,
        )
        write_manifest(rank_manifest, output_paths.rank_artifact_manifest_path)

        if context.is_multi_process and context.is_primary:
            root_manifest = build_multi_process_manifest(
                context=context,
                paths=output_paths,
                canonical_artifacts=artifacts,
            )
            write_manifest(root_manifest, output_paths.root_manifest_path)
            logger.info(
                "Canonical Stage-1 multi-process manifest saved to %s",
                output_paths.root_manifest_path,
            )

        logger.info(json.dumps(artifacts, indent=2, ensure_ascii=False))
        logger.info("STAGE1_COMPLETED | output_root=%s", output_root)
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        failure_context = dict(execution_context)
        if isinstance(exc, Stage1NumericalError):
            failure_context.update(exc.details)
        last_good_checkpoint_path = None
        if output_paths is not None:
            candidate_last_good_path = os.path.join(
                output_paths.rank_output_root,
                "last_good_checkpoint.pth",
            )
            if os.path.isfile(candidate_last_good_path):
                last_good_checkpoint_path = candidate_last_good_path
        failure_report = make_json_safe({
            "schema_version": 1,
            "status": "failed",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback.format_exc(),
            "execution_context": failure_context,
            "runtime": collect_runtime_diagnostics(context),
            "args": vars(args),
            "last_good_checkpoint_path": last_good_checkpoint_path,
        })

        if logger is not None:
            logger.exception(
                "Stage-1 failure detected | error_type=%s | error=%s | context=%s",
                error_type,
                error_message,
                json.dumps(make_json_safe(failure_context), ensure_ascii=False),
            )
        else:
            traceback.print_exc()

        if output_paths is not None:
            failure_report_path = get_failure_report_path(context, output_paths)
            try:
                write_failure_report(failure_report, failure_report_path)
                failure_manifest = build_rank_failure_manifest(
                    context=context,
                    paths=output_paths,
                    effective_seed=effective_seed,
                    failure_report_path=failure_report_path,
                    error_type=error_type,
                    error_message=error_message,
                )
                write_manifest(failure_manifest, output_paths.rank_artifact_manifest_path)
                if context.is_multi_process and context.is_primary:
                    root_failure_manifest = build_multi_process_failure_manifest(
                        context=context,
                        paths=output_paths,
                        failure_report_path=failure_report_path,
                        error_type=error_type,
                        error_message=error_message,
                    )
                    write_manifest(root_failure_manifest, output_paths.root_manifest_path)
                if logger is not None:
                    logger.error("Stage-1 failure report saved to %s", failure_report_path)
            except Exception:
                if logger is not None:
                    logger.exception("Failed to persist the Stage-1 failure report.")
                else:
                    traceback.print_exc()
        terminal_message = (
            f"STAGE1_ABORTED | error_type={error_type} | error={error_message}"
        )
        if logger is not None:
            logger.error(terminal_message)
        else:
            print(terminal_message, file=sys.stderr)
        flush_logger(logger)
        raise
    finally:
        flush_logger(logger)
        if process_group_initialized and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
