import argparse
import datetime
import json
import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ag_mtlora.stage1 import create_stage1_logger, run_stage1_pipeline, set_random_seed
from ag_mtlora.stage1_multiprocess import (
    build_multi_process_manifest,
    build_rank_artifact_manifest,
    resolve_process_context,
    resolve_stage1_output_paths,
    write_manifest,
)
from config import get_config


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


def main():
    args = parse_args()
    context = resolve_process_context(cli_local_rank=args.local_rank)
    args.local_rank = context.local_rank
    process_group_initialized = False

    try:
        process_group_initialized = initialize_process_group(context)
        config = get_config(args)
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

        artifacts = run_stage1_pipeline(
            config,
            output_root,
            logger,
            base_cfg_path=os.path.abspath(args.cfg),
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

        logger.info("AG-MTLoRA Stage-1 preparation finished.")
        logger.info(json.dumps(artifacts, indent=2, ensure_ascii=False))
    finally:
        if process_group_initialized and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
