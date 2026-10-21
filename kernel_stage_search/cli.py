"""CLI orchestration. Does not import the main.py training entry point."""

import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .protocol import (PROJECT_ROOT, bind_preparation, content_hash, create_base_protocol,
                       file_hash, make_manifest, read_json, read_manifest, resolved_identity, write_json,
                       validate_scoring_protocol)


def load_config(path, opts=None):
    from config import get_config
    return get_config(SimpleNamespace(cfg=str(Path(path).resolve()), opts=opts, local_rank=0))


def numeric_runtime(device):
    import torch
    device = torch.device(device)
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device_type": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": True, "amp": False, "tf32": False, "cudnn_benchmark": False}


def _immutable_json(path, payload):
    path = Path(path)
    if path.exists() and read_json(path) != payload:
        # Provenance can change while semantic content remains identical.
        old = read_json(path)
        if old.get("protocol_id") and old.get("protocol_id") == payload.get("protocol_id"):
            return old
        raise ValueError(f"Refusing to overwrite immutable artifact: {path}")
    write_json(path, payload)
    return payload


def _check_base(config, base, dataset):
    from .data import load_source_split, dataset_content_hash
    current = create_base_protocol(config, load_source_split(config), dataset_content_hash(dataset))
    if current["protocol_id"] != base["protocol_id"]:
        raise ValueError("Base protocol drift: model/data/split/config/implementation changed; run a new protocol")


def preflight(args):
    import torch
    from .data import load_source_split, build_train_view, dataset_content_hash, plan_from_config
    config = load_config(args.cfg, args.opts)
    split = load_source_split(config)  # Fails before any dataset construction if train/val lists are absent.
    dataset = build_train_view(config)
    base = create_base_protocol(config, split, dataset_content_hash(dataset))
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sample_plan = plan_from_config(config, dataset, split["meta_val_ids"])
    calibration_plan = plan_from_config(config, dataset, split["meta_train_ids"], calibration=True)
    _immutable_json(root/"sample_plan.json", sample_plan)
    _immutable_json(root/"calibration_plan.json", calibration_plan)
    config_path = root/"resolved_config.yaml"
    if config_path.exists():
        previous = load_config(config_path)
        if resolved_identity(previous) != resolved_identity(config):
            raise ValueError("Resolved configuration snapshot drift")
    else:
        snapshot = config.clone()
        snapshot.defrost()
        snapshot.BASE = ['']
        snapshot.freeze()
        config_path.write_text(snapshot.dump(), encoding="utf-8")
    base["paths"].update(config=str(config_path), sample_plan=str(root/"sample_plan.json"),
                         calibration_plan=str(root/"calibration_plan.json"))
    _immutable_json(root/"protocol.json", base)
    node = config.MODEL.KERNEL_SEARCH
    report = {"protocol_id": base["protocol_id"], "identity": base["semantic"]["identity"],
              "split_counts": {key: len(value) for key, value in split.items()}, "cuda_available": torch.cuda.is_available(),
              "torch": torch.__version__, "sample_plan_hash": sample_plan["sample_plan_id"],
              "calibration_plan_hash": calibration_plan["sample_plan_id"],
              "estimated_feature_kernel_bytes_per_candidate": node.INFO_BATCHES*4*node.SAMPLES_PER_TASK**2*4,
              "sampled_feature_dimension_448": 2112, "status": "base_preflight_only_not_bound_to_M_B"}
    write_json(root/"preflight_report.json", report)
    print(f"Base protocol: {root/'protocol.json'} ({base['protocol_id']})")
    return report


def calibrate(args):
    import torch
    from .data import build_train_view, verify_sample_plan
    from .checkpoint import load_scoring_model, register_prepared_checkpoint, state_digest
    from .calibration import calibrate_model, atomic_torch_save, validate_evaluation_labels
    root = Path(args.output_dir).resolve()
    if any((root/name).exists() for name in ("calibration.pt", "calibration.json", "scoring_protocol.json")):
        raise ValueError("Calibration output already exists; use a new directory to retain immutable artifacts")
    base = read_manifest(args.protocol, "base")
    config = load_config(args.cfg or base["paths"]["config"], args.opts)
    dataset = build_train_view(config)
    _check_base(config, base, dataset)
    bind_preparation(config, base)
    checkpoint_path = Path(args.score_checkpoint or config.MODEL.KERNEL_SEARCH.SCORE_CHECKPOINT).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing prepared BBBB checkpoint: {checkpoint_path}")
    prepared_path = checkpoint_path.with_suffix(".prepared.json")
    if not prepared_path.exists():
        preparing = config.clone()
        preparing.defrost()
        preparing.MODEL.KERNEL_SEARCH.PHASE = "prepare"
        preparing.freeze()
        register_prepared_checkpoint(preparing, checkpoint_path)
    sample_plan, calibration_plan = read_json(base["paths"]["sample_plan"]), read_json(base["paths"]["calibration_plan"])
    verify_sample_plan(sample_plan, dataset)
    verify_sample_plan(calibration_plan, dataset)
    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA scoring requested but unavailable")
    model = load_scoring_model(config, checkpoint_path, prepared_path, device)
    started = time.perf_counter()
    before = state_digest(model)
    calibration = calibrate_model(model, config, dataset, calibration_plan)
    if state_digest(model) != before:
        raise ValueError("M_B weights/buffers changed during calibration")
    validate_evaluation_labels(sample_plan, calibration, config.MODEL.KERNEL_SEARCH.ENTROPY_MIN)
    root.mkdir(parents=True, exist_ok=True)
    tensor_path = root/"calibration.pt"
    atomic_torch_save(tensor_path, calibration)
    full = make_manifest("scoring", {**base["semantic"], "base_protocol_id": base["protocol_id"],
        "mb_hash": file_hash(checkpoint_path), "prepared_manifest_id": read_manifest(prepared_path, "prepared")["protocol_id"],
        "calibration_hash": file_hash(tensor_path), "sample_plan_hash": sample_plan["sample_plan_id"],
        "calibration_plan_hash": calibration_plan["sample_plan_id"],
        "numeric_runtime": numeric_runtime(device)},
        {**base["paths"], "base_protocol": str(Path(args.protocol).resolve()), "checkpoint": str(checkpoint_path),
         "prepared_manifest": str(prepared_path), "calibration_tensors": str(tensor_path),
         "preparation_cost": str(checkpoint_path.parent/"preparation_cost.json"),
         "calibration_cost": str(root/"calibration_cost.json"),
         "kernel_cache": str(root.parent/"feature_kernels")}, base["provenance"])
    protocol_path = root/"scoring_protocol.json"
    _immutable_json(protocol_path, full)
    description = {"schema": 1, "protocol_path": str(protocol_path), "protocol_id": full["protocol_id"],
                   "tensor_path": str(tensor_path), "tensor_hash": full["semantic"]["calibration_hash"],
                   "checkpoint": str(checkpoint_path)}
    write_json(root/"calibration.json", description)
    write_json(root/"calibration_cost.json", {"phase": "calibration", "seconds": time.perf_counter()-started,
               "batches": len(calibration_plan["batches"]), "image_memberships": sum(len(b["image_ids"]) for b in calibration_plan["batches"]),
               "scope": "calibration forwards, kernels and artifact writes; excludes data preflight and model loading"})
    print(f"Scoring protocol: {protocol_path} ({full['protocol_id']})")
    return description


def search(args):
    import torch
    from filelock import FileLock
    from .data import build_train_view, verify_sample_plan
    from .checkpoint import load_scoring_model
    from .calibration import load_calibration, validate_evaluation_labels
    from .runner import FrozenScorer, run_search
    description = read_json(args.calibration)
    protocol = read_manifest(description["protocol_path"], "scoring")
    validate_scoring_protocol(protocol)
    if protocol["semantic"].get("synthetic"):
        raise ValueError("Synthetic fixtures cannot be used by the research scoring CLI")
    base = read_manifest(args.protocol)
    expected_id = protocol["semantic"]["base_protocol_id"] if base["kind"] == "base" else protocol["protocol_id"]
    if base["protocol_id"] != expected_id or description["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Calibration and requested protocol do not match")
    original_base = read_manifest(protocol["paths"]["base_protocol"], "base")
    config = load_config(protocol["paths"]["config"])
    dataset = build_train_view(config)
    _check_base(config, original_base, dataset)
    bind_preparation(config, original_base)
    runtime = protocol["semantic"]["numeric_runtime"]
    if runtime != numeric_runtime(args.device):
        raise ValueError("Scoring numeric runtime changed; calibrate a new protocol")
    prepared = read_manifest(protocol["paths"]["prepared_manifest"], "prepared")
    if prepared["protocol_id"] != protocol["semantic"]["prepared_manifest_id"]:
        raise ValueError("Prepared manifest identity drift")
    calibration = load_calibration(protocol["paths"]["calibration_tensors"], protocol["semantic"]["calibration_hash"])
    sample_plan = read_json(protocol["paths"]["sample_plan"])
    verify_sample_plan(sample_plan, dataset)
    if sample_plan["sample_plan_id"] != protocol["semantic"]["sample_plan_hash"]:
        raise ValueError("Sample plan differs from full scoring protocol")
    validate_evaluation_labels(sample_plan, calibration)
    model = load_scoring_model(config, protocol["paths"]["checkpoint"], prepared, args.device)
    cache_root = Path(args.cache_dir or protocol["paths"]["kernel_cache"])
    cache_root.mkdir(parents=True, exist_ok=True)
    # Two CLI processes cannot race on the same shared feature-kernel files.
    with FileLock(str(cache_root/(protocol["protocol_id"]+".lock")), timeout=0):
        scorer = FrozenScorer(model, config, dataset, sample_plan, calibration, protocol, cache_root)
        winner = run_search(protocol, args.output_dir, "true_labels" if args.labels == "true" else "shuffled", scorer)
    print(f"Observed winner: {winner['modes']} J={winner['J']:.9g}, {winner['budget']} observations")
    return winner


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen task-kernel stage search (no candidate training)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "calibrate"):
        child = subparsers.add_parser(command)
        child.add_argument("--cfg", required=command == "preflight")
        child.add_argument("--opts", nargs="+")
        child.add_argument("--output-dir", required=True)
        if command == "calibrate":
            child.add_argument("--protocol", required=True)
            child.add_argument("--score-checkpoint")
            child.add_argument("--device", default="cuda")
        child.set_defaults(handler=preflight if command == "preflight" else calibrate)
    child = subparsers.add_parser("search")
    child.add_argument("--protocol", required=True)
    child.add_argument("--calibration", required=True)
    child.add_argument("--labels", choices=("true", "shuffled"), required=True)
    child.add_argument("--device", default="cuda")
    child.add_argument("--cache-dir")
    child.add_argument("--output-dir", required=True)
    child.set_defaults(handler=search)
    child = subparsers.add_parser("export")
    child.add_argument("--true-run", required=True)
    child.add_argument("--shuffled-run", required=True)
    child.add_argument("--output-dir", required=True)
    def export_handler(args):
        from .export import export_formal_configs
        result = export_formal_configs(args.true_run, args.shuffled_run, args.output_dir)
        print(f"Exported {len(result['runs'])} unique formal runs to {args.output_dir}")
        return result
    child.set_defaults(handler=export_handler)
    args = parser.parse_args(argv)
    # Existing project dataset code resolves db_info relative to the repository.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.chdir(PROJECT_ROOT)
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"{args.command} failed: {error}\n")
