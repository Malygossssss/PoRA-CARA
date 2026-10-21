"""One checkpoint, one validation, one fixed formal endpoint per paired run."""

import math
from pathlib import Path
import statistics

from .protocol import content_hash, file_hash, read_json, write_json, plain

UNITS = {"semseg_miou": "fraction", "human_miou": "fraction", "saliency_miou": "fraction", "normals_mean": "degrees"}
METRICS = tuple(UNITS)


def write_formal_endpoint(config, epoch, checkpoint_path, validation):
    import torch
    from stage_ablation import stage_ablation_metadata
    if epoch != 299 or config.TRAIN.EPOCHS != 300:
        raise ValueError("Formal report requires the fixed epoch 299 endpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != epoch or checkpoint.get("kernel_search_preparation") is not None:
        raise ValueError("Formal endpoint cannot reference preparation or another epoch")
    identity = stage_ablation_metadata(config)
    if checkpoint.get("stage_ablation") != identity:
        raise ValueError("Formal checkpoint identity mismatch")
    metrics = {"semseg_miou": float(validation["semseg"]["mIoU"]),
               "human_miou": float(validation["human_parts"]["mIoU"]),
               "saliency_miou": float(validation["sal"]["mIoU"]),
               "normals_mean": float(validation["normals"]["mean"])}
    record = {"schema": 1, "epoch": epoch, "modes": "".join(identity["modes"]), "seed": config.SEED,
              "c0_hash": config.MODEL.INIT_CHECKPOINT_SHA256, "checkpoint": str(Path(checkpoint_path).resolve()),
              "checkpoint_hash": file_hash(checkpoint_path), "metrics": metrics, "units": UNITS,
              "training_identity": identity,
              "optional_saliency": {k: float(validation["sal"][k]) for k in ("maxF", "Beta maxF") if k in validation["sal"]}}
    record["record_hash"] = content_hash(record)
    write_json(Path(config.OUTPUT)/"formal_endpoint_299.json", record)
    return record


def _summary(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def summarize_records(records, groups):
    indexed = {}
    for record in records:
        if record["epoch"] != 299:
            raise ValueError("Every task must come from the same fixed epoch 299")
        if record["units"] != UNITS or set(record["metrics"]) != set(METRICS):
            raise ValueError("Raw metric units/keys mismatch")
        for name, value in record["metrics"].items():
            if not math.isfinite(value) or value < 0 or value > (180 if name == "normals_mean" else 1):
                raise ValueError("Invalid raw metric value/unit")
        key = (record["modes"], record["seed"])
        if key in indexed:
            raise ValueError("Duplicate run; identical structure/seed must reuse a shared artifact")
        indexed[key] = record
    required = {(modes, seed) for modes in groups.values() for seed in (0, 1, 2)}
    if set(indexed) != required or len({r["c0_hash"] for r in records}) != 1:
        raise ValueError("All groups need paired seeds 0/1/2 from the same C0")
    result = {}
    for group, modes in groups.items():
        rows = [indexed[(modes, seed)] for seed in (0, 1, 2)]
        paired = [indexed[(groups["BBBB"], seed)] for seed in (0, 1, 2)]
        result[group] = {"modes": modes, "per_seed": rows,
                         "metrics": {name: _summary([r["metrics"][name] for r in rows]) for name in METRICS},
                         "paired_differences": {name: _summary([a["metrics"][name]-b["metrics"][name] for a, b in zip(rows, paired)]) for name in METRICS},
                         "shared_modes_with": [name for name, value in groups.items() if name != group and value == modes]}
    return {"schema": 1, "epoch": 299, "units": UNITS, "groups": result,
            "difference_direction": "run minus paired BBBB; positive mIoU and negative normals error improve",
            "unique_training_runs": len(indexed)}


def summarize_files(formal_manifest_path, result_paths, output_path):
    manifest = read_json(formal_manifest_path)
    if manifest.get("manifest_hash") != content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("Formal manifest hash mismatch")
    records, training_costs = [], {}
    for path in result_paths:
        record = read_json(path)
        if (record.get("record_hash") != content_hash({k: v for k, v in record.items() if k != "record_hash"})
                or file_hash(record["checkpoint"]) != record["checkpoint_hash"]):
            raise ValueError("Endpoint record/checkpoint hash mismatch")
        import torch
        checkpoint = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        checkpoint_identity = plain(checkpoint.get("stage_ablation"))
        if (checkpoint.get("epoch") != 299 or checkpoint.get("kernel_search_preparation") is not None
                or checkpoint_identity != record.get("training_identity")
                or "".join(checkpoint_identity["modes"]) != record["modes"]
                or checkpoint_identity["protocol"]["seed"] != record["seed"]):
            raise ValueError("Endpoint checkpoint epoch/preparation/training identity mismatch")
        del checkpoint
        expected = next((run for run in manifest["runs"] if run["modes"] == record["modes"] and run["seed"] == record["seed"]), None)
        if expected is None or file_hash(expected["config"]) != expected["config_hash"]:
            raise ValueError("Result is not one of the locked formal configurations")
        cost_path = Path(path).parent/"training_cost.json"
        training_costs[f"{record['modes']}_seed{record['seed']}"] = read_json(cost_path) if cost_path.exists() else {"status": "unknown_or_not_recorded"}
        records.append(record)
    result = summarize_records(records, manifest["groups"])
    if records[0]["c0_hash"] != manifest["c0_hash"]:
        raise ValueError("Formal results use a different C0")
    result["protocol_id"] = manifest["protocol_id"]
    result["search_costs"] = manifest["search_costs"]
    result["training_costs"] = training_costs
    write_json(output_path, result)
    return result
