"""Training-side BBBB statistics and fixed kernel bandwidths."""

from pathlib import Path
import os
import tempfile

import numpy as np
import torch

from .features import collect_batch_features
from .kernels import median_positive_distance, label_kernels, normalized_information
from .policy import temporary_stage_modes
from .protocol import TASKS, file_hash


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+".", suffix=".tmp")
    os.close(fd)
    try:
        # A file handle gives torch's ZIP archive a stable internal prefix.
        # Passing a random temporary filename would change the bytes/hash even
        # for identical tensors saved in another output directory.
        with open(temporary, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fit_calibration(batches, bandwidth_samples=1024, seed=0, std_floor=1e-6):
    """Welford statistics plus a fixed uniform reservoir, using only meta-train rows.

    Each item is (four-task stage samples, normal labels). The reservoir uses
    random priorities, so memory stays bounded independently of calibration size.
    """
    rng = np.random.default_rng(seed)
    accumulators, reservoirs = {}, {}
    normal_rows, normal_priorities = torch.empty(0, 3), np.empty(0)
    for stages, normals in batches:
        for task in TASKS:
            if task not in accumulators:
                accumulators[task] = [{"n": 0, "mean": None, "m2": None} for _ in range(4)]
            for accumulator, values in zip(accumulators[task], stages[task]):
                values = values.double().cpu()
                n = len(values)
                mean, m2 = values.mean(dim=0), ((values-values.mean(dim=0)).square()).sum(dim=0)
                if accumulator["n"] == 0:
                    accumulator.update(n=n, mean=mean, m2=m2)
                else:
                    old = accumulator["n"]
                    delta = mean-accumulator["mean"]
                    accumulator["mean"] += delta*(n/(old+n))
                    accumulator["m2"] += m2+delta.square()*(old*n/(old+n))
                    accumulator["n"] += n
            rows = torch.cat(stages[task], dim=1).float().cpu()
            priorities = rng.random(len(rows))
            previous, old_priorities = reservoirs.get(task, (rows[:0], np.empty(0)))
            rows, priorities = torch.cat((previous, rows)), np.concatenate((old_priorities, priorities))
            chosen = np.argsort(priorities, kind="stable")[:bandwidth_samples]
            reservoirs[task] = (rows[chosen], priorities[chosen])
        normals = torch.as_tensor(normals).float()
        normal_rows = torch.cat((normal_rows, normals))
        normal_priorities = np.concatenate((normal_priorities, rng.random(len(normals))))
        chosen = np.argsort(normal_priorities, kind="stable")[:bandwidth_samples]
        normal_rows, normal_priorities = normal_rows[chosen], normal_priorities[chosen]
    if set(accumulators) != set(TASKS):
        raise ValueError("Calibration requires all four tasks")
    stats, widths = {}, {}
    for task in TASKS:
        stats[task] = [{"mean": entry["mean"].float(),
                        "std": (entry["m2"]/entry["n"]).clamp_min(0).sqrt().clamp_min(std_floor).float(),
                        "n": entry["n"]} for entry in accumulators[task]]
        mean = torch.cat([entry["mean"] for entry in stats[task]])
        std = torch.cat([entry["std"] for entry in stats[task]])
        widths[task] = median_positive_distance((reservoirs[task][0]-mean)/std)
    return {"schema": 1, "stats": stats, "sigma_features_squared": widths,
            "sigma_normals_squared": median_positive_distance(normal_rows),
            "bandwidth_samples": bandwidth_samples, "seed": seed, "std_floor": std_floor}


def calibrate_model(model, config, dataset, plan):
    def batches():
        for batch in plan["batches"]:
            yield (collect_batch_features(model, dataset, batch, config.MODEL.KERNEL_SEARCH.FORWARD_BATCH_SIZE),
                   batch["tasks"]["normals"]["labels"])
    with temporary_stage_modes(model, config, "BBBB"):
        return fit_calibration(batches(), config.MODEL.KERNEL_SEARCH.BANDWIDTH_SAMPLES,
                               config.MODEL.KERNEL_SEARCH.SAMPLE_SEED, config.MODEL.KERNEL_SEARCH.STD_FLOOR)


def validate_evaluation_labels(sample_plan, calibration, entropy_min=1e-8):
    for batch in sample_plan["batches"]:
        for task, kernel in label_kernels(batch, calibration["sigma_normals_squared"]).items():
            try:
                normalized_information(torch.ones_like(kernel), kernel, entropy_min)
            except ValueError as error:
                raise ValueError(f"Invalid fixed evaluation label entropy: batch={batch['batch']} task={task}") from error


def load_calibration(path, expected_hash):
    if file_hash(path) != expected_hash:
        raise ValueError("Calibration tensor hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or set(payload["stats"]) != set(TASKS):
        raise ValueError("Invalid calibration schema")
    for task in TASKS:
        if len(payload["stats"][task]) != 4 or payload["sigma_features_squared"][task] <= 0:
            raise ValueError("Invalid calibration task/stage/bandwidth")
        for stage in payload["stats"][task]:
            if (not torch.isfinite(stage["mean"]).all() or not torch.isfinite(stage["std"]).all()
                    or (stage["std"] < payload["std_floor"]).any()):
                raise ValueError("Invalid calibration mean/std")
    return payload
