"""Strict old meta split, official-train views, and immutable pixel plans."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from .protocol import TASKS, PROJECT_ROOT, content_hash, file_hash, read_json


def _unique(ids, label):
    ids = [str(value) for value in ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty/duplicate IDs in {label}")
    return ids


def validate_split(source, official_train, official_val):
    train = _unique(source.get("meta_train_ids", []), "meta-train")
    dev = _unique(source.get("meta_val_ids", []), "meta-val")
    official_train = _unique(official_train, "official train")
    official_val = _unique(official_val, "official val")
    if set(train) & set(dev):
        raise ValueError("meta-train/meta-val intersection")
    if set(train) | set(dev) != set(official_train):
        raise ValueError("Source IDs must completely cover official train; missing/unknown IDs")
    if (set(train) | set(dev)) & set(official_val):
        raise ValueError("Official validation leakage")
    return {"meta_train_ids": train, "meta_val_ids": dev,
            "official_train_ids": official_train, "official_val_ids": official_val}


def load_source_split(config):
    if config.DATA.DBNAME != "PASCALContext":
        raise ValueError("Only the fixed PASCALContext protocol is supported")
    directory = Path(config.DATA.DATA_PATH) / "ImageSets" / "Context"
    # Read validation IDs only. Never construct an official-val dataset.
    def read_ids(name):
        return [line.strip() for line in (directory / (name + ".txt")).read_text(encoding="utf-8").splitlines() if line.strip()]
    return validate_split(read_json(config.MODEL.KERNEL_SEARCH.SOURCE_META_SPLIT), read_ids("train"), read_ids("val"))


def index_by_id(dataset):
    ids = _unique(dataset.im_ids, "dataset")
    return {name: index for index, name in enumerate(ids)}


def subset_by_id(dataset, ids):
    indices = index_by_id(dataset)
    missing = set(ids) - indices.keys()
    if missing:
        raise ValueError(f"Dataset is missing source IDs: {sorted(missing)[:10]}")
    return Subset(dataset, [indices[name] for name in ids])


def build_train_view(config, training=False):
    from data.mtl_ds import get_transformations, get_mtl_dataset
    transforms = get_transformations(config.DATA.DBNAME, config.TASKS_CONFIG)
    dataset = get_mtl_dataset(config.DATA.DBNAME, config, transforms[0 if training else 1], split="train")
    if set(index_by_id(dataset)) != set(load_source_split(config)["official_train_ids"]):
        raise ValueError("Dataset must expose every official-train ID without filtering")
    return dataset


def dataset_content_hash(dataset):
    """Hash source bytes, including auxiliary mappings, independently of root path."""
    records = {}
    ids = _unique(dataset.im_ids, "dataset")
    for attribute in ("images", "edges", "semsegs", "parts", "normals", "sals"):
        paths = getattr(dataset, attribute, [])
        if paths and len(paths) != len(ids):
            raise ValueError(f"Source file list length mismatch: {attribute}")
        if paths:
            records[attribute] = {name: file_hash(path) for name, path in zip(ids, paths)}
    if not records:
        raise ValueError("Dataset must expose input files for content fingerprinting")
    if hasattr(dataset, "parts_file"):
        records["parts_index"] = file_hash(dataset.parts_file)
    records["db_info"] = {path.name: file_hash(path) for path in sorted((PROJECT_ROOT / "data" / "db_info").glob("*.json"))}
    return content_hash(records)


def build_prepare_train_loader(config):
    from data.mtl_ds import get_mtl_train_dataloader
    from .protocol import create_base_protocol, bind_preparation, write_json
    split = load_source_split(config)
    dataset = build_train_view(config, training=True)
    base = create_base_protocol(config, split, dataset_content_hash(dataset))
    if config.MODEL.KERNEL_SEARCH.PROTOCOL:
        from .protocol import read_manifest
        supplied = read_manifest(config.MODEL.KERNEL_SEARCH.PROTOCOL, "base")
        if supplied["protocol_id"] != base["protocol_id"]:
            raise ValueError("Preparation preflight protocol drift")
    bind_preparation(config, base)
    write_json(Path(config.OUTPUT) / "preparation_protocol.json", base)
    subset = subset_by_id(dataset, split["meta_train_ids"])
    loader = get_mtl_train_dataloader(config, subset)
    if not len(loader):
        raise ValueError("Preparation loader has no full training batches")
    return subset, None, loader, None, None


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def valid_labels(task, label):
    array = _array(label).astype(np.float64, copy=False)
    if task == "normals":
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError("Normals must be [3,H,W]")
        vectors = array.transpose(1, 2, 0).copy()
        norms = np.linalg.norm(vectors, axis=-1)
        mask = np.isfinite(vectors).all(axis=-1) & np.isfinite(norms) & (vectors != 255).all(axis=-1) & (norms > 1e-6)
        vectors[mask] /= norms[mask, None]
        vectors[~mask] = 0
        return vectors, mask
    if task not in ("semseg", "sal", "human_parts"):
        raise ValueError(f"Unknown task: {task}")
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Categorical labels must be finite [1,H,W] or [H,W]")
    mask = array != 255
    maximum = {"semseg": 20, "human_parts": 6, "sal": 1}[task]
    selected = array[mask]
    if ((selected < 0) | (selected > maximum) | (selected != np.floor(selected))).any():
        raise ValueError(f"Invalid {task} category after evaluation transforms")
    return array.astype(np.int64), mask


def sample_fingerprint(sample):
    digest = hashlib.sha256()
    for key in ("image", *TASKS):
        array = np.ascontiguousarray(_array(sample[key]))
        digest.update(content_hash({"key": key, "shape": list(array.shape), "dtype": str(array.dtype)}).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_sample_plan(dataset, ids, num_batches, initial_images, max_images, samples, seed=0, shuffle_seed=1):
    if min(num_batches, initial_images, max_images, samples) < 1 or max_images < initial_images:
        raise ValueError("Invalid sample plan sizes")
    indices = index_by_id(dataset)
    ids = _unique(ids, "sample pool")
    if set(ids) - indices.keys() or len(ids) < initial_images:
        raise ValueError("Missing/insufficient sample pool IDs")
    rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(shuffle_seed)
    order = [ids[index] for index in rng.permutation(len(ids))]
    fingerprints, batches = {}, []
    for batch_index in range(num_batches):
        members, task_positions, task_labels, shapes = [], {t: [] for t in TASKS}, {t: [] for t in TASKS}, {}
        enough = varied = False
        # Fixed start and forward expansion; batches can overlap when expansion is needed.
        for offset in range(min(max_images, len(order))):
            name = order[(batch_index * initial_images + offset) % len(order)]
            sample = dataset[indices[name]]
            if str(sample["meta"]["image"]) != name:
                raise ValueError("Dataset image ID mismatch")
            image = _array(sample["image"])
            if image.ndim != 3 or image.shape[0] != 3 or not np.isfinite(image).all():
                raise ValueError("Expected a finite transformed [3,H,W] image")
            fingerprint = sample_fingerprint(sample)
            if name in fingerprints and fingerprints[name] != fingerprint:
                raise ValueError("Evaluation transforms changed a previously read sample")
            fingerprints[name] = fingerprint
            members.append(name)
            for task in TASKS:
                labels, mask = valid_labels(task, sample[task])
                if mask.shape != image.shape[-2:]:
                    raise ValueError("Image and all transformed label geometries must match")
                height, width = mask.shape
                shapes[name] = [height, width]
                ys, xs = np.where(mask)
                task_positions[task].append((name, ys, xs))
                task_labels[task].append(labels[ys, xs])
            enough = all(sum(len(part) for part in task_labels[t]) >= samples for t in TASKS)
            def has_variation(parts):
                nonempty = [part for part in parts if len(part)]
                return bool(nonempty) and any(np.any(part != nonempty[0][0]) for part in nonempty)
            varied = all(has_variation(task_labels[t]) for t in TASKS)
            if len(members) >= initial_images and enough and varied:
                break
        if not enough or not varied:
            raise ValueError(f"Insufficient valid positions or degenerate label entropy in batch {batch_index}")
        tasks = {}
        for task in TASKS:
            lengths = [len(part) for part in task_labels[task]]
            cumulative = np.cumsum([0, *lengths])
            rows = rng.choice(int(cumulative[-1]), samples, replace=False)
            labels, coordinates = [], []
            for row in rows:
                image_index = int(np.searchsorted(cumulative[1:], row, side="right"))
                local = int(row-cumulative[image_index])
                name, ys, xs = task_positions[task][image_index]
                labels.append(task_labels[task][image_index][local].tolist())
                coordinates.append([name, int(ys[local]), int(xs[local])])
            if len(np.unique(np.asarray(labels), axis=0)) < 2:
                raise ValueError(f"Degenerate sampled labels: batch={batch_index}, task={task}")
            unique, counts = np.unique(np.asarray(labels), axis=0, return_counts=True)
            tasks[task] = {"coordinates": coordinates, "labels": labels,
                           "permutation": permutation_rng.permutation(samples).tolist(),
                           "label_distribution": {"values": unique.tolist(), "counts": counts.tolist()},
                           "valid_positions": int(cumulative[-1])}
        batches.append({"batch": batch_index, "image_ids": members, "label_shapes": shapes, "tasks": tasks})
    payload = {"schema": 1, "pool_ids": ids, "pool_order": order, "seed": seed, "shuffle_seed": shuffle_seed,
               "samples_per_task": samples, "batches": batches, "fingerprints": fingerprints}
    return {**payload, "sample_plan_id": content_hash(payload)}


def verify_sample_plan(plan, dataset=None):
    payload = {key: value for key, value in plan.items() if key != "sample_plan_id"}
    if content_hash(payload) != plan.get("sample_plan_id"):
        raise ValueError("Sample plan hash mismatch")
    for batch in plan["batches"]:
        for task in TASKS:
            entry = batch["tasks"][task]
            count = plan["samples_per_task"]
            if (len(entry["coordinates"]) != count or len(entry["labels"]) != count
                    or sorted(entry["permutation"]) != list(range(count))):
                raise ValueError("Invalid sample rows/permutation")
    if dataset is not None:
        indices = index_by_id(dataset)
        for name, expected in plan["fingerprints"].items():
            if name not in indices:
                raise ValueError(f"Input content fingerprint drift: {name}")
            sample = dataset[indices[name]]
            if sample_fingerprint(sample) != expected:
                raise ValueError(f"Input content fingerprint drift: {name}")
            for task in TASKS:
                labels, mask = valid_labels(task, sample[task])
                for batch in plan["batches"]:
                    entry = batch["tasks"][task]
                    for index, (image, y, x) in enumerate(entry["coordinates"]):
                        if image != name:
                            continue
                        if (name not in batch["image_ids"] or batch["label_shapes"][name] != list(mask.shape)
                                or not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or not mask[y, x]
                                or not np.allclose(labels[y, x], entry["labels"][index], atol=1e-7, rtol=0)):
                            raise ValueError(f"Sample row/label content mismatch: {name}, {task}")


def plan_from_config(config, dataset, ids, calibration=False):
    node = config.MODEL.KERNEL_SEARCH
    if calibration:
        if len(ids) < node.CALIBRATION_IMAGES:
            raise ValueError("Not enough meta-train images for calibration")
        rng = np.random.default_rng(node.SAMPLE_SEED)
        ids = [ids[index] for index in rng.choice(len(ids), node.CALIBRATION_IMAGES, replace=False)]
        initial = node.FORWARD_BATCH_SIZE
        count = (len(ids) + initial - 1) // initial
    else:
        initial, count = node.INFO_BATCH_IMAGES, node.INFO_BATCHES
    return build_sample_plan(dataset, ids, count, initial, node.MAX_BATCH_IMAGES,
                             node.SAMPLES_PER_TASK, node.SAMPLE_SEED, node.SHUFFLE_SEED)
