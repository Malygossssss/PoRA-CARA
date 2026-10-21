"""Sample actual post-merging stage tokens during full task-routed forwards."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .data import index_by_id
from .protocol import TASKS


@contextmanager
def scoring_numerics():
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)


def sample_stage(tokens, resolution, image_ids, coordinates, label_shapes):
    height, width = map(int, resolution)
    if tokens.ndim != 3 or tokens.shape[0] != len(image_ids) or tokens.shape[1] != height*width:
        raise ValueError("Stage token count does not match layer.output_resolution (possible prompt tokens)")
    if not torch.isfinite(tokens).all():
        raise ValueError("Non-finite stage features")
    maps = tokens.transpose(1, 2).reshape(len(image_ids), tokens.shape[-1], height, width)
    values, rows = [], []
    for image_index, name in enumerate(image_ids):
        selected = [(index, y, x) for index, (image, y, x) in enumerate(coordinates) if image == name]
        if not selected:
            continue
        label_h, label_w = label_shapes[name]
        if any(y < 0 or y >= label_h or x < 0 or x >= label_w for _, y, x in selected):
            raise ValueError("Sample coordinates outside transformed label geometry")
        grid = tokens.new_tensor([[2*(x+0.5)/label_w-1, 2*(y+0.5)/label_h-1] for _, y, x in selected])
        grid = grid.reshape(1, -1, 1, 2)
        sampled = F.grid_sample(maps[image_index:image_index+1], grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        values.append(sampled[0, :, :, 0].T)
        rows.extend(index for index, _, _ in selected)
    return (torch.cat(values, dim=0) if values else tokens.new_empty((0, tokens.shape[-1])), rows)


def collect_batch_features(model, dataset, batch_plan, microbatch=4, stats=None):
    if microbatch < 1 or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Scoring requires positive microbatch, eval mode and fully frozen parameters")
    indices = index_by_id(dataset)
    ids = batch_plan["image_ids"]
    if len(ids) != len(set(ids)) or set(ids)-indices.keys():
        raise ValueError("Invalid batch image IDs")
    device = next(model.parameters()).device
    result, written = {task: None for task in TASKS}, {task: set() for task in TASKS}
    current_ids, seen = [], set()

    def hook(backbone, args, kwargs, outputs):
        task = kwargs.get("task")
        if task not in TASKS or task in seen or kwargs.get("return_stages") is not True:
            raise ValueError("Expected each task exactly once with return_stages=True")
        seen.add(task)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 4 or len(backbone.layers) != 4:
            raise ValueError("Expected four actual backbone stage features")
        coordinates = batch_plan["tasks"][task]["coordinates"]
        for stage, (tokens, layer) in enumerate(zip(outputs, backbone.layers)):
            features, rows = sample_stage(tokens, layer.output_resolution, current_ids, coordinates,
                                          batch_plan["label_shapes"])
            if result[task] is None:
                result[task] = [None]*4
            if result[task][stage] is None:
                result[task][stage] = torch.empty(len(coordinates), tokens.shape[-1], dtype=torch.float32)
            if stats is not None:
                mean, std = stats[task][stage]["mean"].to(features), stats[task][stage]["std"].to(features)
                features = (features-mean)/std
            result[task][stage][rows] = features.float().cpu()
            if stage == 0:
                if set(rows) & written[task]:
                    raise ValueError("Sample row collected twice")
                written[task].update(rows)

    handle = model.backbone.register_forward_hook(hook, with_kwargs=True)
    try:
        with scoring_numerics(), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, len(ids), microbatch):
                current_ids = ids[start:start+microbatch]
                seen = set()
                images = []
                for name in current_ids:
                    sample = dataset[indices[name]]
                    if str(sample["meta"]["image"]) != name:
                        raise ValueError("Image ID mismatch during feature collection")
                    images.append(sample["image"])
                # Execute the real model, including existing task routing/decoder.
                model(torch.stack(images).to(device=device, dtype=torch.float32))
                if seen != set(TASKS):
                    raise ValueError("Missing task forwards")
    finally:
        handle.remove()
    for task in TASKS:
        expected = len(batch_plan["tasks"][task]["coordinates"])
        if written[task] != set(range(expected)) or any(not torch.isfinite(x).all() for x in result[task]):
            raise ValueError("Incomplete/non-finite sampled features")
    return result


def concatenate_features(stages):
    return {task: torch.cat(stages[task], dim=1).float() for task in TASKS}
