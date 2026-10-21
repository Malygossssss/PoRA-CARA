"""Strict M_B registration/loading, with no shared-to-group conversion."""

import hashlib
from pathlib import Path

import torch

from .protocol import (content_hash, file_hash, make_manifest, preparation_metadata,
                       read_manifest, resolved_identity, write_json)


def state_schema(state):
    return {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in sorted(state.items())}


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load(path):
    # Checkpoints are trusted local project artifacts and contain a YACS config.
    return torch.load(path, map_location="cpu", weights_only=False)


def register_prepared_checkpoint(config, path, checkpoint=None, output_path=None):
    from stage_ablation import stage_ablation_metadata
    checkpoint = checkpoint if checkpoint is not None else _load(path)
    expected = preparation_metadata(config)
    actual = checkpoint.get("kernel_search_preparation", {})
    endpoint = config.MODEL.KERNEL_SEARCH.PREPARE_EPOCHS - 1
    if (expected is None or actual.get("schema") != expected["schema"] or actual.get("identity") != expected["identity"]
            or not actual.get("prepared_complete") or actual.get("completed_epoch") != endpoint
            or checkpoint.get("epoch") != endpoint or checkpoint.get("stage_ablation") != stage_ablation_metadata(config)
            or any(key not in checkpoint for key in ("optimizer", "lr_scheduler", "scaler"))):
        raise ValueError("Only a verified fixed preparation endpoint can be registered as M_B")
    manifest = make_manifest("prepared", {"checkpoint_hash": file_hash(path), "preparation": actual,
                                          "state_schema": state_schema(checkpoint["model"])},
                             {"checkpoint": str(Path(path).resolve())})
    write_json(output_path or Path(path).with_suffix(".prepared.json"), manifest)
    return manifest


def verify_prepared(config, checkpoint_path, prepared_manifest):
    from stage_ablation import stage_ablation_metadata
    manifest = read_manifest(prepared_manifest, "prepared") if isinstance(prepared_manifest, (str, Path)) else prepared_manifest
    # Validate in-memory manifests too.
    identity = {key: manifest[key] for key in ("schema", "kind", "semantic")}
    if content_hash(identity) != manifest.get("protocol_id") or manifest.get("kind") != "prepared":
        raise ValueError("Prepared manifest hash mismatch")
    semantic = manifest["semantic"]
    if file_hash(checkpoint_path) != semantic["checkpoint_hash"]:
        raise ValueError("M_B checkpoint hash mismatch")
    preparation = semantic["preparation"]
    expected_identity = preparation["identity"]
    node = config.MODEL.KERNEL_SEARCH
    if not node.RUNTIME_PROTOCOL_ID or not node.RUNTIME_SPLIT_ID:
        raise ValueError("Bind the verified base/split protocol before loading M_B")
    if (preparation.get("schema") != 1 or not preparation["prepared_complete"] or preparation["completed_epoch"] != node.PREPARE_EPOCHS-1
            or expected_identity["modes"] != "BBBB" or expected_identity["phase"] != "prepare"
            or expected_identity["prepare_epochs"] != node.PREPARE_EPOCHS
            or expected_identity["c0_hash"] != config.MODEL.INIT_CHECKPOINT_SHA256
            or expected_identity["scheduler_horizon"] != config.TRAIN.EPOCHS
            or expected_identity["identity"] != resolved_identity(config)):
        raise ValueError("M_B preparation/model identity mismatch")
    if node.RUNTIME_PROTOCOL_ID and expected_identity["base_protocol_id"] != node.RUNTIME_PROTOCOL_ID:
        raise ValueError("M_B base protocol mismatch")
    if node.RUNTIME_SPLIT_ID and expected_identity["split_id"] != node.RUNTIME_SPLIT_ID:
        raise ValueError("M_B split mismatch")
    checkpoint = _load(checkpoint_path)
    bbbb = config.clone()
    bbbb.defrost()
    bbbb.MODEL.STAGE_ABLATION.MODES = list("BBBB")
    bbbb.freeze()
    if (checkpoint.get("kernel_search_preparation") != preparation
            or checkpoint.get("epoch") != preparation["completed_epoch"]
            or checkpoint.get("stage_ablation") != stage_ablation_metadata(bbbb)
            or state_schema(checkpoint["model"]) != semantic["state_schema"]):
        raise ValueError("M_B checkpoint metadata/schema mismatch")
    return checkpoint, manifest


def load_scoring_model(config, checkpoint_path, prepared_manifest, device):
    from models import build_model, build_mtl_model
    from stage_ablation import configure_stage_ablation
    checkpoint, _ = verify_prepared(config, checkpoint_path, prepared_manifest)
    candidate = config.clone()
    candidate.defrost()
    candidate.MODEL.STAGE_ABLATION.MODES = list("BBBB")
    candidate.TRAIN.USE_CHECKPOINT = False
    candidate.freeze()
    model = build_mtl_model(build_model(candidate), candidate)
    configure_stage_ablation(model, candidate)
    load_strict_state(model, checkpoint["model"])
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    return model.to(device=device, dtype=torch.float32).eval()


def load_strict_state(model, state):
    current = model.state_dict()
    # Only Swin attention masks may be rebuilt, from the same layer geometry.
    # Every parameter and all other buffers must match exactly.
    rebuilt = {name for name in set(state) | set(current) if name.endswith(".attn_mask")}
    if any(name not in current for name in state if name.endswith(".attn_mask")):
        raise ValueError("Unexpected attention-mask key outside constructed Swin geometry")
    supplied = {name: tensor for name, tensor in state.items() if name not in rebuilt}
    required = {name: tensor for name, tensor in current.items() if name not in rebuilt}
    if state_schema(supplied) != state_schema(required):
        raise ValueError("M_B model key/shape/dtype mismatch")
    model.load_state_dict({**supplied, **{name: tensor for name, tensor in current.items() if name in rebuilt}}, strict=True)
