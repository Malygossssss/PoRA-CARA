import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models import build_model, build_mtl_model
from models.lora import map_old_state_dict_weights


def serialize_data(value):
    if isinstance(value, dict):
        return {str(k): serialize_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_data(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialize_data(payload), handle, indent=2, ensure_ascii=False)


def set_random_seed(seed, deterministic):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        cudnn.benchmark = True


def build_model_for_experiment(config, device):
    backbone = build_model(config)
    model = build_mtl_model(backbone, config) if config.MTL else backbone
    from stage_ablation import prepare_trainable_parameters
    prepare_trainable_parameters(model, config)
    model.to(device)
    return model


def prepare_state_dict_for_load(config, model, model_state, logger):
    model_state = dict(model_state)

    attn_mask_keys = [key for key in model_state.keys() if "attn_mask" in key]
    for key in attn_mask_keys:
        del model_state[key]

    if config.MODEL.UPDATE_RELATIVE_POSITION:
        drop_keys = [
            key
            for key in model_state.keys()
            if "relative_position_index" in key or "relative_coords_table" in key
        ]
        for key in drop_keys:
            del model_state[key]

    if config.MODEL.MTLORA.ENABLED:
        from utils import _expand_agmtlora_shared_state
        model_state = _expand_agmtlora_shared_state(model_state, model.state_dict(), config)
        mapping = {}
        trainable_layers = []
        mtlora = config.MODEL.MTLORA
        if mtlora.QKV_ENABLED:
            trainable_layers.extend(["attn.qkv.weight", "attn.qkv.bias"])
        if mtlora.PROJ_ENABLED:
            trainable_layers.extend(["attn.proj.weight", "attn.proj.bias"])
        if mtlora.FC1_ENABLED:
            trainable_layers.extend(["mlp.fc1.weight", "mlp.fc1.bias"])
        if mtlora.FC2_ENABLED:
            trainable_layers.extend(["mlp.fc2.weight", "mlp.fc2.bias"])
        if mtlora.DOWNSAMPLER_ENABLED:
            trainable_layers.extend(["downsample.reduction.weight"])

        for key in list(model_state.keys()):
            last_three = ".".join(key.split(".")[-3:])
            prefix = ".".join(key.split(".")[:-3])
            if last_three in trainable_layers:
                weight_bias = last_three.split(".")[-1]
                layer_name = ".".join(last_three.split(".")[:-1])
                mapping[f"{prefix}.{layer_name}.{weight_bias}"] = (
                    f"{prefix}.{layer_name}.linear.{weight_bias}"
                )

        if mapping:
            model_state = map_old_state_dict_weights(
                model_state, mapping, "", config.MODEL.MTLORA.SPLIT_QKV
            )

    incompatible = model.load_state_dict(model_state, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    policy = getattr(config.MODEL, "STAGE_ABLATION", None)
    if policy is not None and policy.ENABLED:
        meaningful_missing = [key for key in missing if "attn_mask" not in key]
        if meaningful_missing or unexpected:
            raise ValueError(f"Evaluation checkpoint model keys mismatch: missing={meaningful_missing}, unexpected={unexpected}")
    if missing:
        logger.warning("Missing keys while loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys while loading checkpoint: %s", unexpected)


def load_model_state(model, checkpoint_path, config, logger):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    from stage_ablation import validate_ablation_checkpoint
    validate_ablation_checkpoint(config, checkpoint)
    checkpoint_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    prepare_state_dict_for_load(config, model, checkpoint_state, logger)
    return checkpoint
