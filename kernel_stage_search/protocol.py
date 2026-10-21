"""Content-addressed experiment identities and preparation checkpoint contracts."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ALGORITHM_VERSION = "task-kernel-stage-v1"
TASKS = ("semseg", "normals", "sal", "human_parts")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path):
    path = Path(path).expanduser()
    return str((PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def write_json(path, value):
    """Replace atomically; incomplete writes never become valid cache entries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(value) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def make_manifest(kind, semantic, paths=None, provenance=None):
    identity = {"schema": SCHEMA_VERSION, "kind": kind, "semantic": plain(copy.deepcopy(semantic))}
    return {**identity, "protocol_id": content_hash(identity), "paths": dict(paths or {}),
            "provenance": dict(provenance or {})}


def read_manifest(path, kind=None):
    manifest = read_json(path)
    identity = {key: manifest[key] for key in ("schema", "kind", "semantic")}
    if manifest["schema"] != SCHEMA_VERSION or content_hash(identity) != manifest.get("protocol_id"):
        raise ValueError("Protocol schema/hash mismatch")
    if kind and manifest["kind"] != kind:
        raise ValueError(f"Expected {kind} protocol")
    return manifest


def validate_scoring_protocol(manifest):
    identity = {key: manifest[key] for key in ("schema", "kind", "semantic")}
    if (manifest.get("schema") != SCHEMA_VERSION or manifest.get("kind") != "scoring"
            or content_hash(identity) != manifest.get("protocol_id")):
        raise ValueError("A verified complete scoring protocol is required")
    semantic = manifest["semantic"]
    if not semantic.get("synthetic"):
        required = {"base_protocol_id", "mb_hash", "c0_hash", "calibration_hash", "sample_plan_hash",
                    "calibration_plan_hash", "prepared_manifest_id", "numeric_runtime", "identity", "settings",
                    "implementation_hash", "split_id", "data_hash"}
        if not required <= semantic.keys():
            raise ValueError("Scoring protocol is not fully bound to M_B/calibration/sample plans")
    budget = semantic.get("settings", {}).get("SEARCH_BUDGET")
    if not isinstance(budget, int) or not 3 <= budget <= 81:
        raise ValueError("Invalid scoring protocol budget")


def implementation_hash():
    # Includes working-copy content: an uncommitted fix also invalidates old caches.
    files = list((PROJECT_ROOT / "kernel_stage_search").glob("*.py"))
    for directory in ("models", "data"):
        files.extend((PROJECT_ROOT / directory).glob("*.py"))
    files.extend(PROJECT_ROOT / name for name in ("config.py", "stage_ablation.py", "utils.py", "main.py",
                 "optimizer.py", "lr_scheduler.py", "mtl_loss_schemes.py", "ag_mtlora/config_utils.py"))
    return content_hash({str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_hash(path)
                         for path in sorted(files)})


def code_provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "implementation_hash": implementation_hash()}


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def validate_config(config):
    node = config.MODEL.KERNEL_SEARCH
    if node.PHASE not in ("off", "prepare", "score") or bool(node.ENABLED) != (node.PHASE != "off"):
        raise ValueError("KERNEL_SEARCH ENABLED/PHASE mismatch (off/prepare/score)")
    if not node.ENABLED:
        return
    from stage_ablation import resolve_stage_modes
    modes = resolve_stage_modes(config)
    if modes is None or len(modes) != 4 or list(config.TASKS) != list(TASKS):
        raise ValueError("Kernel search requires four stages and fixed task order")
    mtlora = config.MODEL.MTLORA
    if (plain(mtlora.AGMTLORA_GROUP_RANKS) != [[16]*4, [48]*4]
            or dict(mtlora.AGMTLORA_TASK_TO_GROUP) != {
                "semseg": "group_0", "human_parts": "group_0", "normals": "group_1", "sal": "group_1"}):
        raise ValueError("Kernel search fixed grouping/rank mismatch")
    if config.MODEL.PROMPT.NUM_TOKENS != 50 or list(config.DATA.IMG_SIZE) != [448, 448]:
        raise ValueError("Kernel search requires 448 inputs and 50 prompts")
    if not node.SOURCE_META_SPLIT or not config.MODEL.INIT_CHECKPOINT or not config.MODEL.INIT_CHECKPOINT_SHA256:
        raise ValueError("Missing source split or common initialization/hash")
    if not 3 <= node.SEARCH_BUDGET <= 81 or list(node.INITIAL_MODES) != ["BBBB", "PPPP", "GGGG"]:
        raise ValueError("Invalid search budget/initial modes")
    for key in ("PREPARE_EPOCHS", "CALIBRATION_IMAGES", "INFO_BATCHES", "INFO_BATCH_IMAGES",
                "MAX_BATCH_IMAGES", "SAMPLES_PER_TASK", "FORWARD_BATCH_SIZE", "BANDWIDTH_SAMPLES"):
        if int(node[key]) != node[key] or node[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if node.MAX_BATCH_IMAGES < node.INFO_BATCH_IMAGES or node.PREPARE_EPOCHS > config.TRAIN.EPOCHS:
        raise ValueError("Invalid image/epoch budget")
    if node.SAMPLES_PER_TASK < 2 or node.BANDWIDTH_SAMPLES < 2:
        raise ValueError("Kernels require at least two calibration/scoring rows")
    if (node.GP_NOISE_VARIANCE != 1e-6 or node.UCB_BETA != 2.0 or node.STD_FLOOR != 1e-6
            or node.ENTROPY_MIN != 1e-8 or node.LABEL_MODE not in ("true_labels", "shuffled")):
        raise ValueError("Unsupported kernel/GP protocol constants")
    if config.MODEL.RESUME_BACKBONE or config.TRAIN.AUTO_RESUME or config.TRAIN.SKIP_DECODER_CKPT:
        raise ValueError("Kernel search requires full C0 initialization and explicit same-run resume")
    if config.TRAIN.EPOCHS != 300 or config.TRAIN.WARMUP_EPOCHS != 20:
        raise ValueError("Preparation must preserve the 300-epoch scheduler horizon / 20 warmup")
    if node.PHASE == "prepare" and (modes != list("BBBB") or config.MODEL.STAGE_ABLATION.WORLD_SIZE != 1):
        raise ValueError("Preparation requires BBBB and WORLD_SIZE=1")


def resolved_identity(config):
    from stage_ablation import stage_ablation_metadata
    stage = stage_ablation_metadata(config)
    # Paths are stored separately. Identity contains actual, resolved ranks and structure.
    model = plain(config.MODEL)
    for key in ("KERNEL_SEARCH", "NAME", "RESUME", "RESUME_BACKBONE", "PRETRAINED", "INIT_CHECKPOINT",
                "INIT_CHECKPOINT_SHA256", "AGMTLORA", "STAGE_ABLATION"):
        model.pop(key, None)
    return {"tasks": list(config.TASKS), "model": model, "grouping": plain(stage["grouping"]),
            "prompt_tokens": stage["prompt_tokens"], "training": plain(stage["protocol"]),
            "image_size": list(config.DATA.IMG_SIZE), "tasks_config": plain(config.TASKS_CONFIG)}


def search_settings(config):
    values = plain(config.MODEL.KERNEL_SEARCH)
    for key in ("ENABLED", "PHASE", "SOURCE_META_SPLIT", "SCORE_CHECKPOINT", "PROTOCOL",
                "RUNTIME_PROTOCOL_ID", "RUNTIME_SPLIT_ID", "LABEL_MODE"):
        values.pop(key, None)
    return values


def create_base_protocol(config, split, data_hash):
    from stage_ablation import verify_initialization_file
    c0 = resolve_path(config.MODEL.INIT_CHECKPOINT)
    verify_initialization_file(config, c0)
    lineage, lineage_paths = {}, {}
    candidates = {"resolved_grouping": Path(config.MODEL.AGMTLORA.GROUPING_JSON)}
    for name in ("stage1_artifacts.json", "meta_split.json", "grouping.json"):
        candidates[name] = Path(c0).parent/name
    for name, path in candidates.items():
        if path.is_file():
            lineage[name] = {"sha256": file_hash(path), "metadata": read_json(path)}
            lineage_paths[name] = str(path.resolve())
        else:
            lineage[name] = {"status": "not_available"}
    return make_manifest("base", {
        "algorithm": ALGORITHM_VERSION, "implementation_hash": implementation_hash(),
        "identity": resolved_identity(config), "settings": search_settings(config),
        "c0_hash": file_hash(c0), "source_split_hash": file_hash(config.MODEL.KERNEL_SEARCH.SOURCE_META_SPLIT),
        "split": split, "split_id": content_hash(split), "data_hash": data_hash,
        "initialization_lineage": lineage,
        "earlier_pretraining_data_history": "unknown; available Stage-1 metadata alone does not establish independence",
        "transforms": "official-train/eval-FixedResize-AddIgnoreRegions-ToTensor-ImageNetNorm",
        "numeric": {"model": "float32", "distances": "float32", "entropy_gp": "float64",
                    "amp": False, "tf32": False, "cudnn_benchmark": False, "deterministic": True}},
        {"c0": c0, "source_split": resolve_path(config.MODEL.KERNEL_SEARCH.SOURCE_META_SPLIT), "lineage": lineage_paths},
        code_provenance())


def bind_preparation(config, base):
    config.defrost()
    config.MODEL.KERNEL_SEARCH.RUNTIME_PROTOCOL_ID = base["protocol_id"]
    config.MODEL.KERNEL_SEARCH.RUNTIME_SPLIT_ID = base["semantic"]["split_id"]
    config.freeze()


def preparation_metadata(config, epoch=None):
    node = getattr(config.MODEL, "KERNEL_SEARCH", None)
    if node is None or not node.ENABLED or node.PHASE != "prepare":
        return None
    if not node.RUNTIME_PROTOCOL_ID or not node.RUNTIME_SPLIT_ID:
        raise ValueError("Preparation protocol must be bound before checkpointing")
    identity = {"phase": "prepare", "base_protocol_id": node.RUNTIME_PROTOCOL_ID,
                "split_id": node.RUNTIME_SPLIT_ID, "c0_hash": config.MODEL.INIT_CHECKPOINT_SHA256,
                "prepare_epochs": node.PREPARE_EPOCHS, "scheduler_horizon": config.TRAIN.EPOCHS,
                "modes": "BBBB", "identity": resolved_identity(config)}
    return {"schema": SCHEMA_VERSION, "identity": identity, "completed_epoch": epoch,
            "prepared_complete": epoch == min(node.PREPARE_EPOCHS, config.TRAIN.EPOCHS)-1}


def validate_preparation_checkpoint(config, checkpoint, initialization=False):
    actual = checkpoint.get("kernel_search_preparation")
    expected = preparation_metadata(config)
    if initialization:
        if actual is not None:
            raise ValueError("Preparation/M_B cannot be used as common C0 initialization")
        return
    if expected is None and actual is None:
        return
    if (expected is None or actual is None or actual.get("schema") != SCHEMA_VERSION
            or actual.get("identity") != expected["identity"]):
        raise ValueError("Preparation checkpoint protocol mismatch; cannot resume as formal training")
    epoch = checkpoint.get("epoch")
    if epoch != actual.get("completed_epoch") or not isinstance(epoch, int) or not 0 <= epoch < config.MODEL.KERNEL_SEARCH.PREPARE_EPOCHS:
        raise ValueError("Invalid preparation epoch")
    if actual.get("prepared_complete") != (epoch == config.MODEL.KERNEL_SEARCH.PREPARE_EPOCHS-1):
        raise ValueError("Invalid preparation endpoint marker")
    if any(key not in checkpoint for key in ("optimizer", "lr_scheduler", "scaler")):
        raise ValueError("Preparation resume requires complete training state")


def validate_training_entry(config, world_size=None):
    validate_config(config)
    node = config.MODEL.KERNEL_SEARCH
    if not node.ENABLED:
        return False
    if node.PHASE == "score":
        raise ValueError("PHASE=score requires scripts/search_stage_kernels.py; main.py cannot train it")
    if config.EVAL_MODE or config.THROUGHPUT_MODE or config.MODEL.RESUME_BACKBONE:
        raise ValueError("Preparation rejects eval/throughput/backbone-only/teacher entries")
    if world_size is not None and world_size != 1:
        raise ValueError("Preparation requires WORLD_SIZE=1")
    return True
