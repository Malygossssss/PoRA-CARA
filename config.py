# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Swin Transformer (https://github.com/microsoft/Swin-Transformer)
#
# Original file:
# Copyright (c) 2021 Microsoft
# Licensed under the MIT License
# Written by Ze Liu
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------


import os
import yaml
import re
from yacs.config import CfgNode as CN
from data.mtl_ds import get_tasks_config
import json

from ag_mtlora.config_utils import (
    load_grouping_json,
    normalize_partition_granularity,
    resolve_artifact_path,
    resolve_group_shared_ranks,
    resolve_stagewise_group_shared_ranks,
    stage_index_to_key,
)

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'nyud'
# Input image size
_C.DATA.IMG_SIZE = [224, 224]
# _C.DATA.IMG_SIZE = (480, 640)
# _C.DATA.IMG_SIZE = (448, 448)
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1


# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.DECODER_DIM = 256
_C.MODEL.SWIN.DECODER_PATCH_RES = [7, 7, 14, 28]

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWINV2.DECODER_PATCH_RES = [7, 7, 14, 28]
_C.MODEL.SWINV2.DECODER_DIM = 128

# Swin Transformer MoE parameters
_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = None
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 1
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.COSINE_ROUTER = False
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = False
_C.MODEL.SWIN_MOE.USE_BPR = True
_C.MODEL.SWIN_MOE.IS_GSHARD_LOSS = False
_C.MODEL.SWIN_MOE.GATE_NOISE = 1.0
_C.MODEL.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# [SimMIM] Norm target during training
_C.MODEL.SIMMIM = CN()
_C.MODEL.SIMMIM.NORM_TARGET = CN()
_C.MODEL.SIMMIM.NORM_TARGET.ENABLE = False
_C.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = 47


# Multi task deocders
_C.MODEL.DECODER_HEAD = CN()
_C.MODEL.DECODER_HEAD['semseg'] = 'hrnet'
_C.MODEL.DECODER_HEAD['normals'] = 'hrnet'
_C.MODEL.DECODER_HEAD['sal'] = 'hrnet'
_C.MODEL.DECODER_HEAD['human_parts'] = 'hrnet'
_C.MODEL.DECODER_HEAD['edge'] = 'hrnet'
_C.MODEL.DECODER_HEAD['depth'] = 'hrnet'
_C.MODEL.DECODER_CHANNELS = [18, 36, 72, 144]

_C.MODEL.SEGFORMER_CHANNELS = 256

# Minimal visual prompt tuning support for CARA + AG-MTLoRA.
_C.MODEL.PROMPT = CN()
_C.MODEL.PROMPT.ENABLED = False
_C.MODEL.PROMPT.NUM_TOKENS = 0
_C.MODEL.PROMPT.DEEP = True
_C.MODEL.PROMPT.LOCATION = 'prepend'
_C.MODEL.PROMPT.DROPOUT = 0.0
_C.MODEL.PROMPT.INITIATION = 'random'
_C.MODEL.PROMPT.DYNAMIC_PROMPT = False
_C.MODEL.PROMPT.SHARE_TASK_PROMPT = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
# _C.TRAIN.BASE_LR = 5e-5
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []
_C.TRAIN.SKIP_DECODER_CKPT = False

# MTLoRA Related
_C.TRAIN.FREEZE_PATCH_EMBED = False
_C.TRAIN.FREEZE_LAYER_NORM = False
_C.TRAIN.FREEZE_RELATIVE_POSITION_BIAS = False
_C.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION = False

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to run validation
_C.EVAL_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
_C.DETERMINISTIC = False
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False

# Debug / reproducibility diagnostics
_C.DEBUG_REPRO_STEPS = 0
_C.FUSED_LAYERNORM = False
_C.SKIP_INITIAL_EVAL = False
_C.MODEL.DECODER_DOWNSAMPLER = True
_C.MODEL.PER_TASK_DOWNSAMPLER = True
_C.MODEL.UPDATE_RELATIVE_POSITION = False

_C.MODEL.MTLORA = CN()
_C.MODEL.MTLORA.ENABLED = False
_C.MODEL.MTLORA.BIAS = 'none'  # none, all, lora_only
_C.MODEL.MTLORA.R = [8, 8, 8, 8]
_C.MODEL.MTLORA.SHARED_SCALE = [2.0, 2.0, 2.0, 2.0]
_C.MODEL.MTLORA.TASK_SCALE = [2.0, 2.0, 2.0, 2.0]
_C.MODEL.MTLORA.DROPOUT = [0.05, 0.05, 0.05, 0.05]
_C.MODEL.MTLORA.TRAINABLE_SCALE_SHARED = False
_C.MODEL.MTLORA.TRAINABLE_SCALE_PER_TASK = False
_C.MODEL.MTLORA.INTERMEDIATE_SPECIALIZATION = False
_C.MODEL.MTLORA.FREEZE_PRETRAINED = True
_C.MODEL.MTLORA.SPLIT_QKV = False
_C.MODEL.MTLORA.R_PER_TASK = CN(new_allowed=True)
_C.MODEL.MTLORA.SCALE_PER_TASK = CN(new_allowed=True)
_C.MODEL.MTLORA.SHARED_MODE = 'matrix'  # 'matrix', 'addition', lora_only
_C.MODEL.MTLORA.QKV_ENABLED = True
_C.MODEL.MTLORA.PROJ_ENABLED = True
_C.MODEL.MTLORA.FC1_ENABLED = True
_C.MODEL.MTLORA.FC2_ENABLED = True
_C.MODEL.MTLORA.DOWNSAMPLER_ENABLED = False
_C.MODEL.MTLORA.AGMTLORA_ENABLED = False
_C.MODEL.MTLORA.AGMTLORA_STAGE = 0
_C.MODEL.MTLORA.AGMTLORA_PARTITION_GRANULARITY = 'global'
_C.MODEL.MTLORA.AGMTLORA_GROUPS = []
_C.MODEL.MTLORA.AGMTLORA_GROUP_NAMES = []
_C.MODEL.MTLORA.AGMTLORA_GROUP_RANKS = []
_C.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP = CN(new_allowed=True)
_C.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)

_C.MODEL.AGMTLORA = CN()
_C.MODEL.AGMTLORA.ENABLED = False
_C.MODEL.AGMTLORA.STAGE = 1
_C.MODEL.AGMTLORA.PARTITION_GRANULARITY = 'global'
_C.MODEL.AGMTLORA.MAX_GROUPS = 2
_C.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET = 64
_C.MODEL.AGMTLORA.GROUP_SHARED_RANKS = []
_C.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION = 'equal_split'
_C.MODEL.AGMTLORA.GROUPING_SOURCE = 'search'  # search, fixed_json
_C.MODEL.AGMTLORA.GROUPING_JSON = ''
_C.MODEL.AGMTLORA.AFFINITY_COLLECT_EPOCHS = 5
_C.MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS = -1
_C.MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS = 50
_C.MODEL.AGMTLORA.AFFINITY_SAVE_PATH = ''
_C.MODEL.AGMTLORA.GROUPING_SAVE_PATH = ''
_C.MODEL.AGMTLORA.DATA_SPLIT_MODE = 'train_meta_strict'  # train_meta_strict, official_val
_C.MODEL.AGMTLORA.META_VAL_RATIO = 0.2
_C.MODEL.AGMTLORA.META_SPLIT_SEED = -1
_C.MODEL.AGMTLORA.META_SPLIT_SAVE_PATH = ''
_C.MODEL.AGMTLORA.RESOLVED_META_SPLIT_SEED = 0
_C.MODEL.AGMTLORA.SEARCH_OBJECTIVE = 'mean_final_predicted_gain'
_C.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE = 'final_predictions'
_C.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_BUDGET = 0
_C.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_STRATEGY = 'all_singletons+all_pairs+random_higher_order'
_C.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS = 5
_C.MODEL.AGMTLORA.BASE_PREDICTOR = 'spline_ridge'
_C.MODEL.AGMTLORA.BASE_PREDICTOR_KWARGS = CN(new_allowed=True)
_C.MODEL.AGMTLORA.RESIDUAL_PREDICTOR = 'ridge'
_C.MODEL.AGMTLORA.RESIDUAL_ALPHA = 1.0
_C.MODEL.AGMTLORA.VISUALIZE_SYMMETRIC_AFFINITY = True
_C.MODEL.AGMTLORA.RESOLVED_GROUPS = []
_C.MODEL.AGMTLORA.RESOLVED_GROUPS_BY_STAGE = []
_C.MODEL.AGMTLORA.RESOLVED_GROUP_NAMES = []
_C.MODEL.AGMTLORA.RESOLVED_GROUP_SLOT_NAMES = []
_C.MODEL.AGMTLORA.RESOLVED_GROUP_SHARED_RANKS = []
_C.MODEL.AGMTLORA.RESOLVED_GROUP_RANK_SOURCE = ''
_C.MODEL.AGMTLORA.RESOLVED_NUM_GROUPS = 0
_C.MODEL.AGMTLORA.RESOLVED_NUM_GROUPS_BY_STAGE = []
_C.MODEL.AGMTLORA.RESOLVED_PARTITION_GRANULARITY = 'global'
_C.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP = CN(new_allowed=True)
_C.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    if 'DATA' in yaml_cfg and 'IMG_SIZE' in yaml_cfg['DATA']:
        img_size = yaml_cfg['DATA']['IMG_SIZE']
        if isinstance(img_size, int):
            yaml_cfg['DATA']['IMG_SIZE'] = [img_size, img_size]

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_other_cfg(CN(yaml_cfg))
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('ckpt_freq'):
        config.SAVE_FREQ = args.ckpt_freq
    if _check_args('eval_freq'):
        config.EVAL_FREQ = args.eval_freq
    else:
        config.EVAL_FREQ = 1

    if _check_args('skip_initial_validation'):
        config.SKIP_INITIAL_EVAL = True

    if _check_args('eval_training_freq'):
        config.EVAL_TRAINING = args.eval_training_freq
    else:
        config.EVAL_TRAINING = None

    if _check_args('epochs'):
        config.TRAIN.EPOCHS = args.epochs
    if _check_args('mti'):
        config.MODEL.MTI = args.mti
    if _check_args('decoder_map'):
        with open(args.decoder_map, 'r') as f:
            task_dec_map = json.load(f)
            for task, head in task_dec_map.items():
                config.MODEL.DECODER_HEAD[task] = head
    if _check_args('skip_decoder'):
        config.TRAIN.SKIP_DECODER_CKPT = args.skip_decoder
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    db_name = "NYUD"
    if _check_args('nyud'):
        config.DATA.NYUD = args.nyud
        config.DATA.DATA_PATH = args.nyud
        db_name = "NYUD"
    elif _check_args('pascal'):
        config.DATA.PASCAL = args.pascal
        config.DATA.DATA_PATH = args.pascal
        db_name = "PASCALContext"
    config.DATA.DBNAME = db_name

    if _check_args('tasks'):
        config.TASKS = re.compile(r'\s*,\s*').split(args.tasks)
        assert 'shared' not in config.TASKS, 'shared is a reserved task name'
        config.MTL = True
        tsk_config, _ = get_tasks_config(
            db_name, config.TASKS, config.DATA.IMG_SIZE)
        tsk_config = dict(tsk_config)
        config.TASKS_CONFIG = CN(tsk_config)
        config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT = CN(
            dict(config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT))
        config.TASKS_CONFIG.ALL_TASKS.FLAGVALS = CN(
            dict(config.TASKS_CONFIG.ALL_TASKS.FLAGVALS))
        config.TASKS_CONFIG.ALL_TASKS.INFER_FLAGVALS = CN(
            dict(config.TASKS_CONFIG.ALL_TASKS.INFER_FLAGVALS))
        config.MODEL.NUM_CLASSES = 0
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('resume_backbone'):
        config.MODEL.RESUME_BACKBONE = args.resume_backbone
    else:
        config.MODEL.RESUME_BACKBONE = False

    if _check_args('freeze_backbone'):
        config.MODEL.FREEZE_BACKBONE = args.freeze_backbone
    else:
        config.MODEL.FREEZE_BACKBONE = False

    if _check_args('save_sample'):
        config.MODEL.SAVE_SAMPLE = args.save_sample
    else:
        config.MODEL.SAVE_SAMPLE = False

    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    if _check_args('seed'):
        config.SEED = args.seed
    if _check_args('deterministic'):
        config.DETERMINISTIC = True

    if _check_args('debug_repro_steps'):
        config.DEBUG_REPRO_STEPS = int(args.debug_repro_steps)

       # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    # Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    if _check_args('name'):
        config.MODEL.NAME = args.name
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    if config.MODEL.PROMPT.ENABLED:
        if not config.MODEL.MTLORA.ENABLED:
            raise ValueError("MODEL.PROMPT.ENABLED=True requires MODEL.MTLORA.ENABLED=True.")
        if str(config.MODEL.PROMPT.LOCATION) != 'prepend':
            raise ValueError("MODEL.PROMPT.LOCATION must be 'prepend' for CARA prompt support.")
        if not bool(config.MODEL.PROMPT.DEEP):
            raise ValueError("MODEL.PROMPT.DEEP must be True for CARA prompt support.")
        if int(config.MODEL.PROMPT.NUM_TOKENS) <= 0:
            raise ValueError("MODEL.PROMPT.NUM_TOKENS must be > 0 when MODEL.PROMPT.ENABLED=True.")
        if str(config.MODEL.PROMPT.INITIATION) != 'random':
            raise ValueError("MODEL.PROMPT.INITIATION must be 'random' for CARA prompt support.")
        if bool(config.MODEL.PROMPT.DYNAMIC_PROMPT):
            raise ValueError("MODEL.PROMPT.DYNAMIC_PROMPT is not supported in CARA's minimal prompt path.")
        if bool(config.MODEL.PROMPT.SHARE_TASK_PROMPT):
            raise ValueError("MODEL.PROMPT.SHARE_TASK_PROMPT is not supported in CARA's minimal prompt path.")

    # Normalize MTLoRA config
    if config.MODEL.MTLORA.ENABLED:
        if not isinstance(config.MODEL.MTLORA.R, list):
            config.MODEL.MTLORA.R = [
                config.MODEL.MTLORA.R] * len(config.MODEL.SWIN.DEPTHS)
        elif len(config.MODEL.MTLORA.R) == 1:
            config.MODEL.MTLORA.R = config.MODEL.MTLORA.R * \
                len(config.MODEL.SWIN.DEPTHS)
        else:
            assert len(config.MODEL.MTLORA.R) == len(
                config.MODEL.SWIN.DEPTHS), "MTLoRA ranks length should be the same as the number of layers"
        if not isinstance(config.MODEL.MTLORA.SHARED_SCALE, list):
            config.MODEL.MTLORA.SHARED_SCALE = [
                config.MODEL.MTLORA.SHARED_SCALE] * len(config.MODEL.SWIN.DEPTHS)
        elif len(config.MODEL.MTLORA.SHARED_SCALE) == 1:
            config.MODEL.MTLORA.SHARED_SCALE = config.MODEL.MTLORA.SHARED_SCALE * \
                len(config.MODEL.SWIN.DEPTHS)
        else:
            assert len(config.MODEL.MTLORA.SHARED_SCALE) == len(
                config.MODEL.SWIN.DEPTHS), "MTLoRA shared scale length should be the same as the number of layers"
        if not isinstance(config.MODEL.MTLORA.TASK_SCALE, list):
            config.MODEL.MTLORA.TASK_SCALE = [
                config.MODEL.MTLORA.TASK_SCALE] * len(config.MODEL.SWIN.DEPTHS)
        elif len(config.MODEL.MTLORA.TASK_SCALE) == 1:
            config.MODEL.MTLORA.TASK_SCALE = config.MODEL.MTLORA.TASK_SCALE * \
                len(config.MODEL.SWIN.DEPTHS)
        else:
            assert len(config.MODEL.MTLORA.TASK_SCALE) == len(
                config.MODEL.SWIN.DEPTHS), "MTLoRA task scale length should be the same as the number of layers"
        if not isinstance(config.MODEL.MTLORA.DROPOUT, list):
            config.MODEL.MTLORA.DROPOUT = [
                config.MODEL.MTLORA.DROPOUT] * len(config.MODEL.SWIN.DEPTHS)
        elif len(config.MODEL.MTLORA.DROPOUT) == 1:
            config.MODEL.MTLORA.DROPOUT = config.MODEL.MTLORA.DROPOUT * \
                len(config.MODEL.SWIN.DEPTHS)
        else:
            assert len(config.MODEL.MTLORA.DROPOUT) == len(
                config.MODEL.SWIN.DEPTHS), "MTLoRA dropout length should be the same as the number of layers"

        if len(config.MODEL.MTLORA.R_PER_TASK) == 0:
            for task in config.TASKS:
                config.MODEL.MTLORA.R_PER_TASK[task] = config.MODEL.MTLORA.R[:]
            config.MODEL.MTLORA.R_PER_TASK['shared'] = config.MODEL.MTLORA.R[:]
        else:
            for task in config.TASKS + ['shared']:
                if not isinstance(config.MODEL.MTLORA.R_PER_TASK[task], list):
                    config.MODEL.MTLORA.R_PER_TASK[task] = [
                        config.MODEL.MTLORA.R_PER_TASK[task]] * len(config.MODEL.SWIN.DEPTHS)
                elif len(config.MODEL.MTLORA.R_PER_TASK[task]) == 1:
                    config.MODEL.MTLORA.R_PER_TASK[task] = config.MODEL.MTLORA.R_PER_TASK[task] * \
                        len(config.MODEL.SWIN.DEPTHS)
                else:
                    assert len(config.MODEL.MTLORA.R_PER_TASK[task]) == len(
                        config.MODEL.SWIN.DEPTHS), "MTLoRA ranks length should be the same as the number of layers"

        if len(config.MODEL.MTLORA.SCALE_PER_TASK) == 0:
            for task in config.TASKS:
                config.MODEL.MTLORA.SCALE_PER_TASK[task] = config.MODEL.MTLORA.SHARED_SCALE[:]
        else:
            for task in config.TASKS:
                if not isinstance(config.MODEL.MTLORA.SCALE_PER_TASK[task], list):
                    config.MODEL.MTLORA.SCALE_PER_TASK[task] = [
                        config.MODEL.MTLORA.SCALE_PER_TASK[task]] * len(config.MODEL.SWIN.DEPTHS)
                elif len(config.MODEL.MTLORA.SCALE_PER_TASK[task]) == 1:
                    config.MODEL.MTLORA.SCALE_PER_TASK[task] = config.MODEL.MTLORA.SCALE_PER_TASK[task] * \
                        len(config.MODEL.SWIN.DEPTHS)
                else:
                    assert len(config.MODEL.MTLORA.SCALE_PER_TASK[task]) == len(
                        config.MODEL.SWIN.DEPTHS), "MTLoRA task scale length should be the same as the number of layers"
        config.MODEL.MTLORA.R_PER_TASK_LIST = []
        config.MODEL.MTLORA.SCALE_PER_TASK_LIST = []
        for i in range(len(config.MODEL.SWIN.DEPTHS)):
            layer_task_r = {
                'shared': config.MODEL.MTLORA.R[i] if 'shared' not in config.MODEL.MTLORA.R_PER_TASK else config.MODEL.MTLORA.R_PER_TASK['shared'][i]
            }
            layer_task_scale = {
            }
            for task in config.TASKS:
                layer_task_r[task] = config.MODEL.MTLORA.R_PER_TASK[task][i]
                layer_task_scale[task] = config.MODEL.MTLORA.SCALE_PER_TASK[task][i]
            config.MODEL.MTLORA.R_PER_TASK_LIST.append(layer_task_r)
            config.MODEL.MTLORA.SCALE_PER_TASK_LIST.append(layer_task_scale)

    config.MODEL.MTLORA.AGMTLORA_ENABLED = False
    config.MODEL.MTLORA.AGMTLORA_STAGE = 0
    config.MODEL.MTLORA.AGMTLORA_PARTITION_GRANULARITY = "global"
    config.MODEL.MTLORA.AGMTLORA_GROUPS = []
    config.MODEL.MTLORA.AGMTLORA_GROUP_NAMES = []
    config.MODEL.MTLORA.AGMTLORA_GROUP_RANKS = []
    config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP = CN(new_allowed=True)
    config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)

    if config.MODEL.AGMTLORA.ENABLED:
        if not config.MODEL.MTLORA.ENABLED:
            raise ValueError("AGMTLoRA requires MODEL.MTLORA.ENABLED=True.")
        if int(config.MODEL.AGMTLORA.STAGE) != 1:
            raise ValueError("Only AG-MTLoRA Stage-1 is currently supported.")
        if str(config.MODEL.AGMTLORA.DATA_SPLIT_MODE) not in {"train_meta_strict", "official_val"}:
            raise ValueError("MODEL.AGMTLORA.DATA_SPLIT_MODE must be one of {'train_meta_strict', 'official_val'}.")
        if not 0.0 < float(config.MODEL.AGMTLORA.META_VAL_RATIO) < 1.0:
            raise ValueError("MODEL.AGMTLORA.META_VAL_RATIO must satisfy 0 < META_VAL_RATIO < 1.")
        if int(config.MODEL.AGMTLORA.META_SPLIT_SEED) < 0:
            config.MODEL.AGMTLORA.RESOLVED_META_SPLIT_SEED = int(config.SEED)
        else:
            config.MODEL.AGMTLORA.RESOLVED_META_SPLIT_SEED = int(config.MODEL.AGMTLORA.META_SPLIT_SEED)
        if int(config.MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS) < 0:
            config.MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS = int(config.MODEL.AGMTLORA.AFFINITY_COLLECT_EPOCHS)
        if int(config.MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS) < 0:
            raise ValueError("MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS must be >= 0.")
        if str(config.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE) not in {"final_predictions", "group_proxy"}:
            raise ValueError("MODEL.AGMTLORA.SEARCH_SCORE_SOURCE must be one of {'final_predictions', 'group_proxy'}.")
        config.MODEL.AGMTLORA.PARTITION_GRANULARITY = normalize_partition_granularity(
            getattr(config.MODEL.AGMTLORA, "PARTITION_GRANULARITY", "global")
        )

        base_output_dir = config.OUTPUT
        config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH = resolve_artifact_path(
            base_output_dir,
            config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH,
            "ag_mtlora_stage1/affinity.json",
        )
        config.MODEL.AGMTLORA.GROUPING_SAVE_PATH = resolve_artifact_path(
            base_output_dir,
            config.MODEL.AGMTLORA.GROUPING_SAVE_PATH,
            "ag_mtlora_stage1/grouping.json",
        )
        config.MODEL.AGMTLORA.META_SPLIT_SAVE_PATH = resolve_artifact_path(
            base_output_dir,
            config.MODEL.AGMTLORA.META_SPLIT_SAVE_PATH,
            "ag_mtlora_stage1/meta_split.json",
        )
        if config.MODEL.AGMTLORA.GROUPING_JSON:
            config.MODEL.AGMTLORA.GROUPING_JSON = resolve_artifact_path(
                base_output_dir,
                config.MODEL.AGMTLORA.GROUPING_JSON,
                config.MODEL.AGMTLORA.GROUPING_JSON,
            )

        grouping_payload = None
        config.MODEL.AGMTLORA.RESOLVED_GROUPS = []
        config.MODEL.AGMTLORA.RESOLVED_GROUPS_BY_STAGE = []
        config.MODEL.AGMTLORA.RESOLVED_GROUP_NAMES = []
        config.MODEL.AGMTLORA.RESOLVED_GROUP_SLOT_NAMES = []
        config.MODEL.AGMTLORA.RESOLVED_GROUP_SHARED_RANKS = []
        config.MODEL.AGMTLORA.RESOLVED_GROUP_RANK_SOURCE = ""
        config.MODEL.AGMTLORA.RESOLVED_NUM_GROUPS = 0
        config.MODEL.AGMTLORA.RESOLVED_NUM_GROUPS_BY_STAGE = []
        config.MODEL.AGMTLORA.RESOLVED_PARTITION_GRANULARITY = config.MODEL.AGMTLORA.PARTITION_GRANULARITY
        config.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP = CN(new_allowed=True)
        config.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)

        if (
            str(config.MODEL.AGMTLORA.GROUPING_SOURCE) == "search"
            and str(config.MODEL.AGMTLORA.PARTITION_GRANULARITY) == "stage"
            and str(config.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE) != "group_proxy"
        ):
            raise ValueError(
                "Stage-wise partition search currently requires MODEL.AGMTLORA.SEARCH_SCORE_SOURCE='group_proxy'."
            )

        if str(config.MODEL.AGMTLORA.GROUPING_SOURCE) == 'fixed_json':
            if not config.MODEL.AGMTLORA.GROUPING_JSON:
                raise ValueError("MODEL.AGMTLORA.GROUPING_JSON is required when GROUPING_SOURCE=fixed_json.")
            grouping_payload = load_grouping_json(
                config.MODEL.AGMTLORA.GROUPING_JSON,
                config.TASKS,
            )
            resolved_partition_granularity = grouping_payload.get(
                "partition_granularity",
                config.MODEL.AGMTLORA.PARTITION_GRANULARITY,
            )
            config.MODEL.AGMTLORA.PARTITION_GRANULARITY = resolved_partition_granularity
            manual_group_ranks = config.MODEL.AGMTLORA.GROUP_SHARED_RANKS
            rank_source = ""
            if (not manual_group_ranks) and grouping_payload.get("group_shared_ranks"):
                manual_group_ranks = grouping_payload["group_shared_ranks"]
                rank_source = "grouping_json"

            if resolved_partition_granularity == "stage":
                resolved_groups = []
                resolved_groups_by_stage = grouping_payload["groups_by_stage"]
                resolved_group_names = list(grouping_payload["group_slot_names"])
                resolved_group_ranks, rank_source = resolve_stagewise_group_shared_ranks(
                    manual_group_ranks,
                    config.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET,
                    len(resolved_group_names),
                    grouping_payload["num_groups_by_stage"],
                    len(config.MODEL.SWIN.DEPTHS),
                    config.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION,
                )
                resolved_num_groups_by_stage = list(grouping_payload["num_groups_by_stage"])
                resolved_task_to_group = {}
                resolved_task_to_group_by_stage = grouping_payload["task_to_group_by_stage"]
            else:
                resolved_groups = grouping_payload["groups"]
                resolved_groups_by_stage = []
                resolved_task_to_group = grouping_payload["task_to_group"]
                resolved_task_to_group_by_stage = []
                resolved_group_names = [f"group_{idx}" for idx in range(len(resolved_groups))]
                resolved_num_groups_by_stage = []
                resolved_group_ranks, rank_source = resolve_group_shared_ranks(
                    manual_group_ranks,
                    config.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET,
                    len(resolved_groups),
                    len(config.MODEL.SWIN.DEPTHS),
                    config.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION,
                )

            if manual_group_ranks and rank_source != "grouping_json":
                rank_source = "manual"

            config.MODEL.AGMTLORA.RESOLVED_GROUPS = resolved_groups
            config.MODEL.AGMTLORA.RESOLVED_GROUPS_BY_STAGE = resolved_groups_by_stage
            config.MODEL.AGMTLORA.RESOLVED_GROUP_NAMES = resolved_group_names
            config.MODEL.AGMTLORA.RESOLVED_GROUP_SLOT_NAMES = resolved_group_names
            config.MODEL.AGMTLORA.RESOLVED_GROUP_SHARED_RANKS = resolved_group_ranks
            config.MODEL.AGMTLORA.RESOLVED_GROUP_RANK_SOURCE = rank_source
            config.MODEL.AGMTLORA.RESOLVED_NUM_GROUPS = (
                len(resolved_group_names) if resolved_partition_granularity == "stage" else len(resolved_groups)
            )
            config.MODEL.AGMTLORA.RESOLVED_NUM_GROUPS_BY_STAGE = resolved_num_groups_by_stage
            config.MODEL.AGMTLORA.RESOLVED_PARTITION_GRANULARITY = resolved_partition_granularity
            config.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP = CN(new_allowed=True)
            for task, group_name in resolved_task_to_group.items():
                config.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP[task] = group_name
            config.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)
            for stage_idx, task_to_group in enumerate(resolved_task_to_group_by_stage):
                stage_key = stage_index_to_key(stage_idx)
                config.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP_BY_STAGE[stage_key] = CN(new_allowed=True)
                for task, group_name in task_to_group.items():
                    config.MODEL.AGMTLORA.RESOLVED_TASK_TO_GROUP_BY_STAGE[stage_key][task] = group_name

            config.MODEL.MTLORA.AGMTLORA_ENABLED = True
            config.MODEL.MTLORA.AGMTLORA_STAGE = int(config.MODEL.AGMTLORA.STAGE)
            config.MODEL.MTLORA.AGMTLORA_PARTITION_GRANULARITY = resolved_partition_granularity
            config.MODEL.MTLORA.AGMTLORA_GROUPS = resolved_groups
            config.MODEL.MTLORA.AGMTLORA_GROUP_NAMES = resolved_group_names
            config.MODEL.MTLORA.AGMTLORA_GROUP_RANKS = resolved_group_ranks
            config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP = CN(new_allowed=True)
            for task, group_name in resolved_task_to_group.items():
                config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP[task] = group_name
            config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)
            for stage_idx, task_to_group in enumerate(resolved_task_to_group_by_stage):
                stage_key = stage_index_to_key(stage_idx)
                config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE[stage_key] = CN(new_allowed=True)
                for task, group_name in task_to_group.items():
                    config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE[stage_key][task] = group_name
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
