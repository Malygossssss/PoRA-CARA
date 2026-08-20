# MTLoRA

Official PyTorch implementation of MTLoRA for efficient multi-task learning with Swin Transformer backbones.

This cleaned repository keeps the MTLoRA training and evaluation path, standard Swin baselines, dataset utilities, and a few general analysis scripts. Non-core experimental side branches have been removed.

## Environment

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ for training

Install dependencies:

```bash
pip install -r requirements.txt
```

Backbone checkpoints are expected under `backbone/`, for example:

- `backbone/swin_tiny_patch4_window7_224.pth`
- `backbone/swin_base_patch4_window7_224.pth`

## Datasets

Pass dataset roots either in the config or on the command line:

- PASCAL-MT: `--pascal /path/to/PASCAL_MT`
- NYUDv2-MT: `--nyud /path/to/NYUD_MT`

NYUDv2 preparation:

```bash
python scripts/prepare_nyudv2.py --src /path/to/nyud_raw --dst /path/to/NYUD_MT --edge-source real --edge-root /path/to/nyud_edge_gt --edge-format auto --overwrite
```

If you only need segmentation-derived edge labels:

```bash
python scripts/derive_nyud_edge_from_semseg.py --dataset-root /path/to/NYUD_MT --replace-edge --overwrite
```

## Main Entry

`main.py` is the standard training and evaluation entrypoint.

Common modes:

- Train from backbone checkpoint: `--resume-backbone ...`
- Resume full training: `--resume ...`
- Evaluation only: `--resume ... --eval`

### PASCAL-MT training

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq=20 \
  --eval-freq=5 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

### PASCAL-MT evaluation

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 32 \
  --resume /path/to/checkpoint.pth \
  --eval
```

### NYUDv2 four-task training

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg configs/mtlora/tiny_448/nyud/mtlora_tiny_448_r64_scale4_pertask_nyud.yaml \
  --nyud /path/to/NYUD_MT \
  --tasks semseg,normals,depth,edge \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq=20 \
  --eval-freq=5 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

### Process launch semantics

```bash
# Single process
torchrun --nproc_per_node=1 main.py --cfg configs/... --tasks semseg,normals,sal,human_parts ...

# Two independent GPU processes (aligned with UniPoRA)
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29501 \
  main.py --cfg configs/... --tasks semseg,normals,sal,human_parts ...
```

The multi-process command intentionally matches the current UniPoRA training
semantics. It does **not** wrap the model with `DistributedDataParallel` and it
does **not** shard the MTL dataset. Every rank trains an independent model over
the complete dataset; only the rank-0 model is written by the checkpoint path.
`DATA.BATCH_SIZE` remains the batch size used by each independent process, while
the learning-rate scaling formula still includes `WORLD_SIZE`, as in UniPoRA.

Stage-1 grouping search supports the same two-process launcher:

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --nproc_per_node 2 --master_port 29501 \
  scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

Each rank performs a complete independent search with seed `SEED + rank` and
writes to `run_<timestamp>/rank_<rank>/`. Rank 0 is canonical; use its resolved
config and checkpoint for Step-2. No DDP, data sharding, gradient synchronization,
or affinity averaging is performed. Single-process Stage-1 paths remain unchanged.

## Available Configs

Kept MTLoRA configs:

- `configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r16_scale4_pertask.yaml`
- `configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r32_scale4_pertask.yaml`
- `configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask.yaml`
- `configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask_r16.yaml`
- `configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask_r32.yaml`
- `configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask_r64.yaml`
- `configs/mtlora/base_448/pascal/mtlora_base_448_r64_scale4_pertask.yaml`
- `configs/mtlora/tiny_448/nyud/mtlora_tiny_448_r64_scale4_pertask_nyud.yaml`
- `configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml`
- `configs/mtlora/tiny_448/nyud/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_nyud.yaml`
- `configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_stagewise_proxy.yaml`
- `configs/mtlora/tiny_448/nyud/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_nyud_stagewise_proxy.yaml`

Standard Swin baselines are kept under `configs/swin/`.

## Utilities

AG-MTLoRA Stage-1 specific workflow, search configuration, replay-search usage, and the new stage-wise partition mode are documented in `README_AG_MTLORA_STAGE1.md`.

Stage-wise partition example configs:

- PASCAL: `configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_stagewise_proxy.yaml`
- NYUD: `configs/mtlora/tiny_448/nyud/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_nyud_stagewise_proxy.yaml`

These example configs set:

- `MODEL.AGMTLORA.PARTITION_GRANULARITY=stage`
- `MODEL.AGMTLORA.SEARCH_SCORE_SOURCE=group_proxy`

Step-2 training still uses the Stage-1 generated resolved config, for example `resolved_agmtlora_config__group_proxy.yaml`.

Compute Delta-m from logs:

```bash
python compute_delta_m.py --help
```

Inspect FLOPs for a checkpoint:

```bash
python run_mtlora_flops.py \
  --cfg configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --checkpoint /path/to/checkpoint.pth
```

Inspect parameter summary:

```bash
python run_mtlora_parameter_summary.py \
  --cfg configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --checkpoint /path/to/checkpoint.pth
```

Benchmark inference on GPU:

```bash
python run_mtlora_inference_benchmark.py \
  --cfg configs/mtlora/tiny_448/pascal/mtlora_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --checkpoint /path/to/checkpoint.pth \
  --batch-size 8
```

Offline edge evaluation:

```bash
python scripts/eval_edge_predictions.py --dataset nyud --pred-dir /path/to/predictions --gt-root /path/to/NYUD_MT
```

## Citation

```bibtex
@inproceedings{agiza2024mtlora,
  title={MTLoRA: Low-Rank Adaptation Approach for Efficient Multi-Task Learning},
  author={Agiza, Ahmed and Neseem, Marina and Reda, Sherief},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16196--16205},
  year={2024}
}
```

## License

MIT License. See `LICENSE`.
