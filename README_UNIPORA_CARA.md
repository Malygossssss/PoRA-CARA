# UniPoRA + CARA 使用流程

本文档说明如何在当前代码库中运行一种两阶段流程：

1. Stage-1 使用 CARA/AG-MTLoRA 搜索任务分组。
2. 搜索阶段启用每个 task 独立的 prompt，使 task affinity 与最终模型的 prompted forward 保持一致。
3. 分组完成后，在正式训练阶段继续使用并训练这些 task-specific prompt。
4. LoRA 侧采用 UniPoRA 风格：关闭 task-specific LoRA，只保留 shared / group-shared LoRA。

本文中的 UniPoRA 指：将所有任务的 `R_PER_TASK` 设为 0，只保留 `shared` rank。这样 backbone adapter 中没有 task-LoRA，Stage-1 affinity 基于 prompt-conditioned forward 下 shared TA-LoRA 参数的任务梯度关系来计算。

## 1. 推荐配置

推荐直接使用端到端 2lr 配置：

```text
configs/mtlora/tiny_448/pascal/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr.yaml
```

该文件通过 `BASE` 继承原来的 UniPoRA-CARA 配置，只覆盖实验名和学习率：

```yaml
BASE:
  - unipora_cara_tiny_448_r64_prom50_global_group_proxy.yaml

MODEL:
  NAME: unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr

TRAIN:
  BASE_LR: 1.0e-3
  WARMUP_LR: 1.0e-6
  MIN_LR: 1.0e-5
```

被继承的基础配置已经将 `MODEL.MTLORA.R_PER_TASK` 设置为：

```yaml
R_PER_TASK:
  semseg: [0]
  normals: [0]
  sal: [0]
  human_parts: [0]
  shared: [64]
```

同时保持 prompt 配置开启：

```yaml
PROMPT:
  ENABLED: True
  INITIATION: random
  LOCATION: prepend
  NUM_TOKENS: 50
  DEEP: True
  DROPOUT: 0.0
```

并保持 CARA 搜索配置：

```yaml
AGMTLORA:
  ENABLED: True
  STAGE: 1
  GROUPING_SOURCE: search
  SEARCH_SCORE_SOURCE: group_proxy
  PARTITION_GRANULARITY: global
```

关键含义：

- `semseg/normals/sal/human_parts: [0]` 会在配置解析时扩展成每个 Swin stage 都为 0。
- `shared: [64]` 会扩展成每个 Swin stage 的 shared rank 都为 64。
- YAML 中的 `BASE_LR=1e-3` 是原始实验值。Stage-1 和正式训练都会在运行时对 `BASE_LR/WARMUP_LR/MIN_LR` 只缩放一次，公式为 `batch_size * WORLD_SIZE * accumulation_steps / 512`。
- Stage-1 的 5 个 warmup epoch 使用线性 warmup，后续 50 个 affinity epoch 使用 cosine decay；训练路径与正式训练一样启用 autocast、GradScaler 和 `CLIP_GRAD`。
- Stage-1 搜索时还没有固定 group，因此使用的是普通 global shared LoRA。
- 分组完成后，正式训练使用生成的 resolved config，代码会把 global shared LoRA 路由切换成 group-shared LoRA。

## 2. Stage-1 分组搜索

Stage-1 的目标是在 task-specific prompt 开启的条件下，只对 shared LoRA 参数评估 task affinity，并搜索任务分组。基础配置中的 `MODEL.PROMPT.ENABLED: True` 会直接用于 Stage-1，不再通过 CLI 临时关闭 prompt：

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port 29501 \
  scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr.yaml \
  --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 9 \
  --resume-backbone backbone/Swin/swin_tiny_patch4_window7_224.pth
```

若要让 Stage-1 与正式训练采用相同的双进程启动方式，可运行：

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --master_port 29501 \
  scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr.yaml \
  --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 32 \
  --resume-backbone backbone/Swin/swin_tiny_patch4_window7_224.pth
```

这里严格延续当前 UniPoRA 双卡语义：两个 rank 都读取完整数据、各自构建模型并独立完成 Stage-1 搜索，不使用 DDP、`DistributedSampler`、梯度同步或 affinity 平均。rank 0 使用 `SEED`，rank 1 使用 `SEED + 1`；rank 0 的 grouping、resolved config 和 checkpoint 是后续正式训练的规范结果，rank 1 只作为独立诊断结果。

2lr 是端到端实验配置，必须用它创建新的 Stage-1 run。不要通过 `--resume-stage1-dir` 复用旧学习率生成的 affinity/grouping，也不要把旧 `post_affinity_checkpoint.pth` 用作新的 2lr 正式训练初始化。

尤其不要复用 2026-08-28 修复前生成的 Stage-1 产物：旧实现曾以未缩放的 `BASE_LR` 做固定学习率训练，且没有 AMP、scheduler、梯度裁剪和有限值验收。即使文件存在，也不能据此继续 Stage-2。

这一步会发生以下事情：

- `MODEL.PROMPT.ENABLED` 保持为 `True`，Stage-1 会构建并训练每个 task 独立的 prompt。
- 每个 task 使用自己的 prompted backbone forward，prompt 会影响 shared LoRA 上测得的任务梯度关系。
- task-LoRA rank 为 0，因此 backbone 中不会创建 task-LoRA 参数。
- shared LoRA rank 为 64，warmup 和 affinity 阶段会训练 shared LoRA、task-specific prompt 以及 decoder。
- affinity 仍然只在 backbone 的 shared TA-LoRA 参数上计算；prompt 参数不直接纳入 affinity 参数集合，但会通过 prompted forward 影响 affinity。
- `SEARCH_SCORE_SOURCE=group_proxy` 时，Stage-1 会跳过 predictor chain，直接使用 group proxy 做 partition search。

Prompt-on 会为每个 task 分别执行 prompted backbone forward，显存占用和运行时间通常高于 Prompt-off。如果 `--batch-size 32` 显存不足，应按实际设备减小 batch size。

单进程 Stage-1 的输出目录保持不变，通常位于：

```text
output/<MODEL.NAME>/<TAG>/ag_mtlora_stage1_prepare/run_<timestamp>/
```

重点产物：

```text
affinity.json
affinity.csv
group_proxy.json
group_proxy.csv
grouping__group_proxy.json
partition_search_results__group_proxy.json
resolved_agmtlora_config__group_proxy.yaml
resolved_agmtlora_runtime_snapshot__group_proxy.yaml
warmup_checkpoint.pth
post_affinity_checkpoint.pth
last_good_checkpoint.pth
stage1_artifacts.json
```

成功条件是日志最后出现 `STAGE1_COMPLETED`，同时
`stage1_artifacts.json` 的 `status` 为 `complete`。若检测到 NaN/Inf、异常
gradient/state 或 AMP smoke test 失败，进程会以非零状态退出，日志出现
`STAGE1_ABORTED`，并写入 `failure_report.json`（多进程为
`failure_report_rankN.json`）、`status: failed` 的 manifest 和最近一个健康
epoch 的 `last_good_checkpoint.pth`。失败 run 的 grouping 和
`post_affinity_checkpoint.pth` 不可用于正式训练。

双进程 Stage-1 会在共享 run 根目录下按 rank 隔离产物：

```text
output/<MODEL.NAME>/<TAG>/ag_mtlora_stage1_prepare/run_<timestamp>/
├── rank_0/                       # 规范结果，供正式训练使用
│   ├── resolved_agmtlora_config__group_proxy.yaml
│   ├── post_affinity_checkpoint.pth
│   └── stage1_artifacts.json
├── rank_1/                       # 独立诊断结果
│   └── stage1_artifacts.json
└── stage1_multi_process_manifest.json
```

双进程续跑时，`--resume-stage1-dir` 应传共享的 `run_<timestamp>` 根目录；脚本会为每个进程自动选择对应的 `rank_<n>` 子目录。若相应 rank 目录不存在，脚本会直接报错，避免不同 rank 混用 artifacts。

其中最重要的是：

- `grouping__group_proxy.json`：搜索得到的 task group 划分。
- `resolved_agmtlora_config__group_proxy.yaml`：正式训练应该使用的配置。
- `post_affinity_checkpoint.pth`：Stage-1 affinity 结束后的 baseline checkpoint，可选地用于初始化正式训练。

## 3. Stage-1 与正式训练中的 prompt

Stage-1 和正式训练都继承基础配置中的 `MODEL.PROMPT.ENABLED: True`。Stage-1 生成的 `resolved_agmtlora_config__group_proxy.yaml` 会把原始 `--cfg` 文件作为 `BASE`，并额外指定：

```yaml
MODEL:
  AGMTLORA:
    GROUPING_SOURCE: fixed_json
    GROUPING_JSON: <stage1_dir>/grouping__group_proxy.json
    GROUP_SHARED_RANKS: ...
```

因此，正式训练读取 resolved config 后会继续保持 prompt 开启，并在 prompt-conditioned forward 下将 global shared LoRA 切换为 group-shared LoRA。

Stage-1 和正式训练构建模型后都会调用 `mark_prompt_as_trainable()`，让 `prompt_embeddings` 和 `deep_prompt_embeddings` 参与梯度更新。Stage-1 的 affinity 只针对 shared TA-LoRA 参数计算，但计算使用的特征和梯度已经受到 task-specific prompt 的影响。

## 4. 正式训练

正式训练使用 Stage-1 输出的 resolved config：

```powershell
STAGE1_DIR=output/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr/default/ag_mtlora_stage1_prepare/run_新时间戳

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port 29501 \
  main.py \
  --cfg "$STAGE1_DIR/resolved_agmtlora_config__group_proxy.yaml" \
  --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 10 \
  --epochs 300 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --resume "$STAGE1_DIR/post_affinity_checkpoint.pth"
```

如果需要与 UniPoRA 当前代码严格一致的双卡启动：

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun \
  --nproc_per_node=2 \
  --master_port=29501 \
  main.py \
  --cfg output/<MODEL.NAME>/<TAG>/ag_mtlora_stage1_prepare/run_<timestamp>/rank_0/resolved_agmtlora_config__group_proxy.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --epochs 300 \
  --resume output/<MODEL.NAME>/<TAG>/ag_mtlora_stage1_prepare/run_<timestamp>/rank_0/post_affinity_checkpoint.pth
```

这里的“双卡”严格采用 UniPoRA 语义：两个 rank 各自读取完整训练集并独立更新模型，不使用 DDP，也不平均梯度；checkpoint 仍只保存 rank 0 的模型。每个进程使用 `--batch-size` 指定的 batch，同时学习率仍按 `WORLD_SIZE=2` 缩放。

推荐使用 `--resume post_affinity_checkpoint.pth`，原因是：

- Stage-1 已经训练过 global shared LoRA。
- 正式训练的 resolved config 使用 group-shared LoRA。
- `utils.py` 中的加载逻辑会把 checkpoint 中的单一 shared LoRA 权重复制到每个 group-specific LoRA bank。
- Stage-1 已经在 Prompt-on 条件下训练过 task-specific prompt，因此 checkpoint 中包含 prompt 参数；正式训练会继承这些 prompt 权重并继续更新。

如果不希望继承 Stage-1 的 LoRA 权重，也可以不用 `--resume post_affinity_checkpoint.pth`，改用 backbone 初始化：

```powershell
torchrun --nproc_per_node=1 main.py `
  --cfg output\<MODEL.NAME>\<TAG>\ag_mtlora_stage1_prepare\run_<timestamp>\resolved_agmtlora_config__group_proxy.yaml `
  --pascal D:\path\to\PASCAL_MT `
  --tasks semseg,normals,sal,human_parts `
  --batch-size 8 `
  --epochs 300 `
  --resume-backbone backbone\swin_tiny_patch4_window7_224.pth
```

## 5. 正式训练阶段的模型结构

正式训练时模型结构是：

```text
Swin backbone
  + group-shared LoRA
  + task-specific prompt
  + no task-LoRA
  + task decoders
```

更具体地说：

- task-LoRA 仍然没有，因为 `R_PER_TASK` 中各任务 rank 为 0。
- shared LoRA 会根据 `grouping__group_proxy.json` 路由为 group-shared LoRA。
- prompt 会被构建，并且每个 task 有独立的 prompt 参数。
- `MultiTaskSwin.forward()` 会对每个 task 调用 prompted backbone，因此每个 task 使用自己的 prompt 前向。
- decoder heads 仍然是每个 task 独立的，并且默认参与训练。

## 6. 评估

训练完成后，使用同一个 resolved config 和正式训练 checkpoint 评估：

```powershell
torchrun --nproc_per_node=1 main.py `
  --cfg output\<MODEL.NAME>\<TAG>\ag_mtlora_stage1_prepare\run_<timestamp>\resolved_agmtlora_config__group_proxy.yaml `
  --pascal D:\path\to\PASCAL_MT `
  --tasks semseg,normals,sal,human_parts `
  --batch-size 32 `
  --resume output\<MODEL.NAME>\<TAG>\default\ckpt_epoch_<N>.pth `
  --eval
```

## 7. 检查点

Stage-1 prepare 日志中应确认：

- `MODEL.PROMPT.ENABLED: True`
- `TASKS: [semseg, normals, sal, human_parts]`
- `MODEL.MTLORA.R_PER_TASK` 中各 task 为 0，`shared` 为 64。
- `MODEL.AGMTLORA.GROUPING_SOURCE: search`
- `MODEL.AGMTLORA.SEARCH_SCORE_SOURCE: group_proxy`
- 输出了 `grouping__group_proxy.json` 和 `resolved_agmtlora_config__group_proxy.yaml`。
- `Runtime learning-rate scaling` 中的峰值 LR 符合当前 batch；例如单进程 batch 9 时，`BASE_LR=1e-3` 应缩放为 `1.7578125e-5`。
- 最终标记为 `STAGE1_COMPLETED`，不存在 `STAGE1_ABORTED`。

正式训练日志中应确认：

- `MODEL.PROMPT.ENABLED: True`
- `MODEL.AGMTLORA.GROUPING_SOURCE: fixed_json`
- `MODEL.AGMTLORA.GROUPING_JSON` 指向 Stage-1 的 grouping 文件。
- `MODEL.MTLORA.AGMTLORA_ENABLED: True`
- `MODEL.MTLORA.AGMTLORA_GROUP_NAMES` 非空。
- 如果使用 `--resume post_affinity_checkpoint.pth`，日志中出现 group LoRA 相关 missing/unexpected keys 时不要立刻视为错误；单一 shared LoRA 到 group-shared LoRA 的扩展逻辑会处理可对齐的 shared 权重。

## 8. 常见问题

### Stage-1 为什么启用 prompt？

最终模型使用 task-specific prompt，而 prompt 会改变 attention、任务特征以及 shared LoRA 上的梯度关系。Stage-1 同样启用 prompt，可以让搜索阶段测得的 task affinity 更接近正式训练时的任务交互。

### Prompt 参数会直接参与 affinity 计算吗？

不会。affinity 的求导参数集合仍然只有 backbone 中的 shared TA-LoRA 参数。Prompt 会正常参与前向和训练，并通过改变 task-specific 特征及其 shared-LoRA 梯度间接影响 affinity。

### 如果从 Stage-1 checkpoint resume，prompt 会是什么状态？

Stage-1 使用 Prompt-on，因此 `post_affinity_checkpoint.pth` 中包含已经训练过的 task-specific prompt。正式训练会加载这些 prompt 参数并继续更新；正常情况下不应再出现 prompt keys 缺失。

### 如果不从 Stage-1 checkpoint resume，会不会影响分组？

不会。分组由 resolved config 中的 `GROUPING_JSON` 固定。区别只是正式训练不会继承 Stage-1 学到的 shared LoRA 权重。

### `R_PER_TASK=0` 会不会删除 decoder？

不会。它只影响 backbone 里的 task-specific LoRA rank。每个任务的 decoder head 仍然存在并参与训练。

## 9. 推荐命名

建议将这一组实验命名为：

```text
unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr
```

含义：

- `unipora`：task-LoRA rank 为 0，只保留 shared / group-shared LoRA。
- `cara`：使用 CARA/AG-MTLoRA Stage-1 搜索 task grouping。
- `r64`：shared LoRA rank 为 64。
- `prom50`：Stage-1 和正式训练都使用 50 个 prompt token。
- `global_group_proxy`：全网络共享一个 task grouping，并用 group proxy 做搜索评分。
- `2lr`：Stage-1 的 `BASE_LR` 和正式训练的三个学习率字段均为原默认值的 2 倍。
