# UniPoRA + CARA 使用流程

本文档说明如何在当前代码库中运行一种两阶段流程：

1. Stage-1 使用 CARA/AG-MTLoRA 搜索任务分组。
2. 搜索阶段禁用 prompt，避免 prompt token 影响 task affinity。
3. 分组完成后，在正式训练阶段重新启用每个 task 独立的 prompt。
4. LoRA 侧采用 UniPoRA 风格：关闭 task-specific LoRA，只保留 shared / group-shared LoRA。

本文中的 UniPoRA 指：将所有任务的 `R_PER_TASK` 设为 0，只保留 `shared` rank。这样 backbone adapter 中没有 task-LoRA，Stage-1 affinity 主要基于 shared TA-LoRA 参数的任务梯度关系来计算。

## 1. 推荐配置

基础配置使用：

```text
configs/mtlora/tiny_448/pascal/ag_promlora_tiny_448_r64_prom50_global_group_proxy.yaml
```

建议先复制一份新配置，避免覆盖原始实验配置，例如：

```powershell
Copy-Item `
  configs\mtlora\tiny_448\pascal\ag_promlora_tiny_448_r64_prom50_global_group_proxy.yaml `
  configs\mtlora\tiny_448\pascal\unipora_cara_tiny_448_r64_prom50_global_group_proxy.yaml
```

然后把新配置中的 `MODEL.MTLORA.R_PER_TASK` 改成：

```yaml
R_PER_TASK:
  semseg: [0]
  normals: [0]
  sal: [0]
  human_parts: [0]
  edge: [0]
  depth: [0]
  shared: [64]
```

保持 prompt 配置开启：

```yaml
PROMPT:
  ENABLED: True
  INITIATION: random
  LOCATION: prepend
  NUM_TOKENS: 50
  DEEP: True
  DROPOUT: 0.0
```

保持 CARA 搜索配置：

```yaml
AGMTLORA:
  ENABLED: True
  STAGE: 1
  GROUPING_SOURCE: search
  SEARCH_SCORE_SOURCE: group_proxy
  PARTITION_GRANULARITY: global
```

关键含义：

- `semseg/normals/sal/human_parts/edge/depth: [0]` 会在配置解析时扩展成每个 Swin stage 都为 0。
- `shared: [64]` 会扩展成每个 Swin stage 的 shared rank 都为 64。
- Stage-1 搜索时还没有固定 group，因此使用的是普通 global shared LoRA。
- 分组完成后，正式训练使用生成的 resolved config，代码会把 global shared LoRA 路由切换成 group-shared LoRA。

## 2. Stage-1 分组搜索

Stage-1 的目标是只用 shared LoRA 来评估 task affinity，并搜索任务分组。为了避免 prompt 影响 affinity，Stage-1 prepare 命令中临时关闭 prompt：

```powershell
python scripts\ag_mtlora_stage1_prepare.py `
  --cfg configs\mtlora\tiny_448\pascal\unipora_cara_tiny_448_r64_prom50_global_group_proxy.yaml `
  --pascal D:\path\to\PASCAL_MT `
  --tasks semseg,normals,sal,human_parts `
  --batch-size 8 `
  --resume-backbone backbone\swin_tiny_patch4_window7_224.pth `
  --opts MODEL.PROMPT.ENABLED False
```

这一步会发生以下事情：

- `MODEL.PROMPT.ENABLED` 被 CLI 临时覆盖为 `False`。
- Stage-1 模型不会构建 prompt，也不会在 forward 中插入 prompt token。
- task-LoRA rank 为 0，因此 backbone 中不会创建 task-LoRA 参数。
- shared LoRA rank 为 64，warmup 和 affinity 阶段会训练 shared LoRA 以及 decoder。
- affinity 只在 backbone 的 shared TA-LoRA 参数上计算。
- `SEARCH_SCORE_SOURCE=group_proxy` 时，Stage-1 会跳过 predictor chain，直接使用 group proxy 做 partition search。

Stage-1 输出目录通常位于：

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
```

其中最重要的是：

- `grouping__group_proxy.json`：搜索得到的 task group 划分。
- `resolved_agmtlora_config__group_proxy.yaml`：正式训练应该使用的配置。
- `post_affinity_checkpoint.pth`：Stage-1 affinity 结束后的 baseline checkpoint，可选地用于初始化正式训练。

## 3. 为什么 Stage-1 关闭 prompt 后，正式训练还会打开 prompt

Stage-1 命令使用的是 CLI override：

```text
--opts MODEL.PROMPT.ENABLED False
```

这个覆盖只作用于 Stage-1 prepare 运行时配置。Stage-1 生成的 `resolved_agmtlora_config__group_proxy.yaml` 会把原始 `--cfg` 文件作为 `BASE`，并额外指定：

```yaml
MODEL:
  AGMTLORA:
    GROUPING_SOURCE: fixed_json
    GROUPING_JSON: <stage1_dir>/grouping__group_proxy.json
    GROUP_SHARED_RANKS: ...
```

因此，只要原始配置文件中 `MODEL.PROMPT.ENABLED: True`，正式训练读取 resolved config 时 prompt 会重新开启。

正式训练进入 `main.py` 后，如果 `MODEL.PROMPT.ENABLED=True`，代码会调用 `mark_prompt_as_trainable()`，让 `prompt_embeddings` 和 `deep_prompt_embeddings` 参与梯度更新。

## 4. 正式训练

正式训练使用 Stage-1 输出的 resolved config：

```powershell
torchrun --nproc_per_node=1 main.py `
  --cfg output\<MODEL.NAME>\<TAG>\ag_mtlora_stage1_prepare\run_<timestamp>\resolved_agmtlora_config__group_proxy.yaml `
  --pascal D:\path\to\PASCAL_MT `
  --tasks semseg,normals,sal,human_parts,edge,depth `
  --batch-size 8 `
  --epochs 300 `
  --ckpt-freq 20 `
  --eval-freq 5 `
  --resume output\<MODEL.NAME>\<TAG>\ag_mtlora_stage1_prepare\run_<timestamp>\post_affinity_checkpoint.pth
```

推荐使用 `--resume post_affinity_checkpoint.pth`，原因是：

- Stage-1 已经训练过 global shared LoRA。
- 正式训练的 resolved config 使用 group-shared LoRA。
- `utils.py` 中的加载逻辑会把 checkpoint 中的单一 shared LoRA 权重复制到每个 group-specific LoRA bank。
- Stage-1 搜索时 prompt 被禁用，因此 checkpoint 中不会有 prompt 参数；正式训练模型里的 prompt 会保持随机初始化，并从正式训练开始更新。

如果不希望继承 Stage-1 的 LoRA 权重，也可以不用 `--resume post_affinity_checkpoint.pth`，改用 backbone 初始化：

```powershell
torchrun --nproc_per_node=1 main.py `
  --cfg output\<MODEL.NAME>\<TAG>\ag_mtlora_stage1_prepare\run_<timestamp>\resolved_agmtlora_config__group_proxy.yaml `
  --pascal D:\path\to\PASCAL_MT `
  --tasks semseg,normals,sal,human_parts,edge,depth `
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
  --tasks semseg,normals,sal,human_parts,edge,depth `
  --batch-size 32 `
  --resume output\<MODEL.NAME>\<TAG>\default\ckpt_epoch_<N>.pth `
  --eval
```

如果评估 edge，代码会按现有评估路径生成 edge cache 并调用 edge evaluator。

## 7. 检查点

Stage-1 prepare 日志中应确认：

- `MODEL.PROMPT.ENABLED: False`
- `MODEL.MTLORA.R_PER_TASK` 中各 task 为 0，`shared` 为 64。
- `MODEL.AGMTLORA.GROUPING_SOURCE: search`
- `MODEL.AGMTLORA.SEARCH_SCORE_SOURCE: group_proxy`
- 输出了 `grouping__group_proxy.json` 和 `resolved_agmtlora_config__group_proxy.yaml`。

正式训练日志中应确认：

- `MODEL.PROMPT.ENABLED: True`
- `MODEL.AGMTLORA.GROUPING_SOURCE: fixed_json`
- `MODEL.AGMTLORA.GROUPING_JSON` 指向 Stage-1 的 grouping 文件。
- `MODEL.MTLORA.AGMTLORA_ENABLED: True`
- `MODEL.MTLORA.AGMTLORA_GROUP_NAMES` 非空。
- 如果使用 `--resume post_affinity_checkpoint.pth`，日志中出现 group LoRA 相关 missing/unexpected keys 时不要立刻视为错误；单一 shared LoRA 到 group-shared LoRA 的扩展逻辑会处理可对齐的 shared 权重。

## 8. 常见问题

### Stage-1 为什么不直接使用 prompt？

prompt token 即使初始化为 0，也会改变 attention 序列长度和 softmax 归一化范围，因此不是严格 no-op。为了让 task affinity 更干净，Stage-1 搜索阶段建议直接禁用 prompt。

### 为什么不用把原始 YAML 里的 `PROMPT.ENABLED` 改成 `False`？

不要这样做。原始 YAML 要保持 `PROMPT.ENABLED: True`，这样 Stage-1 生成的 resolved config 在正式训练时才会自动启用 prompt。Stage-1 关闭 prompt 只通过 CLI override 完成。

### 如果正式训练也加了 `--opts MODEL.PROMPT.ENABLED False` 会怎样？

prompt 会继续关闭，正式训练不会变成 UniPoRA + CARA + prompt。正式训练不要再加这个 override。

### 如果从 Stage-1 checkpoint resume，prompt 会是什么状态？

Stage-1 搜索时 prompt 被禁用，所以 checkpoint 中没有 prompt 参数。正式训练模型会重新构建 prompt，并保持随机初始化；加载 checkpoint 时 prompt keys 缺失是预期现象。

### 如果不从 Stage-1 checkpoint resume，会不会影响分组？

不会。分组由 resolved config 中的 `GROUPING_JSON` 固定。区别只是正式训练不会继承 Stage-1 学到的 shared LoRA 权重。

### `R_PER_TASK=0` 会不会删除 decoder？

不会。它只影响 backbone 里的 task-specific LoRA rank。每个任务的 decoder head 仍然存在并参与训练。

## 9. 推荐命名

建议将这一组实验命名为：

```text
unipora_cara_tiny_448_r64_prom50_global_group_proxy
```

含义：

- `unipora`：task-LoRA rank 为 0，只保留 shared / group-shared LoRA。
- `cara`：使用 CARA/AG-MTLoRA Stage-1 搜索 task grouping。
- `r64`：shared LoRA rank 为 64。
- `prom50`：正式训练使用 50 个 prompt token。
- `global_group_proxy`：全网络共享一个 task grouping，并用 group proxy 做搜索评分。
