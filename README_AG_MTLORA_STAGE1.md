# AG-MTLoRA Stage-1

## 1. 方法概述

AG-MTLoRA Stage-1 是在当前 MTLoRA / UniPoRA 代码基础上实现的一个 grouped MTLoRA 版本。

与原始 MTLoRA 的区别：

- 原始 MTLoRA：所有任务共享一套全局 shared TA-LoRA，每个任务各自拥有自己的 TS-LoRA。
- AG-MTLoRA Stage-1：先做 task grouping，再让同一个 group 内的任务共享一套 group-specific TA-LoRA；TS-LoRA 保持逐任务独立不变。
- 当前版本只实现 Stage-1，也就是整网使用同一套 global grouping，不做 stage-wise 或 layer-wise grouping。
- 当前版本强制 partition 约束：每个 task 只能属于一个 group，不能重叠。

本实现参考 `./refer/ETAP` 的 prediction pipeline，而不是直接照搬其最终 search space。

与 ETAP 对齐的部分：

- directed pairwise affinity
- group-by-task proxy
- sampled training groups
- base predictor: affine calibration + spline/ridge 一维回归
- residual predictor: group multi-hot mask + ridge regression
- affinity-score 的多 epoch 在线累计方式

为了适配 grouped TA-LoRA 做的改动：

- ETAP 原始最终 search 不是严格 partition，允许重叠 group。
- AG-MTLoRA Stage-1 训练时必须给每个 task 一个唯一 group，才能路由到唯一的 shared TA-LoRA bank。
- 因此这里保留 ETAP 的 prediction pipeline，但把最后一步改成 partition-constrained exhaustive search。
- 当前实现仍保留显式 warmup 这段工程化适配，再进入 ETAP 风格的在线 affinity-score 累计阶段。

## 2. 代码改动说明

### 新增文件

- `ag_mtlora/config_utils.py`
  - grouping / partition 枚举、group rank 解析、artifact 路径解析。
- `ag_mtlora/stage1.py`
  - Stage-1 prepare 主流程：warmup、多 epoch directed affinity、group proxy、predictor chain、partition search、artifact 导出。
- `ag_mtlora/stage1_multiprocess.py`
  - 解析 rank 环境、隔离各 rank 输出目录，并生成单 rank 与共享 run manifest。
- `scripts/ag_mtlora_stage1_prepare.py`
  - 两步工作流中的 Step-1 入口脚本。
- `scripts/ag_mtlora_stage1_replay_search.py`
  - 只读取已有 Stage-1 输出目录、重放 partition search 的离线脚本。
- `configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml`
  - PASCAL 四任务默认 Stage-1 配置。
- `configs/mtlora/tiny_448/nyud/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_nyud.yaml`
  - NYUD 四任务默认 Stage-1 配置。

### 关键修改

- `config.py`
  - 新增 `AFFINITY_WARMUP_EPOCHS` 和 `AFFINITY_SCORE_EPOCHS`。
  - 新增 `SEARCH_SCORE_SOURCE`，支持用 `final_predictions` 或 `group_proxy` 做 partition search。
  - 保留 `AFFINITY_COLLECT_EPOCHS` 作为兼容别名；如果没有显式设置 `AFFINITY_WARMUP_EPOCHS`，则回退到该旧字段。
- `models/lora.py`
  - 将单一 shared TA-LoRA 扩展为多个 group-specific TA-LoRA banks。
  - TS-LoRA 逻辑保持不变。
- `models/swin_transformer_mtlora.py`
  - 增加 AG 路由所需的 task stream 传递。
  - forward 中实现 `task -> group -> selected TA-LoRA`。
- `utils.py`
  - 支持把 baseline checkpoint 中的单一 shared TA-LoRA 自动复制到每个 group-specific TA bank。
- `README_AG_MTLORA_STAGE1.md`
  - 当前说明文档。

## 3. ETAP 风格 Stage-1 流程

### Step A. warmup + 多 epoch directed affinity

当前 Step-1 的 baseline 生成分为两段：

1. warmup 阶段

- 用 baseline MTLoRA 正常训练 `MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS`
- 这一步不累计 affinity-score
- 结束后保存 `warmup_checkpoint.pth`

2. affinity-score 阶段

- 在 warmup 结束后的同一个 baseline MTLoRA 上继续训练 `MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS`
- 这一步按 batch 在线累计 directed affinity
- 结束后保存 `post_affinity_checkpoint.pth`

这一步与 ETAP 对齐的地方是：

- affinity 来自 baseline 持续训练轨迹
- 从 post-warmup baseline 的第一个 batch 起在线累计
- 累计窗口覆盖多个 epoch，而不是 warmup 后单独 1 个 epoch

主 affinity 定义为：

`A[i->j] = mean_b( g_j^(b) · u_i^(b) / eta_b )`

其中：

- `g_t` 是 task `t` 在 shared TA-LoRA 参数上的梯度。
- `u_t` 是 task `t` 的 pseudo-update。
- `i->j` 表示 task `i` 的更新对 task `j` 的近似影响。

注意：

- 搜索使用的是 directed affinity。
- 对称矩阵 `0.5 * (A + A^T)` 只做可视化导出，不参与搜索。

### Step B. group-by-task proxy

对每个 candidate group `G`，先为 group 内每个 task 单独构造 proxy，而不是先把整个 group 压成一个分数。

对 `t in G`：

- 若 `|G| = 1`，则 `proxy_G[t] = 0`
- 否则 `proxy_G[t] = avg_{s in G, s != t} A[s->t]`

也就是：用 group 内其他成员指向当前 task 的 directed affinity 入边均值，作为这个 task 在该 group 下的 proxy。

### Step C. predictor chain

默认情况下，搜索之前先训练一个 ETAP 风格的 predictor chain。

如果 `MODEL.AGMTLORA.SEARCH_SCORE_SOURCE=group_proxy`，则 Step C 会被整体跳过，直接把 Step B 的 `group-by-task proxy` 作为 Step D 的输入。

1. sampled training groups

- 默认策略：`all_singletons + all_pairs + random_higher_order`
- 如果 budget 小于单例组数量，代码会自动保留所有 singleton 组，因为它们是 gain=0 的锚点
- 如果总 candidate groups 不大于 budget，则直接全部使用

2. base predictor

- 输入：group-by-task proxy
- 标签：短程 group 训练相对 singleton 训练的 per-task gain
- 默认实现：affine calibration + `SplineTransformer + Ridge`
- 可选实现：`knn`、`rf`

3. residual predictor

- 输入：group multi-hot mask
- 目标：`gt_gain - base_prediction`
- 默认实现：Ridge regression

4. final prediction

- `initial_predictions = base(proxy)`
- `residual_predictions = residual(mask)`
- `final_predictions = initial_predictions + residual_predictions`

默认情况下，后续 partition search 读取 `final_predictions`。

可选地，将 `MODEL.AGMTLORA.SEARCH_SCORE_SOURCE` 设为 `group_proxy` 后，partition search 会直接读取 `group_proxy`，不再使用 `initial_predictions`、`residual_predictions` 和 `final_predictions`。

### Step D. partition-constrained search

对所有 candidate groups 先准备好 group-by-task search score，然后再做 partition-constrained exhaustive search。

可用 search score source：

- `final_predictions`
  - 默认选项，使用 Step C 输出的 `initial_predictions + residual_predictions`
- `group_proxy`
  - 直接使用 Step B 输出的 `group-by-task proxy`

partition `P` 的得分为：

`score(P) = mean_{t in T} search_score[group_of_P(t)][t]`

tie-break 顺序固定为：

- partition score 更高优先
- group 数更少优先
- group 字典序更小优先

## 4. Grouped TA-LoRA 训练侧实现

训练阶段保持以下约束：

- 每个 group 一套 group-specific TA-LoRA。
- 每个 task 仍保留自己的 TS-LoRA。
- forward 路由为 `task -> group -> selected TA-LoRA`。
- grouping 对整网所有 TA-LoRA 层统一生效。
- AG-MTLoRA 关闭时，原始 MTLoRA 行为保持不变。

baseline checkpoint 初始化规则：

- `warmup_checkpoint.pth` 是 warmup 结束后的中间检查点。
- `post_affinity_checkpoint.pth` 是多 epoch affinity-score 累计结束后的 baseline 检查点。
- Step-2 默认推荐从 `post_affinity_checkpoint.pth` 初始化。
- 加载时会自动把单一 shared TA 权重复制到每个 group-specific TA bank。

## 5. Shared rank 配置

`MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET` 只作为默认自动分配参考，不是硬约束。

支持两种方式：

- 默认自动分配：`GROUP_RANK_ALLOCATION: equal_split`
- 手动覆盖：`GROUP_SHARED_RANKS`

默认自动分配规则：

- 先按 `equal_split` 把 `TOTAL_SHARED_RANK_BUDGET` 平均分给各 group
- 余数按前几个 group 分摊
- 默认分配结果会显式写入 Step-1 生成的 `resolved_agmtlora_config.yaml`

手动覆盖规则：

- `GROUP_SHARED_RANKS` 优先级高于自动分配
- 可以写成一维列表，例如 `[32, 32]`
- 也可以写成按 stage 指定的二维列表，例如 `[[32,32,32,32], [48,48,48,48]]`
- 不要求各 group rank 之和等于 `TOTAL_SHARED_RANK_BUDGET`

如果直接用 `grouping.json` 训练且没有显式设置 `GROUP_SHARED_RANKS`，代码会优先读取 `grouping.json` 中保存的 rank 设置。

## 6. 关键配置项

主要配置位于 `MODEL.AGMTLORA`：

- `ENABLED`
- `STAGE`
- `MAX_GROUPS`
- `TOTAL_SHARED_RANK_BUDGET`
- `GROUP_SHARED_RANKS`
- `GROUP_RANK_ALLOCATION`
- `GROUPING_SOURCE`
- `GROUPING_JSON`
- `AFFINITY_WARMUP_EPOCHS`
- `AFFINITY_SCORE_EPOCHS`
- `AFFINITY_COLLECT_EPOCHS`
- `AFFINITY_SAVE_PATH`
- `GROUPING_SAVE_PATH`
- `SEARCH_OBJECTIVE`
- `SEARCH_SCORE_SOURCE`
- `PREDICTOR_TRAIN_GROUP_BUDGET`
- `PREDICTOR_TRAIN_GROUP_STRATEGY`
- `PREDICTOR_GROUP_TRAIN_EPOCHS`
- `BASE_PREDICTOR`
- `BASE_PREDICTOR_KWARGS`
- `RESIDUAL_PREDICTOR`
- `RESIDUAL_ALPHA`
- `VISUALIZE_SYMMETRIC_AFFINITY`

兼容规则：

- 推荐新配置使用 `AFFINITY_WARMUP_EPOCHS` 和 `AFFINITY_SCORE_EPOCHS`
- 旧配置如果只写了 `AFFINITY_COLLECT_EPOCHS`，则它会被解释为 warmup epoch 数

`SEARCH_SCORE_SOURCE` 说明：

- `final_predictions`
  - 默认值，保持原始 predictor-chain + partition search 逻辑
- `group_proxy`
  - 跳过 Step C，直接用 `group_proxy` 做 Step D

## 7. 运行流程

### 环境要求

环境依赖继续沿用仓库根目录 `README.md`。

额外要求：

- 安装 `scikit-learn`
- 准备好 backbone checkpoint，例如 `backbone/swin_tiny_patch4_window7_224.pth`

### Step-1: 生成 affinity / grouping

PASCAL 四任务示例：

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 24 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

#### 双进程独立搜索

Stage-1 现在也支持与正式训练一致的双进程启动：

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --master_port 29501 \
  scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 24 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

也可使用等价的 `torchrun --nproc_per_node=2 --master_port=29501 ...`。脚本同时接受 launcher 注入的 `--local_rank` 和 `--local-rank`。

双进程严格采用当前 UniPoRA 的独立进程语义：

- 每个 rank 读取完整数据并独立执行 warmup、affinity 和分组搜索。
- 不使用 DDP、`DistributedSampler`、梯度同步、affinity 平均或 grouping 合并。
- 每个进程的 `--batch-size` 含义不变；rank `n` 使用 `config.SEED + n`。
- rank 0 产物是后续正式训练的规范输入，其他 rank 产物用于稳定性诊断。

输出会写入同一个 run 根目录下互不覆盖的 rank 子目录：

```text
output/<model_name>/<tag>/ag_mtlora_stage1_prepare/run_<timestamp>/
├── rank_0/
│   ├── resolved_agmtlora_config.yaml
│   ├── post_affinity_checkpoint.pth
│   └── stage1_artifacts.json
├── rank_1/
│   └── stage1_artifacts.json
└── stage1_multi_process_manifest.json
```

正式训练应读取 `rank_0/resolved_agmtlora_config*.yaml`，如需继承 Stage-1 权重则同时读取 `rank_0/post_affinity_checkpoint.pth`。

NYUD 四任务示例：

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/nyud/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_nyud.yaml \
  --nyud /path/to/NYUD_MT \
  --tasks semseg,normals,depth,edge \
  --batch-size 8 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

如果希望在主流程里直接改用 `group_proxy` 做 search，可以通过 `--opts` 覆盖：

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth \
  --opts MODEL.AGMTLORA.SEARCH_SCORE_SOURCE group_proxy
```

双进程续跑时仍用相同的 launcher，并把共享 run 根目录传给 `--resume-stage1-dir`：

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --master_port 29501 \
  scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --resume-stage1-dir output/<model_name>/<tag>/ag_mtlora_stage1_prepare/run_<timestamp> \
  --opts MODEL.AGMTLORA.SEARCH_SCORE_SOURCE group_proxy
```

每个进程会自动映射到自己的 `rank_<n>` 子目录；缺少相应 rank 目录时会报错。单进程续跑仍直接传原来的 Stage-1 输出目录，目录兼容性不变。

如果你已经有一个完整的 Stage-1 输出目录，并且只想复用已有 affinity / group proxy 重做 search，不重新跑 warmup、affinity 和 predictor chain，可以直接使用离线 replay 脚本：

```bash
python scripts/ag_mtlora_stage1_replay_search.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --stage1-dir output/ag_mtlora_stage1_tiny_448_r64_scale4_pertask/default/ag_mtlora_stage1_prepare/run_20260406_065115 \
  --score-source group_proxy
```

如果你更倾向继续走 `scripts/ag_mtlora_stage1_prepare.py`，也可以利用 `--resume-stage1-dir` 复用已有 run 目录中的 affinity artifacts：

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --resume-stage1-dir output/ag_mtlora_stage1_tiny_448_r64_scale4_pertask/default/ag_mtlora_stage1_prepare/run_20260406_065115 \
  --opts MODEL.AGMTLORA.SEARCH_SCORE_SOURCE group_proxy
```

Step-1 输出目录形如：

`output/<model_name>/<tag>/ag_mtlora_stage1_prepare/run_<timestamp>/`

### Step-2: 训练 grouped AG-MTLoRA

推荐直接使用 Step-1 自动生成的 `resolved_agmtlora_config.yaml`，并用 `post_affinity_checkpoint.pth` 做初始化。

PASCAL 四任务示例：

`resolved_agmtlora_config.yaml` is the schema-safe training override generated by Step-1 and can be passed directly to `main.py --cfg`.
`resolved_agmtlora_runtime_snapshot.yaml` is a full Stage-1 runtime dump for inspection/debug only; do not pass it back into `main.py --cfg`.
```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/resolved_agmtlora_config.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --resume /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/post_affinity_checkpoint.pth
```

NYUD 四任务示例：

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/resolved_agmtlora_config.yaml \
  --nyud /path/to/NYUD_MT \
  --tasks semseg,normals,depth,edge \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --resume /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/post_affinity_checkpoint.pth
```

### 直接指定已有 grouping json 训练

推荐方式：

- 复制一份 AG config
- 将 `MODEL.AGMTLORA.GROUPING_SOURCE` 改成 `fixed_json`
- 将 `MODEL.AGMTLORA.GROUPING_JSON` 改成已有 `grouping.json` 路径
- 如需手动改 rank，再设置 `MODEL.AGMTLORA.GROUP_SHARED_RANKS`
- 正式训练仍推荐从 `post_affinity_checkpoint.pth` 初始化

如果你直接使用 Step-1 生成的 `resolved_agmtlora_config.yaml`，这一步已经自动完成。

## 8. 输出文件说明

Step-1 至少会生成以下 artifacts：

- `affinity.json`
  - 多 epoch 聚合后的 directed affinity 主结果
- `affinity.csv`
  - directed affinity 可读矩阵
- `affinity_epoch_history.json`
- `affinity_epoch_history.csv`
  - 每个 affinity epoch 的 directed affinity 历史
- `affinity_symmetric.json`
- `affinity_symmetric.csv`
  - 对称化 affinity，可视化用途
- `group_proxy.json`
- `group_proxy.csv`
  - 每个 candidate group 对每个 task 的 proxy
- `predictor_train_groups.json`
- `predictor_train_groups.csv`
  - predictor 训练用 groups、singleton loss、ground-truth gains
- `initial_predictions.json`
- `initial_predictions.csv`
  - base predictor 输出
- `residual_predictions.json`
- `residual_predictions.csv`
  - residual predictor 输出
- `final_predictions.json`
- `final_predictions.csv`
  - final predicted gains
- `partition_search_results.json`
- `partition_search_results.csv`
  - 所有 partition 的得分、每 task predicted gain 和排序
- `grouping.json`
  - 最终分组结果
- `resolved_agmtlora_config.yaml`
  - 显式写入 `GROUP_SHARED_RANKS` 和 `GROUPING_JSON` 的训练配置
- `warmup_checkpoint.pth`
  - warmup 结束后的中间 checkpoint
- `post_affinity_checkpoint.pth`
  - Step-2 默认推荐初始化 checkpoint

当 `SEARCH_SCORE_SOURCE=group_proxy` 时：

- Step C 被跳过，因此默认不会生成 `predictor_train_groups.json`、`initial_predictions.json`、`residual_predictions.json`、`final_predictions.json` 这类 predictor-chain artifacts。
- search 相关产物会并排写成带后缀的文件，避免覆盖默认 `final_predictions` 版本：
  - `partition_search_results__group_proxy.json`
  - `partition_search_results__group_proxy.csv`
  - `grouping__group_proxy.json`
  - `resolved_agmtlora_config__group_proxy.yaml`
  - `resolved_agmtlora_runtime_snapshot__group_proxy.yaml`
- `partition_search_results*.json` 和 `grouping*.json` 中会额外记录：
  - `search_score_source`
  - `search_score_path`

离线 replay 脚本也遵循同样的 side-by-side 命名规则；默认的 `partition_search_results.json` 和 `grouping.json` 不会被覆盖。

正式训练日志和 checkpoint 继续使用原始训练目录格式：

`output/<model_name>/<tag>/run_<timestamp>/`

## 9. 最小实验示例

以 PASCAL 四任务为例：

1. 先做 Step-1

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

2. 再做 Step-2

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/resolved_agmtlora_config.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --resume /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/post_affinity_checkpoint.pth
```

## 10. 注意事项与已知限制

- 当前只实现 Stage-1。
- 当前 grouping 是 global grouping，不是 stage-wise / layer-wise grouping。
- 当前实现贴合的是 ETAP 的 prediction pipeline 和多 epoch 在线 affinity-score 累计方式，不是 ETAP 原始最终 search space。
- 当前最终 search 强制 partition 约束，这是为了适配 grouped TA-LoRA 的唯一归组需求。
- 默认配置和当前实现主要面向 4-task PASCAL / NYUD 小任务数场景，因此 partition search 采用穷举。
- 当前 AG 路由重点覆盖 QKV、Proj、FC1、FC2 等 TA-LoRA 注入位置；默认配置中 `MODEL.MTLORA.DOWNSAMPLER_ENABLED=False`，这也是当前推荐设置。
- 如果后续扩展到 Stage-2 或 stage-wise grouping，优先修改 `models/swin_transformer_mtlora.py` 中的 task routing 和 `config.py` 中的 group rank / grouping 解析逻辑。

## 11. Stage-Wise Partition（group_proxy only）

当前仓库已经支持一个最小改动版的 stage-wise partition 路径，但有明确边界：

- 只支持 `MODEL.AGMTLORA.PARTITION_GRANULARITY=stage`
- 只支持 `MODEL.AGMTLORA.SEARCH_SCORE_SOURCE=group_proxy`
- 不支持 `stage + final_predictions`
- Stage-2 训练仍然推荐直接使用 Stage-1 自动生成的 resolved config，而不是手写 fixed-json 训练配置

仓库内已经补了两份可直接使用的 Stage-1 样例：

- `configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_stagewise_proxy.yaml`
- `configs/mtlora/tiny_448/nyud/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_nyud_stagewise_proxy.yaml`

这两份配置的核心开关是：

```yaml
MODEL:
  AGMTLORA:
    GROUPING_SOURCE: search
    PARTITION_GRANULARITY: stage
    SEARCH_SCORE_SOURCE: group_proxy
    MAX_GROUPS: 2
    TOTAL_SHARED_RANK_BUDGET: 64
```

语义上它表示：

- Stage-1 先按 stage 拆分 shared TA-LoRA affinity
- 每个 stage 独立构造自己的 `group_proxy`
- 每个 stage 独立运行一次现有的 partition exhaustive search
- group slot 名字固定为 `group_0 .. group_{MAX_GROUPS-1}`
- 某个 slot 如果在某个 stage 没被用到，该 stage 上的 shared rank 自动置 0

Stage-wise 模式下，Step-1 的关键输出会变成：

- `affinity.json`
  - 额外包含 `directed_affinity_by_stage`
- `group_proxy.json`
  - 额外包含 `group_proxy_by_stage`
- `partition_search_results__group_proxy.json`
  - 包含 `ranked_partitions_by_stage`
- `grouping__group_proxy.json`
  - 包含 `partition_granularity`
  - 包含 `group_slot_names`
  - 包含 `groups_by_stage`
  - 包含 `task_to_group_by_stage`
  - 包含 `num_groups_by_stage`
  - 包含按 `[slot][stage]` 排布的 `group_shared_ranks`

运行建议：

1. Step-1 直接使用新的 stage-wise 样例 YAML。
2. Step-2 直接使用输出目录里的 `resolved_agmtlora_config__group_proxy.yaml`。
3. 如果想手动接管训练配置，再把 `GROUPING_SOURCE` 改成 `fixed_json`，并指向生成的 `grouping__group_proxy.json`。

PASCAL 四任务示例：

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_stagewise_proxy.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

随后训练：

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/resolved_agmtlora_config__group_proxy.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --resume /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/post_affinity_checkpoint.pth
```
