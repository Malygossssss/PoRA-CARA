# 任务相关核信息 Stage 搜索：项目代码修改计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在固定 CARA 分组、group rank 和 Prompt 数量的条件下，以同一份冻结 BBBB 模型的任务相关核信息评分为唯一目标，搜索四个 stage 的 P/G/B 配置，再从共同初始化开展正式训练。

**Architecture:** 复用现有 stage 开关、任务路由、checkpoint 初始化和正式训练入口；新增独立的 `kernel_stage_search` 包，负责数据协议、冻结前向、核评分、类别 GP、缓存及配置导出。`main.py` 只增加一次 BBBB 评分模型准备所需的训练分支；候选评分由独立脚本执行，不进入训练循环。

**Tech Stack:** 项目现有 Python、PyTorch、NumPy、YACS、unittest；GP 用小规模 float64 Cholesky 求解，不新增 BoTorch、GPyTorch 或 COMBO 依赖。真实训练继续使用 CUDA/NCCL 单进程环境。

---

日期：2026-10-20。状态：**待实施的代码修改计划；本次只创建分支及本文件，没有修改实现代码或启动训练。**

- 工作目录：`D:/05_Experiments/PoRA-CARA`。
- 新分支：`feat/kernel-information-stage-search`。
- 起点：`feat/swin-stage-ablation`，commit `2358122264dc93587820bd1117c20240e45181c5`。
- 研究依据：[用户提供的最终研究方案](C:/Users/96372/Documents/Codex/2026-10-13/ban/outputs/2026-10-20-kernel-information-guided-stage-search.md)。
- 研究方案 SHA256：`e29c65a521c731409172b43ebe43a985c0a918762441f41c293084d1dc234b7f`。
- 文首英文句为 writing-plans 的交接模板，不表示本轮已进入实现，也不要求安装不存在的 `superpowers` 命名空间；后续可使用当前可用的 executing-plans 技能。
- 本文将附件作为研究设计输入；附件中的训练、搜索、论文写作内容不构成本轮立即执行这些工作的授权。
- 下文新增模块、配置键、命令、测试均为**待实现合同**。现有文件的位置按上述 commit 核查，行号只是定位参考。
- 2026-10-11 的旧实现计划及旧研究文档保留为历史。本计划替代其中的小预测器、性能干预评分、候选 60/120 epoch 短训、混合 GP 核及重新切分开发集的设计。

## 1. 本版边界与完成标准

1. 搜索空间为四位 `P/G/B`，共 81 个配置；所有任务共享该四位配置，任务身份及 group 路由保留。
2. 初始评分顺序固定为 `BBBB → PPPP → GGGG`，总预算为 12 个不同配置，随后 9 个由 GP 提议。
3. 评分只运行同一个 `M_B` 的冻结前向与核计算；不建立候选 optimizer、不执行 backward、不更新参数或运行统计。
4. 目标只有四任务等权的归一化核信息均值 `J`，不引入损失、mIoU、法线误差、参数惩罚或时间惩罚。
5. GP 可以计算全部 81 点的采集值；模型只为实际提议的配置前向。最终从已评分配置中按 `J` 选胜者。
6. 标签打乱对照复用同一 `M_B`、采样坐标、核参数、初始候选和预算；单独运行 GP，允许后续提议不同。
7. 正式训练使用锁定结构、共同初始化 `C0` 和配对种子 `[0, 1, 2]`；不得从 `M_B` 续训胜者。
8. 搜索可正常选中 `BBBB`。不能为了产生新结构而强制选择非基线配置。
9. 验收区分代码正确性和研究有效性。测试通过不代表信息分数能预测最终性能，也不保证 12 次搜索找到 81 点的全局最优。

## 2. 已核查的项目基础与实际缺口

| 现有位置 | 已有能力或问题 | 本次处理 |
|---|---|---|
| `stage_ablation.py:19`，`resolve_stage_modes` | 校验四级 B/P/G、fixed_json、零 task rank、关闭 backbone downsampler LoRA | 直接复用，不添加 N 模式 |
| `stage_ablation.py:43`，`configure_stage_ablation` | 设置完整 stage 模式、Prompt 开关、共享/group LoRA 开关 | 用于临时候选切换；新增恢复上下文，不改训练冻结逻辑 |
| `models/swin_transformer_vpt.py:164`，`forward_features` | 按 task 返回四级图像 token；已移除 Prompt token | 直接采集，不另造近似特征 |
| `models/swin_transformer_vpt.py:260`，`PromptedBasicLayer.forward` | G 真正删除 token，后续 B/P 可重注入 deep prompt | 增加组合与恢复测试，原则上不改 backbone |
| `models/swin_mtl.py:203`，`MultiTaskSwin.forward` | 每个任务独立执行 backbone，再进入 decoder | 用 backbone hook 捕获本次完整模型前向的任务特征 |
| `models/lora.py:347`、`models/lora.py:467` | active_task 路由以及普通/split-QKV LoRA 开关 | 保留，验证动态切换没有遗漏子模块 |
| `data/mtl_ds.py:708`、`:834`、`:909` | 稳定图像 ID、eval transforms、指定 split 的 dataset 构造 | 显式在官方 train 上读取旧 meta ID |
| `data/custom_transforms.py:242` | 无人体部件标注整图置 255；零法线置 255 | 按变换后的有效标签采样 |
| `ag_mtlora/stage1.py:858` | 可读取旧 meta 划分，但会过滤不存在的 ID；模块还包含旧搜索训练逻辑 | 参考读取方式；新增严格加载器，不直接运行旧 pipeline |
| `ag_mtlora/stage1.py:982` | `maybe_load_initial_weights` 未处理新的 `MODEL.INIT_CHECKPOINT` 路径 | 不用它准备 `M_B`，沿用 `main.py` 的正确初始化流程 |
| `utils.py:152`，`load_checkpoint` | 支持 C0 权重初始化、shared-to-group rank 对齐、同实验恢复身份校验 | C0 继续复用；M_B 用单独严格权重读取接口 |
| `main.py:162`、`:297`、`:374`、`:402`，`data/__init__.py:6`、`data/build.py:202` | MTL 经 data 包分派到官方 train/val，包含初始、周期与最终验证 | 在 main 的数据分派前接入 prepare loader，并统一关闭这些验证入口 |
| `configs/stage_ablation/swin_t/resolved_baseline.yaml` | 已固定 448、50 tokens、16/48 rank、300 epoch 和初始化 hash | 作为新配置的唯一基线来源 |
| `compute_delta_m.py:294` | 默认从日志挑最高 Delta m 的 epoch | 正式报告增加独立固定终点汇总，不直接用此默认挑选逻辑 |

### 2.1 固定身份

- 任务顺序：`semseg, normals, sal, human_parts`。
- group_0：`semseg/human_parts`，四个 stage 的 rank 均为 16。
- group_1：`normals/sal`，四个 stage 的 rank 均为 48。
- JSON 原始 rank 为 32/32，但 resolved YAML 的手动 16/48 覆盖优先；身份记录必须保存**实际生效 rank**。
- 所有 task-specific rank 为 0，backbone downsampler LoRA 关闭，decoder downsampler 保持原设置。
- `C0` 候选沿用现有 `post_affinity_checkpoint.pth`，SHA256 为 `b05959165782d239377e0ab144ff52f6426c3f0327c4851235ba75ff67e20281`。
- 源目录为 `output/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr/default/ag_mtlora_stage1_prepare/run_20260907_031808/`。
- 已读取该目录的 `meta_split.json`、`stage1_artifacts.json`、分组文件和日志：旧 meta-train/meta-val 为 3998/1000；前序 CARA 已使用 meta-val 作为开发数据。
- 已存在的 post-affinity 文件不自动等价于“以固定 16/48 group 结构训练完成的 BBBB 评分模型”。预设流程仍需从 C0 准备一次 M_B，除非提供可核验的同协议 M_B。
- 保留完整初始化来源链；本次读取到的 Stage-1 元数据不能独自证明所有更早预训练的数据历史。未知来源应记录为未知，不能宣称全流程独立。

### 2.2 特征尺寸必须按当前代码定义

当前 `return_stages=True` 返回各层执行完 PatchMerging 后的输出，S4 无 further merging。448 输入时：

| 输出 | 空间尺寸 | 通道数 |
|---|---:|---:|
| S1 | 56 × 56 | 192 |
| S2 | 28 × 28 | 384 |
| S3 | 14 × 14 | 768 |
| S4 | 14 × 14 | 768 |

四级拼接为 2112 维。实现应读取 `layer.output_resolution` 和 tensor shape，断言 token 数匹配；不能硬编码通用 Swin 的 112/56/28/14 或 96/192/384/768。

## 3. 方案选择与模块边界

| 可选做法 | 判断 |
|---|---|
| 新建独立评分包，复用既有模型及正式训练，仅为 M_B 准备接入 `main.py` | **采用**；候选评分和训练控制流清楚，现有九组消融行为容易回归 |
| 将核评分嵌入 `ag_mtlora/stage1.py` | 不采用；容易触发 affinity/grouping/proxy 的旧流程，且该模块不完整支持 C0 初始化入口 |
| 每个候选重新构造模型或借用 `main.py --eval` | 不采用；反复加载容易引入初始化差异，现有 eval 又面向任务性能指标 |

预期只修改 3 个现有实现文件：`config.py`、`main.py`、`utils.py`。backbone、LoRA、decoder、旧 Stage-1 pipeline、已有九组 YAML 均优先复用。若组合测试发现现有缺陷，另列最小修复及回归证据，不能无依据重构。

```text
kernel_stage_search/
  __init__.py
  protocol.py       # 身份、配置、来源链、hash、准备期元数据、产物 schema
  data.py           # 严格 split、标签有效性、固定组批/坐标/排列
  checkpoint.py    # M_B 严格加载、参数及 buffer 状态验证
  policy.py        # 全四级临时模式、异常恢复
  features.py      # task-aware hook、真实尺度、坐标插值、采样特征
  kernels.py       # Gaussian/equality 核、S2/I2、等权 J、退化诊断
  calibration.py  # meta-train 的固定均值/标准差/核宽度
  surrogate.py    # 81 点、类别 GP、UCB、去重与最终胜者
  runner.py        # 前向评分、原子缓存、恢复、true/shuffled 两条记录
  export.py        # 从 C0 的正式配置导出
  reporting.py    # 同一正式终点的四指标、配对结果与成本
scripts/
  search_stage_kernels.py
  check_kernel_stage_search.py
  summarize_kernel_stage_search.py
configs/kernel_stage_search/
  swin_t_pascal.yaml
  prepare_bbbb.yaml
tests/
  test_kernel_stage_protocol.py
  test_kernel_stage_data.py
  test_kernel_stage_prepare.py
  test_kernel_stage_policy.py
  test_kernel_stage_features.py
  test_kernel_stage_kernels.py
  test_kernel_stage_surrogate.py
  test_kernel_stage_runner.py
  test_kernel_stage_export.py
README_KERNEL_STAGE_SEARCH.md
```

不新增 probes、performance-ablation、候选训练队列、120 epoch refinement、随机路径训练、强化学习门控或参数代价目标。

## 4. 预先固定的实验协议

以下为本项目首版建议值，用来补足附件未指定的工程细节，**均未经过性能调参或验证**。必须在搜索前写入协议；修改后产生新协议及新缓存命名空间。

| 项目 | 首版设置 |
|---|---|
| BBBB 准备 | 从 C0，仅在旧 meta-train 上训练 60 epoch，取固定终点；不按开发或官方 val 指标选择 M_B |
| 准备期 scheduler | 沿用 300 epoch horizon、20 warmup，新增独立停止位置 60；完成 epoch 索引 0…59 |
| 准备期 batch/LR | 沿用 batch=10、WORLD_SIZE=1、基础 LR=1e-3，经原函数缩放一次为 1.953125e-5 |
| 校准图像 | 从旧 meta-train 按固定 seed 选 256 张；使用 eval transforms，不能使用随机训练增强 |
| 信息评估图像池 | 原有 1000 张 meta-val，保持 split 不变；固定排列并从中构造评分批次 |
| 信息批次 | 16 个；每批初始 8 张图像；每任务固定 n=512 个有效位置 |
| 组批不足 | 按固定 ID 顺序扩充至最多 32 张；仍不足或标签熵退化则该协议准备失败，不在候选间换规则 |
| 评分微批次 | 4 张，独立于训练 batch；组批统计跨固定微批次累积 |
| 像素采样 | 每任务在该批所有有效位置中均匀无放回抽取 512 个，保存 `(image_id,y,x)`；不额外做类均衡 |
| 采样及校准 seed | 0，使用独立 RNG；不要消耗训练 RNG 或 Python 随机 hash |
| 标准化 | 每任务 × 每 stage × 每通道，训练侧 BBBB 统计的均值和总体标准差；std 下限 1e-6 |
| 特征核宽度 | 每任务在训练侧标准化后的拼接特征中固定抽最多 1024 行，取非对角正平方距离的中位数作为 `sigma_F²` |
| 法线标签核宽度 | 校准侧有效单位法线，用相同中位数规则确定 `sigma_Y²` |
| 评分数值 | 模型 float32、AMP 关闭、eval、inference_mode；关闭 cudnn benchmark 和 TF32，记录确定性算法设置；距离计算显式 float32，熵与 GP 累积/求解 float64 |
| 退化阈值 | `S2(B) > 1e-8`；无正距离的校准宽度直接失败，不用候选数据兜底 |
| GP | `exp(-d_H)`，幅度和长度尺度固定为 1；标准化评分；观测噪声方差 1e-6 |
| GP 稳定性 | jitter 从 1e-10 逐级至 1e-4，记录实际值；仍不稳定则失败 |
| UCB | `mu + 2*std`；总计 12 个实际观测 |
| 标签打乱 | seed=1；每任务、每信息批次保存一个跨图像采样行的固定排列；法线三维向量整体移动 |
| 正式训练 | 原官方 train、300 epoch、种子 0/1/2；原 loss 权重及冻结策略；最终使用 epoch 索引 299 的同一 checkpoint |

批次可含空间相关样本，512 个像素不等于 512 个独立观测。保留每批次分数、图像数与任务标签分布；不得把跨像素波动或 GP 方差写成正式性能的置信区间。

组批、坐标、有效掩码、标签熵检查全部在读取候选结果前完成。preflight 可先检查分类标签熵及法线有效性/非恒定性；法线的精确核标签熵须等训练侧 sigma 校准后检查，若不满足阈值则协议准备失败，不启动搜索。同一批四任务可使用不同坐标，但每个候选、每种标签分支必须复用同一份坐标。16 批等权、四任务等权，不按有效像素总量额外加权。

### 4.1 标签处理

- `semseg`：去掉 255，校验类别为 0…20，保存整数标签。
- `human_parts`：去掉 255，校验类别为 0…6；全 255 图像对该任务不贡献位置。不能把无标注图像当成有效背景。
- `sal`：当前 `_load_sal_distilled` 已用 `>0.5` 二值化；变换后仅接受 0/1，忽略 255。不改成连续标签核。
- `normals`：在归一化前排除含 255、NaN/Inf 或范数 ≤1e-6 的向量，再单位化；禁止对三个通道分别打乱。记录当前 resize/ignore 流程的语义，不顺带改变全项目 transforms。
- 核宽度对候选固定。标签打乱只改变配对关系，保持标签多重集合、有效坐标、标签核谱和 `S2(B)` 不变。

### 4.2 多尺度对齐

在变换后标签的 448×448 坐标系定义 `(y,x)`。将各级 `[B,L,C]` 重排为 `[B,C,h,w]`，在同一目标像素中心做 bilinear 采样：

```python
gx = 2.0 * (x + 0.5) / label_width - 1.0
gy = 2.0 * (y + 0.5) / label_height - 1.0
# grid_sample(..., mode="bilinear", padding_mode="border", align_corners=False)
# 数值上验证其等价于先 interpolate 到标签尺寸再 gather。
```

采样后使用固定均值和 std 标准化四级特征，再沿通道拼接。不使用整图池化、不加随机投影/PCA、不把单 stage 特征或 P/G 特征相加近似完整配置。

## 5. 核评分与 GP 的实现合同

### 5.1 核信息

对称 Gaussian/equality 核的对角线均为 1，因此 `A=KF/n`、`B=KY/n`。二阶熵通过平方和计算，禁止为了求 S2 做 eigendecomposition。

以下为核心实现参考；实际代码需补上 shape、有限值、对角线及对称性检查：

```python
def entropy2(matrix):
    matrix = matrix.to(torch.float64)
    purity = matrix.square().sum()
    if not torch.isfinite(purity) or purity <= 0:
        raise ValueError("Invalid kernel purity")
    return -torch.log2(purity)

def normalized_information(kf, ky, entropy_min=1e-8):
    n = kf.shape[0]
    a, b = kf.to(torch.float64) / n, ky.to(torch.float64) / n
    sy = entropy2(b)
    if sy <= entropy_min:
        raise ValueError("Degenerate label batch")
    joint = a * b
    joint = joint / torch.trace(joint)
    sx, sxy = entropy2(a), entropy2(joint)
    i2 = sx + sy - sxy
    return {"sx": sx, "sy": sy, "sxy": sxy, "i2": i2, "ratio": i2 / sy}
```

`J = mean_batch(mean_task(ratio))`。额外记录 `delta_J=J-J_BBBB`，不将其作为另一个目标。不将 `I2` 或 ratio 强行裁剪到 [0,1]；异常值先记录并诊断数值和核退化。

距离矩阵使用平方范数加矩阵乘法，微小负距离夹至 0，核强制对称且对角置 1。保存非对角均值/方差、接近 0/1 的比例、特征方差、Sy 等诊断；这些诊断不进入 J。校准阶段失败应停下；某个候选核接近常数或单位阵时保留原始分数及标记，不能偷偷改变其 sigma。

可单独用相同样本记录 sigma 乘数 0.5/1/2 的敏感性。该诊断必须独立编号，不自动重选、换指标或根据官方 val 挑带宽。

### 5.2 类别 GP

候选字符串可内部编码成四个整数，但只能用相等/不等计算距离，不能直接把整数差送进 RBF。

```python
distance = (x[:, None, :] != y[None, :, :]).sum(axis=-1)
kernel = np.exp(-distance.astype(np.float64))
```

每轮仅对已观测 J 计算均值/总体标准差，std <1e-12 时使用 scale=1，中心化结果为零。令 `y=(J-mean)/scale`：

```text
Kobs = k(X,X) + (noise_variance + jitter) * I
L = cholesky(Kobs)
alpha = solve(L.T, solve(L, y))
mu = k(X,Z).T @ alpha
v = solve(L, k(X,Z))
variance = diag(k(Z,Z)) - sum(v*v, axis=0)
acquisition = mu + 2 * sqrt(max(variance, 0))
```

仅对未观测候选选择采集最大值；严重负方差报错，微小舍入负值归零。固定以 `itertools.product(("B","P","G"), repeat=4)` 的顺序破同分，不依赖 set 遍历。最终 `argmax(observed_J)`；J 相同按既定评估顺序破同分。

GP 不优化核超参数，不训练性能预测器，不使用参数量，不传入任何四任务性能指标。

## 6. 按依赖顺序实施的任务

每个任务先加入能暴露关键风险的测试，运行确认失败，完成最小实现后运行对应测试。代码变更可按任务拆成独立提交；本文不要求本轮提交或启动实现。

### Task 1：配置、来源链与协议文件

**Create:** `kernel_stage_search/__init__.py`、`kernel_stage_search/protocol.py`、`configs/kernel_stage_search/swin_t_pascal.yaml`、`configs/kernel_stage_search/prepare_bbbb.yaml`。
**Modify:** `config.py`，默认配置区及路径解析。
**Test:** `tests/test_kernel_stage_protocol.py`。

1. 新增 `MODEL.KERNEL_SEARCH`，默认 `ENABLED=false, PHASE=off`；只允许 `off/prepare/score`，拒绝开启但 phase=off 等矛盾组合。
2. 新配置继承 `../stage_ablation/swin_t/resolved_baseline.yaml`，固定模型身份；不能触发 AGMTLORA 的 grouping search。
3. 配置覆盖附录所列参数；SOURCE_META_SPLIT、SCORE_CHECKPOINT、协议产物路径都按明确规则解析，记录绝对路径及内容 hash。导出 YAML 不依赖当前输出目录中的裸 `grouping.json`。
4. manifest 保存 schema、代码 commit/脏状态、模型/decoder 结构、实际 ranks、C0/M_B hash、split ID、变换和采样规则、数值模式、核参数、GP 参数及 seed。
5. 将 `protocol_id` 定义为语义设置和输入 hash 的 canonical JSON hash，避免仅因输出文件夹变化而失效；模型、数据内容、坐标、变换或评分实现版本变化必须失效。
6. `PHASE=score` 只允许独立评分脚本使用；`main.py` 遇到它立即拒绝训练。prepare 强制 BBBB、WORLD_SIZE=1、固定 split 和独立输出目录。

验收：旧配置默认解析结果和训练身份不变；缺失来源、rank 漂移、错误 task 顺序、坏 hash、非法预算均报错。测试命令：

```bash
python -m unittest discover -s tests -p "test_kernel_stage_protocol.py" -v
```

### Task 2：严格数据加载、固定组批与标签排列

**Create:** `kernel_stage_search/data.py`。
**Read/reuse:** `data/mtl_ds.py`、`data/custom_transforms.py`、旧 `meta_split.json`。
**Test:** `tests/test_kernel_stage_data.py`。

1. 实现 `load_source_split(config)`：按 ID 解析，校验重复、缺失、train/dev 交集及与官方 train ID 集合的关系。首版要求源 manifest 的两个集合完整覆盖当前官方 train。
2. 只读取官方 val 的 ID 清单做泄漏交集校验，不构造官方 val 图像或标签 loader。缺文件时预检失败，不重新随机切分。
3. 实现 `build_prepare_train_loader`，使用原 train transforms 与原 shuffle/drop_last 规则，仅映射 meta-train ID；模型类别数等构造字段保持原 MTL 契约。
4. 实现校准和评分 dataset：均从官方 train 的 eval-transform 视图映射 ID，禁止套用 `get_mtl_val_dataset`。
5. 按第 4 节固定组批，再逐任务采样；不足时按已声明规则扩组，保存实际批次成员和每任务坐标。采用无放回、固定 RNG，所有候选前向前完成。
6. `sample_plan` 保存全部标签、有效性/标签内容指纹、坐标与排列；检查实际输入文件 hash 或等效内容指纹，不能只依赖根目录路径或文件名。
7. 为 true/shuffled 使用同一坐标清单；生成每任务每批唯一排列，续跑直接读回。

验收：缺失 ID 不被静默过滤；无人体部件、255/零法线、单类批次、NaN、乱序 dataset 均有测试；任意候选顺序不改变 sample plan。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_data.py" -v
```

### Task 3：只准备一次 BBBB 评分模型

**Modify:** `main.py`、`utils.py`。
**Extend:** `kernel_stage_search/protocol.py`、`kernel_stage_search/data.py`。
**Test:** `tests/test_kernel_stage_prepare.py`。

1. 在 `main(config)` 建 loader 前识别 prepare 阶段，使用 Task 2 的 meta-train loader；validation dataset/loader 为 None，不调用原官方 loader。
2. 保留模型、损失、冻结、optimizer、AMP、LR 缩放和 C0 权重初始化顺序；不复制 `short_train_model` 的另一套训练逻辑。
3. 训练终点使用 `min(TRAIN.EPOCHS, PREPARE_EPOCHS)`，scheduler 仍按 TRAIN.EPOCHS=300 构造；修改保存条件确保索引 59 必定保存。
4. 建立统一 `allow_task_validation` 条件，覆盖初始/恢复/预训练后、周期、epoch 50 特例和最终 validate。prepare 下拒绝 `--eval`、throughput、teacher 等不属于准备流程的入口；不只依赖 `SKIP_INITIAL_EVAL`。
5. 训练侧损失日志可以保留；禁止用开发/官方指标挑准备 checkpoint。仅终点标记 `prepared_complete=true`。
6. 通过独立 `kernel_search_preparation` checkpoint 元数据记录 phase、split/protocol hash、C0、预算/horizon、完成 epoch。`utils.py` 保存/恢复时校验；不挪用或清除既有 `stage_ablation` 身份。
7. 同准备实验 resume 校验上述身份及 optimizer/scheduler/scaler/epoch；正式训练尝试 resume 准备 checkpoint 时明确拒绝，避免绕过数据协议检查。
8. 只有终点 checkpoint 能登记为 M_B；导出其 hash、模型 key/shape 摘要和准备记录。已有 M_B 可经相同检查登记复用，不重新训练。

验收：mock 验证 prepare 的官方 loader/validate 调用数均为零；确认只训练 BBBB、从 C0 加载、LR 只缩放一次、固定终点保存、scheduler horizon=300；关闭功能时原训练路径通过现有测试。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_prepare.py" -v
python -m unittest discover -s tests -p "test_unipora_training_alignment.py" -v
```

### Task 4：冻结 M_B 加载与可恢复的完整模式切换

**Create:** `kernel_stage_search/checkpoint.py`、`kernel_stage_search/policy.py`。
**Read/reuse:** `stage_ablation.py`、`models/lora.py`。
**Test:** `tests/test_kernel_stage_policy.py`。

接口：

```python
def load_scoring_model(config, checkpoint_path, prepared_manifest, device): ...

@contextmanager
def temporary_stage_modes(model, config, modes): ...
```

1. 在 BBBB 构造下严格验证 M_B 来源、阶段、key/shape、group rank 和准备终点，再仅加载 model state；允许的 attn_mask 重建规则显式列出，其他不匹配失败。
2. M_B 已是固定 group 结构，不再通过 C0 的 shared-to-group 映射；不能将其当 INIT_CHECKPOINT 绕过身份验证，也不能把它作为某个候选的 `--resume`。
3. 加载后 `requires_grad_(False)`、清 grad、`eval()`；评分统一 `torch.inference_mode()`，AMP 关闭。
4. context 使用 config clone 设置四位模式并复用 `configure_stage_ablation`；一次性快照所有父/子模块开关和 stage 属性，finally 精确恢复，原 config 不改。
5. 不调用 `prepare_trainable_parameters` 切候选，以免无意解冻参数；不建 optimizer。记录权重和 buffers 的初始/最终摘要，小模型测试对每候选逐 tensor 比较。
6. 避免跨候选并发访问同一模型；首版候选串行执行。

验收：`BBBB→GGGG→PBGG→BBBB` 输出恢复一致；异常也恢复；G 无 Prompt token，P 对关闭 LoRA 权重扰动不敏感；普通/split-QKV、S1、相邻 G、G→B/P 以及任务路由均覆盖。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_policy.py" -v
python -m unittest discover -s tests -p "test_mtlora_prompt.py" -v
```

### Task 5：任务感知的真实多尺度特征采样

**Create:** `kernel_stage_search/features.py`。
**Read/reuse:** `models/swin_mtl.py:203`、`models/swin_transformer_vpt.py:164`。
**Test:** `tests/test_kernel_stage_features.py`。

1. 在 `model.backbone` 注册 `with_kwargs=True` 的 forward hook，从 kwargs 中读取 task，核对 `return_stages=True`。
2. 运行真实 `model(images)`，使所有任务按既有顺序和完整四级配置执行；丢弃 decoder 预测，不计算任务性能。hook 内尽早采样，避免保留完整多尺度特征。
3. 从对应 layer 读取二维 resolution，将 token 重排，按 sample plan 的图像 ID 和坐标采样，保证行标签一一对应。
4. 按校准统计标准化、拼接，形成每任务 `[512,2112]`；微批次之间以保存的行索引组装，不能依赖 loader 到达顺序。
5. 验证每个预期 task/batch 恰好采集一次、有限值及尺寸；finally 移除 hook。只缓存采样特征或核，禁止存整图特征作为默认行为。

验收：坐标斜坡特征的 bilinear 结果正确；实际 56/28/14/14 尺度正确；hook 与直接 task backbone 调用结果一致；任务/图像顺序变换不会错误配对；没有包含 Prompt token。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_features.py" -v
```

### Task 6：校准、核矩阵与唯一 J 分数

**Create:** `kernel_stage_search/kernels.py`、`kernel_stage_search/calibration.py`。
**Test:** `tests/test_kernel_stage_kernels.py`。

1. 实现第 5.1 节的核和评分；对分类用 equality 核、法线用固定宽度 Gaussian 核。
2. 校准只使用训练侧 BBBB 样本，流式计算每任务每层通道均值/std；第二次遍历或保留固定样本求拼接特征宽度。
3. 宽度及统计一旦写盘，true/shuffled 和所有候选只读；禁止按候选重新求中位数或标准化。
4. 返回每批每任务 `Sx/Sy/Sxy/I2/ratio/n`，随后统一平均 J，额外返回核退化诊断。
5. 存 calibration hash、训练侧 ID、采样行、统计量、sigma、算法版本；任何输入不同禁止复用。

验收：常量特征核 I2≈0；分类特征核等于标签核时 ratio≈1；独立均衡的完整交叉小样本 I2≈0；与小矩阵 eigvalsh 参考值一致；共同排列 KF/KY 不改变分数。对有限随机样本，**不要求每次标签打乱后都严格为零或一定降低**。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_kernels.py" -v
```

### Task 7：离散类别 GP 和 12 次预算

**Create:** `kernel_stage_search/surrogate.py`。
**Test:** `tests/test_kernel_stage_surrogate.py`。

1. 生成 81 个合法字符串，固定三个初始候选；预算要求 3≤budget≤81。
2. 按第 5.2 节实现固定类别核、评分标准化、Cholesky posterior 和 UCB。
3. 建立评估去重集合，GP 只能推荐尚未完成评分的配置。
4. 常量评分、极小方差、同分、数值 jitter 有确定行为；不得把失败候选写成 J=0。
5. 最终选择函数仅读取真实 `observations` 中的 J，不读 acquisition 或未观测预测。

验收：类别重命名不改变距离；初始候选一致；mock scorer 正好被调用 12 次；81 点 acquisition 不引发 81 次模型评分；winner 必定来自已观测且 J 最大者；恢复后下一建议一致。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_surrogate.py" -v
```

### Task 8：独立评分 CLI、缓存和中断恢复

**Create:** `kernel_stage_search/runner.py`、`scripts/search_stage_kernels.py`。
**Test:** `tests/test_kernel_stage_runner.py`。

1. CLI 提供 `preflight/calibrate/search/export` 子命令；评分脚本不导入或执行 `main.py` 的训练入口。
2. preflight 输出 resolved 身份、文件存在性、split、硬件和估算张量量级；数据缺失即明确失败，不自动下载或伪造数据。
3. search 只读取锁定协议、M_B、sample plan、calibration；进入 Task 4 的模式上下文，Task 5/6 完成评分，再 Task 7 提议。
4. 结果按 `protocol_id/label_mode/modes` 原子写入；区分 pending/complete/failed。记录每批分数、J、实际耗时、峰值显存、参数统计、权重摘要和来源。
5. 稳定保存 observation 顺序和提议时的 mu/std/acquisition；完整评分才计入预算。恢复跳过 complete；失败或中断停下，不将它当低分并继续搜索。
6. 若需要避免中断后重做全部前向，按 batch 原子存中间结果；校验成功后补齐缺失 batch。预算按不同配置计算，报告实际发生的所有前向成本。
7. 特征核缓存使用独立 key：M_B、模式、batch、样本、标准化、sigma、数值模式，**不含 label_mode**；评分缓存必须含 label_mode 和 permutation hash。
8. 标签无关的 KF 缓存可让两条搜索共享同一候选的前向。默认逐 batch 落盘，不存完整模型输出；float32 KF 约 64 MiB/候选（16批×4任务×512²），实际记录 I/O 和磁盘使用。
9. 不改变缓存精度以节省磁盘；KF 计算与保存均 float32，读回后与在线结果相同，熵归约 float64。

验收：scorer 中禁止 optimizer/backward；模型 buffers 不变；坏缓存/hash 漂移被拒绝；中断恢复没有重复已完成的配置；CPU 合成样本可走通小预算。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_runner.py" -v
```

### Task 9：固定标签打乱对照

**Extend:** `data.py`、`kernels.py`、`runner.py`。
**Test:** `test_kernel_stage_data.py`、`test_kernel_stage_kernels.py`、`test_kernel_stage_runner.py`。

1. 对 true/shuffled 分别建立 GP observation 表，禁止合并。
2. shuffled 使用预存排列 `y_shuffled=y[perm]`，等效于 `KY[perm][:,perm]`；KF 与坐标不动。
3. 两条分支各自拥有 12 个实际评分名额，包括 BBBB/PPPP/GGGG。共享核缓存只省前向，不增加某条分支的预算或额外观测。
4. 记录每分支的 J_BBBB、winner、序列和成本；两个 winner 可相同，结论如实保存。
5. 校准共用，不为 shuffled 重训 M_B，不按候选更新排列。

验收：排列重载一致、normals 三维整体移动、Sy 精确或容差内不变；候选顺序改变不影响其排列；两条 GP 独立、共享核缓存结果等于重新前向评分。

### Task 10：锁定结构、正式训练配置与固定终点汇总

**Create:** `kernel_stage_search/export.py`、`kernel_stage_search/reporting.py`、`scripts/summarize_kernel_stage_search.py`。
**Test:** `tests/test_kernel_stage_export.py`。

1. export 只接受已完整完成预算的搜索记录，锁定 `max(observed_J)`，保存 winner 及对应记录 hash。
2. 为 BBBB、true winner、shuffled winner 各导出 seed 0/1/2 的配置，继承正式 baseline；配置变化限制在 modes、seed、名称/输出目录和关闭搜索节点。
3. 固定 `MODEL.INIT_CHECKPOINT=C0` 及其 hash、`MODEL.RESUME=''`、`TRAIN.AUTO_RESUME=false`、`START_EPOCH=0`、`EPOCHS=300`；保留 rank、任务、Prompt、loss 权重和 LR 缩放策略。
4. 不携带 prepare 的停止标记；`MODEL.KERNEL_SEARCH.ENABLED=false, PHASE=off`。导出的分组路径必须能从新 YAML 位置解析。
5. 配对实验同一 seed 的三组使用相同 C0 初始化和数据 RNG 协议；当前 checkpoint 加载应完整覆盖模型权重，不匹配即失败。保留原训练的非确定性设置并明确记录，配对 seed 不代表 GPU 位级一致。
6. 若某两组 modes 相同，可复用完全相同配置/seed 的训练产物，并明确标注共同来源，不能当成独立重复实验。
7. 汇总只接收每个 run 索引 299 的 checkpoint/hash 和该次验证四指标：semseg mIoU、human_parts mIoU、sal mIoU、normals mean angular error，与项目现有 `compute_delta_m.py` 的口径保持一致。sal 的 maxF/Beta maxF 可另附，但不替换主指标。保存原始数值及单位，汇总时再格式化，避免把 0…1 和百分数混用。
8. 输出每 seed 的四指标、相对配对 BBBB 的差、三 seed 均值/样本标准差、搜索成本。不得拼接四任务分别最好的 epoch。
9. 如另外报告 Delta m，预先固定参考和方向并注明；不能直接采用 `compute_delta_m.py` 默认 best-epoch 输出作为本实验结论。

验收：输出 YAML 可解析且 modes/grouping/C0 正确；prepare/M_B 不可作为正式 resume；四任务来自同一终点；相同结构去重不丢组别归属。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_export.py" -v
```

### Task 11：集成验证与使用文档

**Create:** `scripts/check_kernel_stage_search.py`、`README_KERNEL_STAGE_SEARCH.md`。
**Extend:** 上述测试，仅补跨模块合同。

1. 小模型 CPU 合成数据验收：冻结权重、真实模式组合、标签掩码、核评分、12 次 GP、恢复、两条标签分支、导出都可闭环。
2. 少数真实配置做 448 输入前向烟测（BBBB/PPPP/GGGG/PBGG/GBPG）；这只是结构验证，不对真实 81 个候选穷举评分。
3. 在具备数据和 CUDA 的环境进行真实标签小批次检查，确认四任务 Sy>0、sigma 有限、候选切换后的权重/buffers 不变、核没有全体数值崩溃。
4. 文档分别说明准备一次、前向搜索、正式训练三类成本；给出 true/shuffled 共用 M_B 的运行方法、产物 schema、缓存失效与恢复方法。
5. 运行已有 Prompt、推理和训练对齐回归；若发现旧失败，与未修改基线比较，不能把既有 README 的历史通过数字当成本次结果。

```bash
python -m unittest discover -s tests -p "test_kernel_stage_*.py" -v
python -m unittest discover -s tests -p "test_mtlora_prompt.py" -v
python -m unittest discover -s tests -p "test_mtlora_inference_benchmark.py" -v
python -m unittest discover -s tests -p "test_unipora_training_alignment.py" -v
python scripts/check_kernel_stage_search.py --device cpu --synthetic
git diff --check
```

GPU/真实数据检查和完整训练不应由普通 unittest 隐式启动。

## 7. 配置草案与预期运行顺序

下列字段均待 Task 1 实现，不代表当前仓库已支持。其余数值和规则按第 4 节加入 config schema。

```yaml
BASE:
  - ../stage_ablation/swin_t/resolved_baseline.yaml
MODEL:
  NAME: swin_t_kernel_stage_search
  KERNEL_SEARCH:
    ENABLED: true
    PHASE: score
    SOURCE_META_SPLIT: output/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr/default/ag_mtlora_stage1_prepare/run_20260907_031808/meta_split.json
    SCORE_CHECKPOINT: ""
    PREPARE_EPOCHS: 60
    CALIBRATION_IMAGES: 256
    INFO_BATCHES: 16
    INFO_BATCH_IMAGES: 8
    MAX_BATCH_IMAGES: 32
    SAMPLES_PER_TASK: 512
    FORWARD_BATCH_SIZE: 4
    BANDWIDTH_SAMPLES: 1024
    SAMPLE_SEED: 0
    SHUFFLE_SEED: 1
    LABEL_MODE: true_labels
    SEARCH_BUDGET: 12
    INITIAL_MODES: [BBBB, PPPP, GGGG]
    GP_NOISE_VARIANCE: 1.0e-6
    UCB_BETA: 2.0
TRAIN:
  EPOCHS: 300
  AUTO_RESUME: false
  USE_CHECKPOINT: false
```

`prepare_bbbb.yaml` 继承上面的配置，覆盖 `MODEL.NAME`、`PHASE=prepare`、`STAGE_ABLATION.MODES=[B,B,B,B]`；保持训练 AMP 配置。评分脚本单独关闭 AMP，不让评分 batch 改变准备/正式训练的 LR。

实现后的命令示例（不是本轮已执行命令）：

```bash
# 1. 检查来源和数据；生成基础 protocol 与初步 sample plan。
python scripts/search_stage_kernels.py preflight --cfg configs/kernel_stage_search/swin_t_pascal.yaml --output-dir output/kernel_stage_search/protocol

# 2. 只准备一次 BBBB；取本次输出目录内的 ckpt_epoch_59.pth。
torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py --cfg configs/kernel_stage_search/prepare_bbbb.yaml --disable_wandb

# 3. 在 meta-train 校准。M_B_PATH 替换为第 2 步的实际 checkpoint。
python scripts/search_stage_kernels.py calibrate --cfg configs/kernel_stage_search/swin_t_pascal.yaml --protocol output/kernel_stage_search/protocol/protocol.json --score-checkpoint M_B_PATH --output-dir output/kernel_stage_search/calibration

# 4. 两次搜索共享模型和校准；缓存由协议身份自动定位。
python scripts/search_stage_kernels.py search --protocol output/kernel_stage_search/protocol/protocol.json --calibration output/kernel_stage_search/calibration/calibration.json --labels true --output-dir output/kernel_stage_search/true
python scripts/search_stage_kernels.py search --protocol output/kernel_stage_search/protocol/protocol.json --calibration output/kernel_stage_search/calibration/calibration.json --labels shuffled --output-dir output/kernel_stage_search/shuffled

# 5. 只生成正式训练配置和运行清单，不在 export 中自动启动训练。
python scripts/search_stage_kernels.py export --true-run output/kernel_stage_search/true --shuffled-run output/kernel_stage_search/shuffled --output-dir output/kernel_stage_search/formal
```

协议产物分为两个不可混淆的身份：preflight 生成包含 C0/split/采样规则的基础协议；calibrate 在验证 M_B、完成训练侧校准及全部评估批次标签熵检查后，生成包含 M_B/calibration/sample-plan hash 的**完整评分协议**。search 必须校验完整组合，不能用尚未绑定 M_B 的 preflight hash 单独命中评分缓存。`calibration.json` 引用完整协议路径及 M_B 路径，供 search 读取；不要原地改写已有协议。完整协议引用校准张量文件的内容 hash，校准描述文件仅引用其路径，避免双方互相对描述文件求 hash 的循环依赖。

各搜索目录至少包含：

```text
resolved_protocol.json
observations.jsonl
scores/<modes>.json
proposal_history.jsonl
search_state.json
winner.json
cost_summary.json
```

`winner.json` 写明 label_mode、四位模式、J、delta_J、四任务分项、预算完成状态、C0/M_B/calibration/sample-plan hash；正式导出必须同时关联 true/shuffled 的记录。

## 8. 实施优先级、风险与当前环境限制

实施顺序：**协议/数据 → M_B 准备入口 → 冻结切换/特征 → 校准/核评分 → GP/runner → 打乱对照 → 正式配置/报告 → 集成验证**。

- 第一阶段交付 Task 1–6：能从合规 M_B 对 BBBB/PPPP/GGGG 得到可复现的 J，参数和 buffers 不变。
- 第二阶段交付 Task 7–9：两条 12 次信息搜索可中断恢复，产生已观测胜者。
- 第三阶段交付 Task 10–11：正式实验配置、固定终点汇总和回归验证齐全。

需要保留的研究限制：固定 BBBB 权重偏向、多尺度通道数对拼接距离的影响、核宽度/采样敏感性、像素相关性、旧 meta-val 的开发数据身份，以及有限预算与最终任务性能之间不存在保证。首版不通过新增网络、投影或复杂核来消除这些限制。

本地核查时 `PASCAL_MT/ImageSets/Context/train.txt` 和 `val.txt` 均不存在。因此可以继续实施单元测试和合成烟测，但**真实标签预检、校准、搜索及正式训练需要完整数据环境**。已有 Stage-1 输出和 C0 文件可以支持身份检查；本轮未验证可运行的 CUDA 训练环境，也未声称任何研究实验已完成。

本轮交付仅为上述新分支和代码修改计划。原有未跟踪的 `docs/figures/`、`docs/research/` 及 2026-10-11 计划均保留，没有覆盖或删除。
