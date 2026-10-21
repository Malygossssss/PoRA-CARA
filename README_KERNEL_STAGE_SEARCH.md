# 任务相关核信息 Stage 搜索

实现计划见 `docs/plans/2026-10-20-kernel-information-stage-search-implementation.md`。
这是独立的冻结前向搜索入口。真实数据、CUDA/NCCL、C0 和完成准备的 M_B 都需要用户环境提供；单元测试和合成检查不会启动真实训练。

## 固定协议

- 四个 stage 共 81 个 B/P/G 配置，所有任务共享结构；任务路由保持原样。
- 固定任务顺序 `semseg, normals, sal, human_parts`，实际 group ranks 为 16/48，50 个 Prompt，448 输入。
- 从 C0 在旧 meta-train 准备 BBBB 60 epoch，保存索引 59 的固定终点。scheduler horizon 仍为 300，warmup 20，原训练 batch=10、LR 只缩放一次。准备过程不创建官方 val loader，也不运行 `validate`。
- 使用官方 train 的 eval-transform 视图读取校准和信息样本。旧 meta-train/meta-val ID 必须完整覆盖官方 train，且与官方 val ID 无交集。缺文件、缺 ID、重复、交集均失败，不重新切分或下载。
- 校准固定选择 256 张训练侧图像，每 4 张组批，每任务均匀无放回取 512 个有效位置；扩组规则与信息批相同。流式计算每任务、每 stage、每通道的总体均值和标准差，std 下限 1e-6。固定优先级 reservoir 最多保留 1024 行用于中位正平方距离宽度；法线先单位化，使用同样的训练侧宽度规则。
- 信息批 16 个，每批从固定 meta-val 排列取 8 张，按排列向后扩至最多 32 张。扩组可能使批次重叠；不足、单类或退化标签熵会使协议准备失败。每任务取 512 个有效位置，微批前向为 4 张。每批的起点为 `batch_index * INFO_BATCH_IMAGES`，扩组不足时不为不同候选重新抽样。
- 完整模型逐任务前向，通过 backbone hook 捕获 post-merging tokens；读取 `layer.output_resolution`，按目标像素中心 bilinear 采样。448 时真实尺度是 56/28/14/14、通道 192/384/768/768，拼接为 2112 维。
- 分类用 equality 核，特征和法线用固定 Gaussian 核。距离/缓存为 float32，熵和 GP 为 float64。`I2=S2(A)+S2(B)-S2(A∘B/trace(A∘B))`，`J` 是每批四任务 `I2/S2(B)` 的等权均值。不裁剪 ratio，不用任务性能、参数量或时间作为目标。
- 初始顺序 `BBBB → PPPP → GGGG`，默认预算 12。GP 核 `exp(-Hamming)`，观测分数标准化，噪声方差 1e-6，UCB 为 `mu+2*std`；jitter 从 1e-10 到 1e-4。候选按 `product((B,P,G), repeat=4)` 顺序破同分，胜者仅从实际观测取最大 J；J 同分保留最早观测。
- true/shuffled 共用 M_B、坐标、统计和宽度。每任务每批固定 seed=1 的行排列，法线三通道整体移动；两条 GP 各自拥有完整预算。BBBB 可以胜出。

修改协议设置、输入文件、变换、模型或评分实现会产生新 identity，旧缓存不可复用。

## 运行顺序

在项目根目录使用已有项目 Python 环境。评分不依赖 BoTorch、GPyTorch 或 COMBO。

```powershell
# 1. 真实输入检查，生成基础协议、配置快照和两个采样计划。
python scripts/search_stage_kernels.py preflight --cfg configs/kernel_stage_search/swin_t_pascal.yaml --output-dir output/kernel_stage_search/protocol

# 2. 只准备一次 M_B。需要支持 CUDA/NCCL 的训练环境。
torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py --cfg configs/kernel_stage_search/prepare_bbbb.yaml --disable_wandb

# 3. 替换为本次准备输出目录内的实际 ckpt_epoch_59.pth。
python scripts/search_stage_kernels.py calibrate --protocol output/kernel_stage_search/protocol/protocol.json --score-checkpoint M_B_PATH --output-dir output/kernel_stage_search/calibration

# 4. 两条独立搜索，共享 feature-kernel 缓存。
python scripts/search_stage_kernels.py search --protocol output/kernel_stage_search/protocol/protocol.json --calibration output/kernel_stage_search/calibration/calibration.json --labels true --output-dir output/kernel_stage_search/true
python scripts/search_stage_kernels.py search --protocol output/kernel_stage_search/protocol/protocol.json --calibration output/kernel_stage_search/calibration/calibration.json --labels shuffled --output-dir output/kernel_stage_search/shuffled

# 5. 锁定已观测胜者，生成正式配置；不自动启动训练。
python scripts/search_stage_kernels.py export --true-run output/kernel_stage_search/true --shuffled-run output/kernel_stage_search/shuffled --output-dir output/kernel_stage_search/formal
```

`--device cpu` 可显式用于评分诊断；完整协议绑定 PyTorch/CUDA 版本和 device type，切换数值环境需要重新校准。评分强制 eval/inference_mode、float32、AMP/TF32/cudnn benchmark 关闭，确定性算法开启；不构造 optimizer，不修改参数或 buffers。准备和正式训练保留原 AMP/冻结策略。单进程 GPU 训练仍使用原 `main.py`；`PHASE=score` 在该入口被拒绝。

相对的核搜索输入路径按项目根目录解析，grouping 继续复用 baseline 的相邻文件解析。生成的快照和正式配置使用可解析的绝对输入路径。迁移机器时需更新路径并重新验证内容，不能盲目复制旧路径。

## 产物、身份与恢复

基础 `protocol.json` 保存 C0、严格 split、原始数据文件内容 hash、实际模型/分组/训练身份、采样/数值/GP 设置、代码 commit 和脏状态。`protocol_id` 只取 canonical semantic JSON hash，不包含输出目录或生成时间。工作副本中的评分、模型、数据和训练入口源码内容也绑定到 identity。

`sample_plan.json` 和 `calibration_plan.json` 保存 image ID、标签坐标、标签、分布、有效位置数、固定排列和变换后 image/label 内容指纹。评分前检查 hash、输入内容及坐标处真实有效标签，避免仅按路径复用。正常无人体部件标注图像由原 transform 置 255，不作为有效背景；零/255/NaN/Inf 法线排除。

准备 checkpoint 使用独立的 `kernel_search_preparation` 元数据，保留原 `stage_ablation` 身份。只有完整终点能生成 `*.prepared.json`。该文件绑定 checkpoint 内容 hash、全部 key/shape/dtype 和准备来源。M_B 严格加载，不经过 C0 的 shared-to-group 映射，仅允许重建 `.attn_mask` buffers。准备 checkpoint 不能作为正式 resume 或 C0。

校准生成 `calibration.pt`、`scoring_protocol.json` 和描述文件 `calibration.json`。完整评分协议绑定 M_B、准备记录、校准张量、两个采样计划及数值环境；不会原地改写基础协议。校准张量 hash 在完整协议中，描述文件引用协议，避免循环 hash。

每条搜索目录包含：

```text
resolved_protocol.json
observations.jsonl
proposal_history.jsonl
search_state.json
scores/BBBB.json ...
winner.json
cost_summary.json
```

`search_state.json` 是原子保存的权威 observation 顺序，两个 JSONL 是恢复后重建的视图。score 区分 pending/complete/failed，只有完整评分计入预算。失败立即停止，不写 J=0。修复环境后重跑相同 `search` 命令，会沿相同 GP 建议继续，跳过完整候选并复用已落盘的批次核。

共享核默认位于 calibration 目录旁的 `feature_kernels/<protocol_id>/<modes>/batch_NNN.pt`。每个 float32 核文件有内容 hash 和独立身份校验；key 不含 label mode。默认约 64 MiB/候选；不降精度、不保存整图特征。损坏或 identity 漂移会明确失败，应使用新的输出/协议目录；不要把修改 hash 当作修复。命令持有共享缓存文件锁，避免两个进程同时写同一协议。

每次实际批前向有独立 cost event，包括中断/失败。恢复不会抹掉已发生的成本；未完成事件无法精确恢复用时，会显式计数。两条分支成本中共享前向的 scope 相同，汇总时不能将同一事件重复相加。

## 正式训练和报告

导出 BBBB、true winner、shuffled winner 的 seed 0/1/2 配置，全部从共同 C0、epoch 0 重新训练 300 epoch。清空 resume，关闭 auto-resume 和搜索分支，不携带准备停止标记。相同 modes/seed 的组复用同一个配置和训练产物；`formal_runs.json` 保留组别归属和两个锁定胜者记录。配对种子不保证 GPU 位级一致。

正式四任务 Stage 实验在最后一次验证后自动写 `formal_endpoint_299.json`，绑定索引 299 的 checkpoint/hash。原始 mIoU 使用 fraction，法线 mean 使用 degrees，sal 主指标仍为原实现的 mIoU；maxF 可另附。汇总只使用同一终点，不拼接任务最优 epoch，不调用默认 best-Delta-m 逻辑：

```powershell
python scripts/summarize_kernel_stage_search.py --formal-manifest output/kernel_stage_search/formal/formal_runs.json --endpoints RUN_1/formal_endpoint_299.json RUN_2/formal_endpoint_299.json ... --output output/kernel_stage_search/formal/summary.json
```

每个唯一 modes/seed 提供一个 endpoint 文件，共 6 或 9 个（全部 winner=BBBB 时为 3 个）。输出逐 seed 四指标、相对同 seed BBBB 的原始差值、三 seed 均值与样本标准差、搜索成本。指标差为 run−BBBB，mIoU 正值改善、法线误差负值改善。

## 验证

```powershell
python -m unittest discover -s tests -p "test_kernel_stage_*.py" -v
python -m unittest discover -s tests -p "test_mtlora_prompt.py" -v
python -m unittest discover -s tests -p "test_mtlora_inference_benchmark.py" -v
python -m unittest discover -s tests -p "test_unipora_training_alignment.py" -v
python scripts/check_kernel_stage_search.py --device cpu --synthetic
python scripts/check_kernel_stage_search.py --device cpu --structure
git diff --check
```

`--synthetic` 使用显式合成标签和缩小的真实 Prompted Swin/MTL 模型，检查采样、训练侧校准、冻结权重、真实模式切换、两条 12 次 GP、批次缓存和恢复。默认用临时目录，`--output-dir` 可保留检查产物；这些产物标明 synthetic，不可用作研究输入。`--structure` 只对五种组合和四个任务做 448 几何前向，不读取真实标签，也不进行候选穷举。真实 CUDA 标签预检、校准和研究训练均不会由普通 unittest 隐式启动。

代码验证通过仅表示实现合同通过。固定 BBBB 权重偏向、通道数对距离的影响、采样/宽度敏感性、像素相关性、旧 meta-val 的开发数据身份和早期未知数据来源仍需如实报告。12 次搜索不保证找到全局最优，J 不保证预测最终任务性能。
