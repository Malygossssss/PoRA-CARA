# Kernel information stage search：实施与验证记录

日期：2026-10-20。分支：`feat/kernel-information-stage-search`。
起点：`2358122264dc93587820bd1117c20240e45181c5`。

已完成计划 Task 1–11 的代码、配置、使用文档和可在本机执行的验证。真实标签预检、M_B 准备训练、校准、研究搜索和正式训练尚未执行。

## 交付

| 计划任务 | 实现 |
|---|---|
| 1：配置与协议 | `config.py` 的默认关闭 schema；`protocol.py` 的 canonical identity、源码/输入 hash、实际 rank、初始化来源链、阶段与预算校验；两份新 YAML |
| 2：严格数据 | ID 集合覆盖/交集/重复检查；只读官方 val ID；官方 train 的 train/eval 视图；固定采样、标签有效性、内容指纹、跨图像行排列 |
| 3：一次 BBBB 准备 | `main.py` 在 loader 构造前分流；60 epoch 停止而 scheduler horizon 保持 300；所有任务验证关闭；独立准备 checkpoint 身份及终点登记；正式 resume/C0 屏障 |
| 4–5：冻结模型与特征 | 严格 M_B 加载；所有父子模式开关精确恢复；完整模型逐任务 forward hook；按真实 post-merging 尺度和像素中心采样；不建 optimizer |
| 6：校准与 J | 训练侧流式均值/总体 std、固定 reservoir 和带宽；float32 Gaussian/equality 核；float64 Renyi-2 熵；四任务等权 J 与退化诊断 |
| 7：类别 GP | 81 点，固定初始三点，Hamming 核、float64 Cholesky/jitter/UCB；只推荐未观测点，只从实际观测选胜者 |
| 8–9：CLI、缓存与对照 | `preflight/calibrate/search/export`；原子状态/分数/批次 float32 KF 缓存；失败停止和中断恢复；两条独立 GP，共享不含 label mode 的特征核；成本事件记录 |
| 10：导出和报告 | 从 C0 导出 paired seeds 0/1/2，重复结构合并产物并保留组归属；保存同一 epoch 299 checkpoint 的四任务原始指标/单位/hash；配对差、均值、样本标准差和成本汇总 |
| 11：集成与文档 | 小型真实 Prompted Swin/MTL 合成闭环、448 几何烟测、使用文档和既有回归 |

既有实现文件只修改 `config.py`、`main.py`、`utils.py`。backbone、LoRA、decoder、Stage-1 pipeline 和九组消融 YAML 保持原样。原有未跟踪的研究文档、图和旧计划保留；未提交 Git commit。

运行方法与产物合同见仓库根目录 `README_KERNEL_STAGE_SEARCH.md`。

## 已运行验证

本机为 Windows，使用仓库内忽略的 `.venv`（Python 3.10.11、PyTorch 2.5.1+cpu、NumPy 1.26.4、YACS 0.1.8、timm 0.9.2）。CPU 测试设置 `OMP_NUM_THREADS=1`；模型训练依赖没有被写入新的 GP requirements，也没有添加 BoTorch/GPyTorch/COMBO。

| 验证 | 本次结果 |
|---|---|
| `test_kernel_stage_*.py` | 33 项通过 |
| `test_mtlora_prompt.py` | 19 项通过 |
| `test_mtlora_inference_benchmark.py` | 7 项通过 |
| `test_unipora_training_alignment.py` | 8 项通过 |
| 上述四组最终联合运行 | **67 项通过，9.973 秒** |
| `check_kernel_stage_search.py --device cpu --synthetic` | 通过；true/shuffled 各 12 个观测；状态摘要不变；共享缓存；导出 9 个唯一配置 |
| `check_kernel_stage_search.py --device cpu --structure` | 通过；BBBB/PPPP/GGGG/PBGG/GBPG，每种均检查四任务 448 前向，输出尺寸 56/28/14/14、通道 192/384/768/768 |
| C0 SHA256 | 已实际读取并核验：`b05959165782d239377e0ab144ff52f6426c3f0327c4851235ba75ff67e20281` |
| `preflight` | 按预期 exit 2，指出缺少 `PASCAL_MT/ImageSets/Context/train.txt`；另行确认 train/val 清单均不存在 |
| `compileall`、`git diff --check`、`pip check` | 通过 |

新增测试还覆盖：准备 loader/官方验证调用隔离、固定终点保存、LR 只缩放一次且准备/评分身份相同、准备 resume 屏障、普通/split-QKV 子模块开关恢复、异常恢复、真实 task backbone 与 hook 一致、坐标插值参考值、熵 eigvalsh 小矩阵参考、标签排列谱不变、缓存损坏/失败恢复、输出目录无关的张量序列化 hash，以及正式 endpoint 写入→重读→汇总。

合成检查得到的 `PGBG`/`GGPG` 只属于显式 synthetic fixture，不是研究胜者；CLI 阻止将 synthetic 记录用于研究评分或正式研究导出。

## 旧 2LR 测试的既有失败

额外运行 `test_unipora_cara_2lr_config.py`：4 项中 2 项通过、2 项失败。使用 `git show HEAD:<file>` 在临时目录重建未修改起点，再运行完全相同测试，复现相同两个失败：

- `test_formal_training_keeps_unipora_lr_scaling` 仍要求 `main.py` 内联的 `linear_scaled_lr` 文本；起点已经使用共享的 `scale_learning_rates`。
- `test_stage1_and_resolved_config_propagate_the_runtime_base_lr` 要求 Stage-1 源码不含 `build_scheduler`；起点已经包含该调用。

没有为本功能改写这两项历史断言，也没有将它们算作本次引入的回归。日志分别在 `output/kernel_2lr_regression.log` 和 `output/kernel_2lr_baseline_comparison.log`。

## 尚未验证的研究执行

- 当前测试环境 `torch.cuda.is_available() == False`，且真实 train/val ID 清单缺失。未声称 CUDA/NCCL 训练、真实标签熵、真实 sigma 或核评分已通过。
- 需要在完整数据及 CUDA/NCCL 环境依次运行真实 preflight、一次 M_B 准备、校准、两条搜索、正式配置导出和配对训练。
- 代码正确性不保证 J 预测最终性能，也不保证 12 次预算找到全局最优；BBBB 偏向、像素相关性、带宽敏感性、旧 meta-val 的开发数据身份和更早未知数据历史均保留。

主要运行日志位于 `output/kernel_stage_validation.log`、`output/kernel_synthetic_check.log`、`output/kernel_structure_check.log`、`output/kernel_preflight_check.log`。这些本地日志沿用仓库的 output 忽略规则。
