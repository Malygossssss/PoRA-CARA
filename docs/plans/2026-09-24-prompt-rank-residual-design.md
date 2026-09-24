# 从 rank 掩码到多 prompt 条件残差

日期：2026-09-24。状态：已实现可选原型，尚无 E5 的 PASCAL 训练结果。下文的性能解释是待检验假设，不是实验结论。

## 1. 已确认的事实

用户的正式训练使用 `rank_extract_e4.yaml`、单进程、batch 8、300 epochs，从 Stage-1 的 `post_affinity_checkpoint.pth` 初始化，并指定对应 grouping。E4 确实开启；Stage-1 本身关闭提取器是当前两阶段设计的正常行为。

| 指标 | 用户 main 对照 | 用户 E4 | E4 − main |
| --- | ---: | ---: | ---: |
| Delta m (%) | 5.274 | 5.121 | −0.153 个百分点 |
| SemSeg mIoU | 70.65 | 70.09 | −0.56 |
| Human Parts mIoU | 61.59 | 61.54 | −0.05 |
| Saliency mIoU | 65.61 | 65.80 | +0.19 |
| Normals mean，越低越好 | 15.94 | 15.94 | 0.00 |

按仓库 `compute_delta_m.py` 的四任务 ST 参考值复算，差异为 −0.15230 个百分点。其中 SemSeg、Human Parts、Saliency 分别贡献 −0.20830、−0.02018、+0.07618 个百分点。首先需要解释的是 SemSeg 的损失，不能描述为所有任务都下降。

本地 `csv/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr-32-32.csv` 的 epoch 290 与用户 main 数字一致。该文件 epoch >= 250 的 Delta m 平均为 5.21418%，范围 5.112–5.274%；这只是同一训练的时间波动，不能替代跨 seed 方差。另一个无 `-32-32` 后缀的历史 CSV 记录 best 5.430%，但缺少对应设置核验，不能据此认定它可与 E4 公平比较。

当前 feature 分支还修正了 prompt 的窗口 batch 排列。main 的旧行为可能将不同样本的 prompt 窗口混合。因此必须保留两个参照：用户 main 是历史性能目标；同分支 extractor-off 的 corrected E0 是判断新机制贡献的直接对照。不能把 main→E4 的全部差异归因于 mask。

在当前命令下，原始 `BASE_LR=1e-3` 经 `8/512` 缩放得到实际峰值 `1.5625e-5`，Stage-1 和 Stage-2 都应核验日志中的实际值。不要在结构比较中同时改变 batch、world size 或 LR。

## 2. 当前机制的局限与不能下的结论

E4 的真实实现是 `2*sigmoid(logits)`，零初始化 logits 得到精确的 mask=1，范围为 (0,2)。所以它已经允许增强，也已具备恒等初始化。将其改写成 `z+(m-1)*z` 没有扩大函数族，不能作为新方案。

已确认的结构限制：

1. 50 个交互后的 prompt token 先平均，再用共享 A 投影；同均值的不同 prompt 集合在此条件支路不可区分。
2. 最终干预为 `diag(m) z`。虽然 mask 网络可以读取所有 rank，写回操作依然逐坐标相乘；固定当前 A/B 时，某坐标 z_j=0 就无法通过该坐标产生新值，正 mask 也不改变 z_j 的符号。
3. stage 3、4 的八个 fc1 全部调制，可能造成多层累积扰动。是否因此损害语义分割，需要 stage-last 消融，不能从当前结果直接断定。
4. LoRA 的分解非唯一：`BA=(BR^-1)(RA)`。rank 轴没有自动获得独立语义，不能假定每个任务的知识能沿坐标轴干净分离。这不证明门控无效，也不证明下面的方案有基变换不变性。

仍未知：E4 的逐层 mask 分布、条件支路梯度、不同任务的相似度、corrected E0 指标、配对 seeds、远端 checkpoint 和 grouping 内容。当前无法确认饱和、梯度冲突或过拟合是实际原因。

## 3. 三条路线及选择

| 路线 | 具体变化 | 性能动机 | 创新性与代价 |
| --- | --- | --- | --- |
| E5：多 prompt 条件残差，优先候选 | 保留所有 prompt；逐 patch 检索；中心化；加性 rank 写回 | 避免平均池化丢失 token 差异，允许补充当前较弱的坐标，保留原 LoRA 主路径 | 有清楚的组内机制假设；attention/残差本身已有先例；增加检索计算 |
| E4-last：缩小现有门控范围 | 只在 stage 3、4 最后 block 的 fc1 开启 | 检验是否是过多调制位置造成退化 | 工程与消融价值高，不作为主要新颖性来源 |
| 轻量 task-specific rank mixer / 小 rank task LoRA | 给任务直接的低成本特化容量 | 判断瓶颈是否只是任务自由度不足 | 可能更容易优化，但偏离“只有 prompt 是任务专属条件”的主线，且与已有 PEFT 工作接近 |

第一轮只实现 E5 与 E4-last，不同时引入空间卷积、多个专家、蒸馏或新增 loss；否则很难解释哪项变化产生收益。

## 4. E5 定义

暂用描述性名称 **Prompt Rank Residual**，不作“首创”声明。

每个目标 fc1 的实际任务流经过 attention 残差与 norm2 后为 `[P_t; X_t]`。用行向量记法：

```text
Z = Dropout_existing(X_t) A_g^T          # [B, N, r]
E = P_t A_g^T                           # [B, P, r]，保留所有 prompt
Q = L2Normalize(Z), K = L2Normalize(E)
S = softmax(Q K^T / tau, dim=prompt)     # [B, N, P]
C = S (E - mean_prompt(E))              # [B, N, r]
Z_new = Z + eta C W_g^T                 # W_g 为 r×r，初始化为 0
Delta H = shared_scale_g Z_new B_g^T
```

实现保持原始 `[P;X]` 上的一次 LoRA dropout 及随机数序列，再切出 patch rank；条件 E 使用 dropout 前的 prompt，和 E4 一致。prompt token 自身的 LoRA 残差不经过此模块。

默认设置：零起始 `STAGES=[2,3]`、`BLOCKS=stage_last`、`MODULES=[fc1]`，`eta=0.1`，`tau=0.25`。每个位置每个组只有一个 W，组内任务共用，不增加 task-specific LoRA、不改变 CARA 分组或原 rank budget。

中心化使均匀检索只贡献零残差：当 patch 没有偏好某些 prompt 时，不额外注入一份全局 prompt 均值。softmax 在同一任务的 prompt token 上归一化，不要求任务互斥，也不要求不同任务的 mask 加和为 1。

只有 W 零初始化，eta 为固定非零数。若二者都为零，会阻断这条支路的起始学习。W 在第一步获得梯度；经过它的 prompt 条件梯度在初始化时为零，W 更新后开始流动。A/B 和 prompt 仍可通过已有 backbone 路径训练。

Q/K 点积、softmax 和检索使用显式关闭 autocast 的 FP32 上下文，再返回原 rank dtype 做写回。不能只调用 `.float()` 就假设 AMP 点积一定保留 FP32。

**边界：**恒等初始化只保证相同旧权重下的起点一致，不保证训练后不退化。E5 可以在 rank 坐标间产生补充，但输出仍经过 B，不能声称扩大到 B 列空间之外。prompt token 不是有标签的语义原型；若它们塌缩为相同向量，中心化残差会退化为零。固定 eta 也不是对残差范数的硬约束。

## 5. 开销与已有工作

假设实际为两个 r=32 的组，两个 stage-last 位置：E5 新增 `2*2*32²=4096` 个参数。E4-last 对应 6336 个参数，原八位置 E4 对应 25344 个参数。实际组 rank 不同时必须重新统计，配置名 r64 指总 budget，不必然指每组 rank64。

448 输入下，stage 3、4 patch 数为 784、196，prompt 数 50。粗略增加约 5.98M MAC/任务前向，包括两次 prompt 投影、query-key/value 检索和 rank 写回，不含归一化/softmax等逐元素操作。低参数量不等于低延迟；需实测峰值显存与吞吐。

- [SpEx+](https://www.isca-archive.org/interspeech_2020/ge20_interspeech.pdf)：共享编码、条件选择是原路线的来源。其语音消融不能证明图像 prompt 与 A 的共享投影一定更优；原模型还包含说话人判别监督，与本任务存在区别。
- [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)：提供连续关联检索与 attention 的联系。这里只借鉴检索视角，不能把其容量或收敛结论直接移植到 E5。
- [LoRA-XS](https://arxiv.org/abs/2405.17604)：在低秩因子之间引入可训练小矩阵已有先例。因此“增加 r×r 矩阵”不能单独声称创新。
- [Conv-LoRA](https://arxiv.org/abs/2401.17868)：低秩路径中加入卷积也已有先例，简单改成 3×3 不能自动满足创新性要求。

E5 待论证的具体贡献是：CARA 解决组间共享，任务 prompt 在共享 rank 空间中提供多个可检索条件，由 patch 选择相对 prompt 均值的补充信息，以低成本加性残差完成组内特化。必须证明多 token、中心化和条件相关性确实有用；否则应收缩论文主张。这不是穷尽文献后的新颖性保证。

## 6. 最小实验顺序

所有实验共用同一次、当前窗口语义的 Stage-1 grouping/checkpoint，训练时长、seed、batch、数据划分、LR 和评估频率一致。用户已经按当前分支重新生成的合格 Stage-1 产物可复用，无需为了 E5 再搜索分组。

| 优先级 | 实验 | 要回答的问题 |
| --- | --- | --- |
| 1 | corrected E0 | 修正窗口后，没有 rank 模块的真实对照是多少？ |
| 2 | E4-last | 八个位置减成两个后是否恢复 SemSeg？ |
| 3 | E5-last | 同样两个位置，检索残差是否优于逐坐标门控？ |
| 4 | E5 `CENTER_VALUES=False` | 中心化是否有用，还是删掉了必要的全局任务信息？ |
| 5 | E5 不同 prompt/固定 prompt、独立条件投影、参数量匹配的普通残差 | 收益是否真的来自任务条件、多 token 和共享空间？这些研究消融尚未全部实现。 |

已有 E4-all 可直接列入对照，前提是与 E0/E5 训练条件一致。先用一个配对 seed 筛选机制；最终至少三个配对 seeds，报告 Delta m 与四项指标的 mean/std 和逐 seed 差值。三次也可能不足以分辨很小的收益，应如实报告不确定性。

首要目标是提升修正后的 E0，同时恢复 SemSeg。5.274% 是历史目标，不能代替 corrected E0。0.1–0.2 个百分点的单次最优差异不能单独作为稳定提升的依据。不把某个预设高于基线的数字当作预测。

按预先固定的完整评估点选择单个多任务 checkpoint，保留四任务配对数值；不拼接各任务各自的最佳 epoch。可同时报告末若干次评估均值来观察稳定性，但不能将其当作独立重复样本。若做 80/100 epoch 初筛，E0 也使用相同 schedule，且短跑结果只决定优先级，不能证明 300 epoch 的最终排序。

建议诊断时记录：E4 的 `mean/std(mask-1)`、极端值比例；E5 的 attention 熵、prompt token 方差、`||eta*C*W^T*B^T|| / (||Z*B^T||+eps)`；两者的逐组 A/B 与模块梯度。当前实现未自动输出这些诊断，不将其写作已观测证据。

若 E5 未超过 E4-last，优先接受更简单的解释；若检索与随机/错任务 prompt 没有差别，不以任务条件检索作为论文主张；若独立投影更好，不宣称共享 A 的对齐优势。

## 7. 服务器命令

直接沿用用户已有的 `STAGE1_DIR` 与 `GROUPING_JSON`，确认两者属于同一 Stage-1 run。

```bash
CFG=configs/mtlora/tiny_448/pascal/unipora_cara_tiny_448_r64_prom50_prompt_rank_residual_e5.yaml
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 --master_port=29502 main.py \
  --cfg "$CFG" --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 --epochs 300 --ckpt-freq 20 --eval-freq 5 \
  --resume "$STAGE1_DIR/post_affinity_checkpoint.pth" \
  --opts MODEL.AGMTLORA.GROUPING_JSON "$GROUPING_JSON"
```

只替换 CFG，分别运行：

```text
configs/mtlora/tiny_448/pascal/unipora_cara_tiny_448_r64_prom50_corrected_e0.yaml
configs/mtlora/tiny_448/pascal/unipora_cara_tiny_448_r64_prom50_rank_extract_e4_last.yaml
configs/mtlora/tiny_448/pascal/unipora_cara_tiny_448_r64_prom50_prompt_rank_residual_e5.yaml
```

显式固定 seed 时给每个命令添加相同的 `--seed`；最终配对重复使用相同的 seed 列表。移除中心化的消融在 opts 后追加 `MODEL.MTLORA.RANK_EXTRACT.CENTER_VALUES False`，并用 `--name` 指定独立实验名。其它语义设置改变时也应使用独立名称。

当前 `main.py` 按保存周期写 checkpoint，不会因多任务 Delta m 最优而自动额外保存。用户的 `--ckpt-freq 20 --eval-freq 5` 可能只有 best epoch 的日志而没有对应权重；如果需要重评估每个候选或分析 best checkpoint，将所有对照统一改为 `--ckpt-freq 5`。这不改变优化步数，但增加磁盘占用。

新模块继续使用 `lora_rank_extractors` 参数命名，因此会被现有 LoRA trainable 规则纳入优化器和参数统计。E5 内部使用不同的 `writeback.weight` keys，E4 checkpoint 不能误当 E5 完整续训；应从共同的 Stage-1 checkpoint 初始化。E5 同结构续训需保留原配置。日志仍统一显示 `Rank extractor params`，应大于零，并核对最终保存的 `MODE=prompt_residual` 与 `BLOCKS=stage_last`。

不要把已训练 300 epoch 的 E0/E4 再微调后，直接与仅训练 300 epoch 的基线比较。若采用这种部署导向路线，基线也必须获得等量继续训练预算。

## 8. 验证记录

验证环境：Windows、Python 3.10.11、PyTorch 2.5.1+cpu，测试时 `torch.set_num_threads(1)`。环境隔离在 git 忽略的 `output/rank_residual_test_env`，未修改项目依赖清单或全局 Python。

- `tests.test_prompt_rank_residual`：13 项通过，包括 FP16/BF16 CPU autocast 子用例、前向恒等性、第一步写回梯度、prompt 梯度、组路由、零坐标补充、同均值但不同 token 的条件响应、prompt 顺序不变性、batch 独立性、中心化对照、模型目标位置与 checkpoint 结构检查。
- `tests.test_mtlora_prompt`：24 项通过，原 E4 与 prompt 路径回归通过。
- `tests.test_unipora_training_alignment`：8 项通过。
- 上述直接相关的 45 项测试通过。日志：`output/prompt_rank_residual_tests.log`；增加 FP16 子用例后的新模块复跑也全部通过，记录在 `output/prompt_rank_residual_stage1_tests.log`。
- `tests.test_ag_mtlora_stage1`：26 项中 21 项通过，2 failures + 3 errors。将已提交 HEAD 导出到独立目录后复跑，得到相同五项失败；对照日志为 `output/prompt_rank_residual_head_stage1_tests.log`。失败分别涉及三个旧测试配置缺少 `AFFINITY_COLLECT_EPOCHS`、模拟参数 `non_lora_weight` 被现有名称匹配规则识别为 LoRA、以及 stage 数与 rank 列表长度不匹配。本次未修改这些既有测试或 Stage-1 实现，不能声称全仓库测试全绿。
- 修改的 Python 文件通过 `py_compile`，`git diff --check` 通过。

运行直接相关测试：

```bash
python -m unittest tests.test_prompt_rank_residual tests.test_mtlora_prompt tests.test_unipora_training_alignment -v
```

CPU 数值测试不替代 CUDA AMP、显存/吞吐与完整训练检查。没有远端 GPU 实验结果，不报告 E5 的 mIoU 或 Delta m。
