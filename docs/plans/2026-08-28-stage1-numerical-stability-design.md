# Stage-1 Numerical Stability Design

## Goal

让 PoRA-CARA Stage-1 在与正式训练一致的数值条件下完成 warmup 和 affinity 搜索，并确保任何 NaN、Inf、异常 BN 状态或 AMP 不兼容 checkpoint 都会被明确记录并拒绝进入 Stage-2。

## Chosen approach

保留现有 affinity 定义、group proxy 和 partition search，不改变方法目标。修复集中在 Stage-1 的训练基础设施与产物验收：复用正式训练的全局 batch 学习率缩放，给 5 个 warmup epoch 和 50 个 affinity epoch 建立统一的 warmup/cosine scheduler，使用 autocast、GradScaler 和梯度裁剪，并安全处理 PASCAL human-parts 全 ignore batch。

Stage-1 的配置学习率仍保存原始实验值。运行时按 `batch_size * world_size / 512` 只缩放一次；生成的 resolved Stage-2 配置继续继承原始 YAML，避免 Stage-2 二次继承已经缩放过的数值。当前单进程 batch 9、`BASE_LR=1e-3` 的 Stage-1 实际峰值学习率应为 `1.7578125e-5`。

## Numerical contract

每个 Stage-1 batch 都检查输入上下文、各任务 loss、总 loss、用于 affinity 的梯度、optimizer step 后的 grad norm，以及 AMP loss scale。全 ignore 的 segmentation/human-parts batch 返回与输出图相连的可求导零损失，而不是 NLL 的 NaN。

模型按训练模式运行时，decoder BatchNorm 可以正常更新；每个成功 epoch 结束后检查整个 state dict 的有限性并原子保存 `last_good_checkpoint.pth`。最终 `post_affinity_checkpoint.pth` 保存前，使用训练模式和 Stage-2 相同的 autocast 执行 forward/backward smoke test，随后恢复 smoke test 前的模型 buffer。任何非有限输出、loss、梯度、参数或 buffer 都阻止正式 checkpoint 与分组产物成为可消费结果。

## Failure reporting

Stage-1 维护 phase、epoch、batch、task、sample IDs、LR、loss scale、per-task loss、有效标签比例、输出统计、首个非有限 tensor、GPU 显存和运行时版本。异常通过 `logger.exception()` 写入 `log_rank0.txt`，同时原子写入 `failure_report.json`（多进程为 `failure_report_rankN.json`）和 `status: failed` 的 `stage1_artifacts.json`。日志结束标记只能是 `STAGE1_ABORTED` 或 `STAGE1_COMPLETED`。

失败不会覆盖 warmup、post-affinity 或 grouping 正式产物。`last_good_checkpoint.pth` 保留用于复现和后续诊断；进程以非零退出码结束，让 torch launcher 能终止同组进程并向调用方报告失败。

## Acceptance criteria

- Stage-1 runtime snapshot 显示 batch 缩放后的学习率。
- 所有 Stage-1 日志不再出现未处理的 NaN/Inf。
- human-parts meta-val loss 为有限数。
- `last_good_checkpoint.pth` 只在完整成功 epoch 后更新。
- post-affinity AMP smoke test 通过，模型参数和 BN buffer 全部有限。
- 成功清单为 `status: complete` 并打印 `STAGE1_COMPLETED`。
- 任何注入的非有限 loss/gradient/state 都生成结构化失败报告、失败清单和 `STAGE1_ABORTED`，且不生成可用 post-affinity checkpoint。

