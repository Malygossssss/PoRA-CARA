# Swin-T backbone 逐 stage 消融

分支：`feat/swin-stage-ablation`。默认 `MODEL.STAGE_ABLATION.ENABLED=False`。
开启时必须使用已解析的 `fixed_json` grouping、零 task-specific rank，并保持
backbone downsampler LoRA 关闭。不重新搜索 grouping 或重新分配 rank。

## 九组配置

位于 `configs/stage_ablation/swin_t/`。九份 YAML 均继承同一份
`resolved_baseline.yaml`，只覆盖模型名称和 MODES。

| 文件 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| baseline.yaml | B | B | B | B |
| S1-P.yaml | P | B | B | B |
| S2-P.yaml | B | P | B | B |
| S3-P.yaml | B | B | P | B |
| S4-P.yaml | B | B | B | P |
| S1-G.yaml | G | B | B | B |
| S2-G.yaml | B | G | B | B |
| S3-G.yaml | B | B | G | B |
| S4-G.yaml | B | B | B | G |

B 启用 task-prompt 与 group-LoRA；P 让该 stage 的 group-LoRA 前向增量为零；
G 真正移除该 stage 的 prompt token，继续按 active_task 路由 group-LoRA。
G 可以接收上游已受 prompt 影响的图像特征，后续 B/P 用自身 deep prompt 重新注入。
原参数、shape、checkpoint key 全保留，关闭参数被冻结并排除在 optimizer 之外。

## 共同起点与固定协议

来源是已完成、schema=2 的：

```text
output/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr/default/
  ag_mtlora_stage1_prepare/run_20260907_031808/
```

`resolved_baseline.yaml` 保留正式 resolved YAML 的分组与手动 group rank，将
旧服务器路径改成可在项目根目录运行的路径。`grouping.json` 是源文件的逐字节副本。
正式 YAML 的 rank 覆盖为 `group_0=[16,16,16,16]`、`group_1=[48,48,48,48]`，
优先于 JSON 中的 32/32。group_0 路由 semseg/human_parts，group_1 路由 normals/sal。

九组均使用其 `post_affinity_checkpoint.pth`，SHA256 为：

```text
b05959165782d239377e0ab144ff52f6426c3f0327c4851235ba75ff67e20281
```

加载沿用原有 shared-to-group rank 对齐规则，将 shared LoRA 权重对齐到 16/48
的 group 张量；九组使用同一映射，关闭模块的权重仍保留。

固定设置：448×448、50 个 prompt token、任务顺序
`semseg,normals,sal,human_parts`、task rank 全为 0、batch size=10、WORLD_SIZE=1、
seed=0、300 epochs、20 warmup epochs、每 5 epoch 验证/保存。
原始 LR：BASE=1e-3、WARMUP=1e-6、MIN=1e-5；启动时按原规则乘以 `10/512` 一次，
有效 BASE_LR=1.953125e-5。保留原 AMP、随机增强与非确定性训练设置，测试另行关闭随机性。
损失权重：semseg=1、normals=10、sal=5、human_parts=2。
正式训练用现有完整官方 train split，评估用官方 val split；Stage-1 meta split 仅用于
生成固定 grouping，不用于切换正式数据划分。patch embedding、LayerNorm、relative bias、
普通 PatchMerging reduction 和 decoder 保持基线训练策略；decoder downsampler 不在消融范围。

## 运行

在项目根目录、已配置 CUDA/NCCL 和依赖的环境运行。本次没有启动完整训练。

单组训练示例（替换 YAML 名称即可运行任意一组）：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py \
  --cfg configs/stage_ablation/swin_t/S3-P.yaml --disable_wandb
```

九组顺序运行（会启动正式训练，交付时未执行）：

```bash
for variant in baseline S1-P S2-P S3-P S4-P S1-G S2-G S3-G S4-G; do
  torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py \
    --cfg "configs/stage_ablation/swin_t/${variant}.yaml" --disable_wandb
done
```

PowerShell 对应循环：

```powershell
foreach ($variant in 'baseline','S1-P','S2-P','S3-P','S4-P','S1-G','S2-G','S3-G','S4-G') {
  torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py --cfg "configs/stage_ablation/swin_t/$variant.yaml" --disable_wandb
}
```

默认加载共同模型权重，optimizer/scheduler/scaler/epoch 使用新状态。
不要用 `--resume post_affinity_checkpoint.pth` 开启新实验。
迁移初始化文件可用 `--init-checkpoint /path/to/post_affinity_checkpoint.pth`，SHA256 仍须匹配。

同一实验续训与评估：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py \
  --cfg configs/stage_ablation/swin_t/S3-P.yaml \
  --resume /path/to/S3-P/ckpt_epoch_100.pth --disable_wandb

torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py \
  --cfg configs/stage_ablation/swin_t/S3-P.yaml \
  --resume /path/to/S3-P/ckpt_epoch_100.pth --eval --disable_wandb
```

续训检查 MODES、grouping 标识/实际 rank、初始化标识、训练协议、optimizer 参数名与顺序、
完整 optimizer/scheduler/scaler/epoch 状态。不兼容会明确报错。
评估同样检查 adapter 身份，但允许不同评估 batch size。
`--eval` 不带 `--resume` 可评估共同初始化在所选消融模式下的行为。

独立参数统计、FLOPs 和推理评估均复用训练冻结策略与 checkpoint 身份检查：

```bash
python run_mtlora_parameter_summary.py --cfg configs/stage_ablation/swin_t/S3-P.yaml --checkpoint /path/to/S3-P/ckpt_epoch_100.pth
python run_mtlora_flops.py --cfg configs/stage_ablation/swin_t/S3-P.yaml --checkpoint /path/to/S3-P/ckpt_epoch_100.pth
python run_mtlora_inference_benchmark.py --cfg configs/stage_ablation/swin_t/S3-P.yaml --checkpoint /path/to/S3-P/ckpt_epoch_100.pth
```

## 测试与解释

```bash
python -m unittest tests.test_mtlora_prompt tests.test_mtlora_inference_benchmark tests.test_unipora_training_alignment -v
python scripts/check_stage_ablation.py --device cpu
```

烟测使用九份真实配置、448×448 合成输入、四个 decoder 和共同初始化，验证前向、
逐任务反向、特征尺寸、关闭参数无梯度、上游梯度及输出精确重载。不会执行 optimizer.step。
报告：`output/stage_ablation_smoke.json`。小模型单元测试另外执行 optimizer.step，检查关闭参数
不更新、完整状态恢复，并覆盖 token 数量、路由、S1/S4、相邻 G、split-QKV 和扰动不变性。

日志 `total_parameters` 为全部存储参数，`stored_adapter_parameters` 为保留的 adapter 张量，
`enabled_adapter_parameters` 为前向实际启用的 adapter 参数，`trainable_parameters` 包括仍训练
的非 adapter 参数。九组均存储 34,871,162 个参数，其中 adapter 张量共 5,405,184 个，
因此关闭模块不降低存储量。

S3 有 6 个 block，其余 stage 各有 2 个；结果衡量整个 stage 的贡献，不能直接解释为
每个 block 的贡献或相同 adapter 参数预算下的比较。
本次验证使用 CPU/float32 与合成输入，未验证 GPU/AMP 数值、实际任务精度或完整收敛。
全量测试中的既有 Stage-1、旧源码断言、Windows 临时目录权限问题已与本次结果分开记录。

2026-10-10 验证记录（本地 PyTorch 2.5.1+cpu）：

- 针对性测试：34 项通过（包含九种模式的子测试）。
- 真实配置烟测：9/9 前向、反向、特征尺寸、checkpoint 重载通过。
- 最终身份/协议检查：9/9 完整尺寸前向及重载通过。
- 语法编译及 `git diff --check` 通过。
- 全量测试：93 项中 85 项通过，4 failure、4 error。相同 8 项问题在未修改 HEAD
  导出的基线副本上复现（83 项中 75 项通过）；没有新增失败。

日志：`output/stage_ablation_tests.log`、`output/stage_ablation_full_tests.log`、
`output/stage_ablation_baseline_tests.log`。真实权重烟测详细结果：
`output/stage_ablation_smoke.json`、`output/stage_ablation_final_reload.json`。

| 配置 | 启用 adapter 参数 | 可训练参数（含非 adapter） |
|---|---:|---:|
| baseline | 5,405,184 | 8,952,890 |
| S1-P | 5,208,576 | 8,756,282 |
| S2-P | 5,011,968 | 8,559,674 |
| S3-P | 3,045,888 | 6,593,594 |
| S4-P | 3,832,320 | 7,380,026 |
| S1-G | 5,366,784 | 8,914,490 |
| S2-G | 5,328,384 | 8,876,090 |
| S3-G | 4,944,384 | 8,492,090 |
| S4-G | 5,097,984 | 8,645,690 |
