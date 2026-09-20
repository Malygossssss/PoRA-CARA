# UniPoRA Training Alignment Design

## Goal

以 `D:\05_Experiments\UniPoRA` 当前实现为唯一训练基准，使 PoRA-CARA 的正式训练控制流、双卡启动语义、数据读取、优化、调度、恢复、验证和日志行为与 UniPoRA 一致；只保留 CARA/AG-MTLoRA 正常运行不可缺少的差异。

## Architecture

正式训练入口继续使用 PoRA-CARA 的模型、配置与 checkpoint 扩展能力，但训练控制器以 UniPoRA 为准。`main.py` 恢复 UniPoRA 的 Constrained MTL、conflict-ratio、梯度累积和 W&B 流程，并移除 PoRA-CARA 后加的 DDP 包装与 `no_sync()`。MTL 数据加载器恢复普通 `shuffle=True` 训练加载和完整验证集加载；即使 `WORLD_SIZE=2`，两个进程也各自遍历完整数据集、独立更新模型，只有 rank 0 checkpoint 被保留。

CARA 必需差异限定为：`mark_prompt_as_trainable()`、AG-MTLoRA/group-shared LoRA 配置、prompt-conditioned forward、Stage-1 搜索脚本，以及把 Stage-1 global-shared checkpoint 扩展到 group-shared bank 的加载逻辑。Stage-1 独立入口不改。

## Validation

回归测试固定以下契约：MTL 双进程构建不产生 `DistributedSampler`；正式训练入口不构造 `DistributedDataParallel` 或 `no_sync()`；UniPoRA 的受约束训练配置和控制器可用；prompt 参数在正式训练中仍被解冻。现有 Stage-1、prompt、edge 和数据准备测试必须继续通过。README 明确说明双卡是两个独立训练进程，而不是同步 DDP。

