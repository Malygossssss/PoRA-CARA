# Stage-1 Independent Multi-Process Design

## Goal

让 AG-MTLoRA Stage-1 可以通过与正式训练相同的 `torchrun --nproc_per_node=2` 入口启动，并严格延续已选定的 UniPoRA-B 语义：每个 rank 使用完整数据、独立模型、独立梯度和独立搜索结果，最终采用 rank 0 产物。

## Architecture

Stage-1 launcher 从 `RANK`、`WORLD_SIZE` 和 `LOCAL_RANK` 建立进程上下文。双卡时初始化 NCCL process group 并绑定当前 CUDA 设备，但不创建 DDP、不切分数据，也不进行梯度或 affinity 平均。rank 0 创建统一 run root 并把路径广播给其他 rank；每个 rank 在 `rank_<n>/` 下运行原有完整 pipeline，使用 `SEED + rank`，从而避免 artifact、日志和 checkpoint 互相覆盖。

单卡继续使用原目录结构。双卡 run root 下写 `stage1_multi_process_manifest.json`，声明 `mode=unipora_independent_processes`、canonical rank 0、各 rank 目录、各 rank artifact manifest，以及正式训练应使用的 rank-0 resolved config 和 post-affinity checkpoint。`--resume-stage1-dir` 在双卡模式下指向共享 run root，各 rank 只恢复自己的子目录；缺失 rank 目录时立即报错。

## Error Handling and Validation

launcher 在 `finally` 中销毁 process group，不设置结束 barrier。每个 rank 完成后先写自己的 `stage1_artifacts.json`；rank 0 再写 canonical root manifest。纯路径、进程上下文和 manifest 逻辑放入不依赖 PyTorch 的模块，以便在当前本地环境测试。源码契约还会断言 Stage-1 不包含 DDP 或 DistributedSampler。

