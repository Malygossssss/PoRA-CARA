# Stage-1 Independent Multi-Process Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 Stage-1 增加 UniPoRA-B 风格双卡独立搜索启动与 rank-0 canonical artifact 规则。

**Architecture:** 新增无 PyTorch 依赖的进程上下文、目录和 manifest 模块；Stage-1 launcher 初始化/销毁 NCCL，并为每个 rank 配置独立输出。原 `run_stage1_pipeline` 算法保持不变。

**Tech Stack:** Python 3.8、PyTorch distributed/NCCL、torchrun、unittest。

---

### Task 1: Add failing multi-process contract tests

**Files:**
- Create: `tests/test_stage1_multiprocess.py`

**Steps:**
1. 测试 torchrun 环境解析和非法 rank/world-size 校验。
2. 测试单卡目录兼容、双卡 rank 目录隔离和 resume 校验。
3. 测试 canonical manifest 固定采用 rank 0。
4. 源码断言 Stage-1 launcher 无 DDP/DistributedSampler。

### Task 2: Add pure runtime/path helpers

**Files:**
- Create: `ag_mtlora/stage1_multiprocess.py`

**Steps:**
1. 实现 `Stage1ProcessContext` 和环境解析。
2. 实现新运行/恢复目录解析。
3. 实现 per-rank artifact 与 canonical manifest 构造、保存。
4. 运行无依赖测试并确认通过。

### Task 3: Integrate torchrun into Stage-1 launcher

**Files:**
- Modify: `scripts/ag_mtlora_stage1_prepare.py`
- Modify: `ag_mtlora/stage1.py`

**Steps:**
1. 同时接受 `--local_rank` 与 `--local-rank`。
2. 双卡初始化 NCCL、绑定设备并广播 run root。
3. 配置 rank 独立 artifact 路径和 `SEED + rank`。
4. 写 per-rank artifact manifest；rank 0 写 canonical manifest。
5. 在 `finally` 中销毁 process group。

### Task 4: Document launch and resume workflow

**Files:**
- Modify: `README_UNIPORA_CARA.md`
- Modify: `README_AG_MTLORA_STAGE1.md`

**Steps:**
1. 增加双卡 Stage-1 torchrun 命令。
2. 说明 rank-0 canonical 产物及正式训练路径。
3. 说明双卡 resume root 规则。

### Task 5: Verify

**Files:**
- Test: `tests/test_stage1_multiprocess.py`
- Test: `tests/test_unipora_training_alignment.py`
- Test: existing Stage-1 and prompt tests when PyTorch is available.

**Steps:**
1. 运行新增无依赖测试。
2. 运行全项目 AST 和 `git diff --check`。
3. 尝试完整测试并如实记录环境阻断。
4. 审查 Stage-1 源码，确认不存在 DDP、数据分片或跨 rank 结果平均。

