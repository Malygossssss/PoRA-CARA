# Active-Task Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate discarded all-task AG-MTLoRA branch computation inside each prompt-conditioned task backbone forward.

**Architecture:** Thread an optional active-task selector through the prompt backbone stack. AG-MTLoRA linear layers use it to return one selected task branch, while calls without a selector preserve the existing all-task behavior.

**Tech Stack:** Python, PyTorch, unittest, YACS configuration

---

### Task 1: Specify active-task linear routing

**Files:**
- Modify: `tests/test_mtlora_prompt.py`
- Modify: `models/lora.py`

**Steps:**

1. Add a failing test proving that selected routing returns only the requested task and rejects an unknown task.
2. Run `python -m unittest tests.test_mtlora_prompt -v` and confirm the new test fails.
3. Add optional `active_task` handling to `MTLoRALinear` and forward it through `MTLoRAQKV`.
4. Re-run the focused test and confirm it passes.

### Task 2: Thread the selector through prompt attention and blocks

**Files:**
- Modify: `models/swin_transformer_mtlora.py`
- Modify: `models/swin_transformer_vpt.py`
- Modify: `tests/test_mtlora_prompt.py`

**Steps:**

1. Add tests comparing selected-task output against the corresponding legacy all-task output and counting attention branches.
2. Pass `active_task` through `Mlp`, prompted blocks, prompted attention, QKV, and projection calls.
3. Iterate over actual returned task keys so omitted branches are never accessed.
4. Run `python -m unittest tests.test_mtlora_prompt -v` and confirm all prompt tests pass.

### Task 3: Regression verification

**Files:**
- Test: `tests/test_mtlora_prompt.py`
- Test: `tests/test_ag_mtlora_stage1.py`
- Test: `tests/test_unipora_training_alignment.py`

**Steps:**

1. Run the prompt and Stage-1 test suites.
2. Run training-alignment and configuration tests.
3. Compile all modified Python modules with `python -m py_compile`.
4. Inspect `git diff --check`, the final diff, and branch status.

