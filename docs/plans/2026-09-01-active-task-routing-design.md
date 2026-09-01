# Active-Task Routing Design

## Problem

Prompt-enabled PoRA-CARA already invokes the backbone once per task. Inside each of those invocations, AG-MTLoRA currently constructs group-routed LoRA outputs for every configured task at each linear layer, runs attention and MLP work for every returned branch, and then keeps only the branch belonging to the outer task. This preserves the desired result but performs substantial discarded computation.

## Design

Add an optional `active_task` argument along the prompt-aware forward path. `PromptedBasicLayer` knows the task selected by the outer multi-task model and passes it through `PromptedSwinTransformerBlock`, `PromptedWindowAttention`, `Mlp`, `MTLoRAQKV`, and `MTLoRALinear`. When AG-MTLoRA is enabled and `active_task` is provided, `MTLoRALinear` validates the task and computes only that task's group-shared and task-specific LoRA branch. Containers iterate over the returned branch keys instead of the complete configured task list.

The argument remains optional. Non-prompted AG-MTLoRA calls omit it and retain the existing all-task stream behavior needed to produce all decoder features from one backbone invocation. Non-AG MTLoRA behavior is unchanged. Checkpoint keys, parameter shapes, optimizer state, grouping files, losses, and public model outputs remain unchanged.

## Verification

Unit tests will compare the selected active-task attention output with the same task branch from the legacy all-task computation while dropout is disabled. Instrumentation will verify that active routing executes one task attention branch rather than every task branch. Existing prompt, Stage-1, training-alignment, and syntax tests will cover compatibility. A focused inference benchmark can subsequently measure GPU speed, but performance improvement is not asserted in a timing-sensitive unit test.

