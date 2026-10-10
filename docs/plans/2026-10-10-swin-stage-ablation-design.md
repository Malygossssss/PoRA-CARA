# Swin-T stage ablation

The four backbone layers use B (prompt + group-LoRA), P (prompt), and G
(group-LoRA). These stage indices are unrelated to CARA's Stage-1 search.
Keep every module, parameter shape and checkpoint key. Build and initialize
all nine models in the same order, then change runtime switches and requires_grad.

`stage_ablation.py` validates fixed, resolved grouping and zero task ranks,
applies the existing LoRA switch API, and owns parameter exclusions. The shared
training setup runs before optimizer construction and validation, in both main
and standalone tools. Prompt marking reapplies exclusions. Existing patch embed,
norms, bias, ordinary merging reductions and decoder training policies remain.

P disables every shared/group branch and trainable scale in its layer without
cutting upstream gradients. G strips the incoming prompt prefix before every
block and forwards active_task through attention/MLP. S1 omits the initial prompt;
later enabled layers insert their own deep prompts. Explicit batch/window reshape
dimensions handle zero-token prefixes. Existing bias/mask and PatchMerging paths
support empty prompts. Decoder image-feature dimensions remain unchanged.

New runs load only common model weights via INIT_CHECKPOINT, with SHA256 recorded.
Resumes check modes, grouping/ranks, initialization identity, training protocol,
ordered optimizer parameter names and complete training state. Evaluation checks
adapter identity while permitting a different evaluation batch size.

All configs inherit a portable formal resolved config from run_20260907_031808.
Its manual rank override is 16/48, taking precedence over JSON's 32/32. There is
no search or budget redistribution. Initialization is its post-affinity checkpoint.
The fixed formal protocol uses existing PASCAL train/val loaders, four tasks,
batch size 10, one process, seed 0, 300 epochs and existing 2lr settings.

Tests cover all-B equality, parameter perturbations, attention token counts,
routing, exclusions/updates, upstream gradients, transitions, S1/S4 boundaries,
nine optimizer/checkpoint roundtrips, global-to-group initialization, main
evaluation/pre-training validation and standalone parameter reporting. Full-size
synthetic smoke checks use all nine configs, real weights and four decoders.
