"""A reversible full-stage switch; never modifies trainability or tensors."""

from contextlib import contextmanager
import copy

from stage_ablation import configure_stage_ablation


@contextmanager
def temporary_stage_modes(model, config, modes):
    modes = list(modes)
    if len(modes) != 4 or any(mode not in "BPG" for mode in modes):
        raise ValueError("Expected four B/P/G modes")
    attributes = ("stage_ablation_modes", "stage_prompt_enabled", "shared_lora_enabled")
    snapshots = [(module, {name: copy.deepcopy(module.__dict__[name]) for name in attributes if name in module.__dict__})
                 for module in model.modules()]
    candidate = config.clone()
    candidate.defrost()
    candidate.MODEL.STAGE_ABLATION.MODES = modes
    candidate.freeze()
    try:
        configure_stage_ablation(model, candidate)
        yield model
    finally:
        # Assign each child flag directly; calling the split-QKV parent setter here
        # would overwrite a child's previously distinct flag.
        for module, state in snapshots:
            for name in attributes:
                if name in state:
                    setattr(module, name, state[name])
                elif name in module.__dict__:
                    delattr(module, name)
