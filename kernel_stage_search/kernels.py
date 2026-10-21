"""Fixed Gaussian/equality kernels and matrix Renyi-2 information."""

import math

import torch

from .protocol import TASKS


def squared_distances(features):
    features = torch.as_tensor(features).float()
    if features.ndim != 2 or features.shape[0] < 2 or not torch.isfinite(features).all():
        raise ValueError("Expected finite [n,d] features with n>=2")
    norms = features.square().sum(dim=1)
    distances = (norms[:, None] + norms[None, :] - 2*(features @ features.T)).clamp_min_(0)
    if not torch.isfinite(distances).all():
        raise ValueError("Non-finite pairwise distances")
    distances = (distances+distances.T)*0.5
    distances.fill_diagonal_(0)
    return distances


def median_positive_distance(features):
    distances = squared_distances(features)
    positive = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1) & (distances > 0)]
    if not positive.numel():
        raise ValueError("No positive calibration distances; cannot set kernel width")
    # numpy-style median (average the middle pair), rather than torch.median's lower value.
    width = torch.quantile(positive.double(), 0.5).item()
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Invalid calibration bandwidth")
    return width


def gaussian_kernel(features, sigma_squared):
    if not math.isfinite(sigma_squared) or sigma_squared <= 0:
        raise ValueError("A fixed positive sigma squared is required")
    kernel = torch.exp(squared_distances(features) * (-0.5/sigma_squared))
    kernel = (kernel+kernel.T)*0.5
    kernel.fill_diagonal_(1.)
    return kernel.float()


def equality_kernel(labels):
    labels = torch.as_tensor(labels)
    if labels.ndim != 1 or not torch.isfinite(labels).all():
        raise ValueError("Equality kernel requires finite scalar categories")
    return (labels[:, None] == labels[None, :]).float()


def validate_kernel(kernel):
    kernel = torch.as_tensor(kernel)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] < 2 or not torch.isfinite(kernel).all():
        raise ValueError("Invalid kernel shape or finite values")
    if (not torch.allclose(kernel, kernel.T, atol=1e-6, rtol=0)
            or not torch.allclose(kernel.diag(), torch.ones_like(kernel.diag()), atol=1e-6, rtol=0)
            or (kernel < -1e-6).any() or (kernel > 1+1e-6).any()):
        raise ValueError("Invalid kernel symmetry/diagonal/range")
    return kernel.double()


def entropy2(matrix):
    purity = matrix.double().square().sum()
    if not torch.isfinite(purity) or purity <= 0:
        raise ValueError("Invalid kernel purity")
    return -torch.log2(purity)


def normalized_information(kf, ky, entropy_min=1e-8):
    a, b = validate_kernel(kf), validate_kernel(ky)
    if a.shape != b.shape:
        raise ValueError("Feature and label kernel sizes differ")
    count = a.shape[0]
    a, b = a/count, b/count
    sy = entropy2(b)
    if sy <= entropy_min:
        raise ValueError("Degenerate label batch")
    joint = a*b
    joint = joint/torch.trace(joint)
    sx, sxy = entropy2(a), entropy2(joint)
    i2 = sx+sy-sxy
    result = {"sx": sx.item(), "sy": sy.item(), "sxy": sxy.item(), "i2": i2.item(),
              "ratio": (i2/sy).item(), "n": count}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite kernel information")
    return result


def kernel_diagnostics(kernel, features=None):
    kernel = validate_kernel(kernel)
    off = kernel[~torch.eye(len(kernel), dtype=torch.bool, device=kernel.device)]
    result = {"off_diagonal_mean": off.mean().item(), "off_diagonal_variance": off.var(unbiased=False).item(),
              "near_zero_fraction": (off < 1e-6).double().mean().item(),
              "near_one_fraction": (off > 1-1e-6).double().mean().item()}
    result["near_constant"] = result["near_one_fraction"] > 0.99
    result["near_identity"] = result["near_zero_fraction"] > 0.99
    if features is not None:
        result["feature_variance"] = features.double().var(dim=0, unbiased=False).mean().item()
    return result


def label_kernels(batch, sigma_normals, label_mode="true_labels"):
    if label_mode not in ("true_labels", "shuffled"):
        raise ValueError("Invalid label mode")
    result = {}
    for task in TASKS:
        entry = batch["tasks"][task]
        labels = torch.as_tensor(entry["labels"])
        kernel = gaussian_kernel(labels, sigma_normals) if task == "normals" else equality_kernel(labels)
        if label_mode == "shuffled":
            permutation = torch.tensor(entry["permutation"], dtype=torch.long)
            kernel = kernel[permutation][:, permutation]
        result[task] = kernel
    return result


def score_batch(feature_kernels, target_kernels, entropy_min=1e-8):
    if set(feature_kernels) != set(TASKS) or set(target_kernels) != set(TASKS):
        raise ValueError("Four fixed tasks required for equally weighted J")
    tasks = {task: normalized_information(feature_kernels[task], target_kernels[task], entropy_min) for task in TASKS}
    return {"tasks": tasks, "J": sum(tasks[task]["ratio"] for task in TASKS)/len(TASKS)}
