"""Small float64 categorical GP. Only completed, actual J observations are used."""

import itertools
import math

import numpy as np

CANDIDATES = tuple("".join(value) for value in itertools.product(("B", "P", "G"), repeat=4))
INITIAL = ("BBBB", "PPPP", "GGGG")


def categorical_kernel(x, y):
    x = np.asarray([list(item) for item in x])
    y = np.asarray([list(item) for item in y])
    distance = (x[:, None, :] != y[None, :, :]).sum(axis=-1)
    return np.exp(-distance.astype(np.float64))


def validate_observations(observations, budget):
    if not 3 <= budget <= 81:
        raise ValueError("Search budget must be in [3,81]")
    modes = [item["modes"] for item in observations]
    if len(modes) != len(set(modes)) or any(mode not in CANDIDATES for mode in modes):
        raise ValueError("Duplicate/invalid observations")
    if len(modes) > budget or any(not math.isfinite(item["J"]) for item in observations):
        raise ValueError("Invalid J/budget")
    if modes[:min(3, len(modes))] != list(INITIAL[:min(3, len(modes))]):
        raise ValueError("Initial observation sequence mismatch")


def posterior(observations, candidates, noise_variance=1e-6):
    values = np.asarray([item["J"] for item in observations], dtype=np.float64)
    scale = values.std()
    y = np.zeros_like(values) if scale < 1e-12 else (values-values.mean())/scale
    observed = [item["modes"] for item in observations]
    kernel = categorical_kernel(observed, observed)
    cross = categorical_kernel(observed, candidates)
    jitter = 1e-10
    while True:
        try:
            factor = np.linalg.cholesky(kernel+(noise_variance+jitter)*np.eye(len(observed)))
            break
        except np.linalg.LinAlgError:
            if jitter >= 1e-4:
                raise ValueError("GP Cholesky failed through jitter=1e-4")
            jitter = min(jitter*10, 1e-4)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
    mean = cross.T @ alpha
    v = np.linalg.solve(factor, cross)
    variance = 1.-np.sum(v*v, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.min(variance) < -1e-8:
        raise ValueError("Invalid GP posterior variance")
    return mean, np.sqrt(np.maximum(variance, 0)), jitter


def propose(observations, budget=12, noise_variance=1e-6, beta=2.):
    validate_observations(observations, budget)
    if noise_variance != 1e-6 or beta != 2.:
        raise ValueError("The fixed GP protocol requires noise=1e-6 and UCB beta=2")
    if len(observations) == budget:
        return None
    if len(observations) < len(INITIAL):
        return {"modes": INITIAL[len(observations)], "mu": None, "std": None, "acquisition": None,
                "jitter": None, "kind": "initial"}
    observed = {item["modes"] for item in observations}
    candidates = [mode for mode in CANDIDATES if mode not in observed]
    mean, std, jitter = posterior(observations, candidates, noise_variance)
    acquisition = mean+beta*std
    index = int(np.argmax(acquisition))
    return {"modes": candidates[index], "mu": float(mean[index]), "std": float(std[index]),
            "acquisition": float(acquisition[index]), "jitter": jitter, "kind": "ucb"}


def select_winner(observations):
    if not observations or any(not math.isfinite(item["J"]) for item in observations):
        raise ValueError("Winner requires finite actual observations")
    return max(observations, key=lambda item: item["J"])
