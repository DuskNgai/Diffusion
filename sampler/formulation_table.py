from typing import Callable

import numpy as np
import torch

FunctionType = Callable[[int | float | np.ndarray | torch.Tensor], int | float | np.ndarray | torch.Tensor]

SAMPLER_FORMULATION_TABLE: dict[str, dict[str, FunctionType]] = {
    # ----- VP SDE -----
    "DDPM": {
        "scale_fn": NotImplementedError,
        "scale_deriv_fn": NotImplementedError,
        "sigma_fn": NotImplementedError,
        "sigma_deriv_fn": NotImplementedError,
        "nsr_inv_fn": NotImplementedError,
    },
    "Ornstein-Uhlenbeck": {
        "scale_fn": lambda t: torch.exp(-t) if isinstance(t, torch.Tensor) else np.exp(-t),
        "scale_deriv_fn": lambda t: torch.exp(-t) if isinstance(t, torch.Tensor) else np.exp(-t),
        "sigma_fn": lambda t: torch.sqrt(1.0 - torch.exp(-2.0 * t)) if isinstance(t, torch.Tensor) else np.sqrt(1.0 - np.exp(-2.0 * t)),
        "sigma_deriv_fn": lambda t: torch.exp(-2.0 * t) / torch.sqrt(1.0 - torch.exp(-2.0 * t)) if isinstance(t, torch.Tensor) else np.exp(-2.0 * t) / np.sqrt(1.0 - np.exp(-2.0 * t)),
        "nsr_inv_fn": lambda nsr: torch.log(1.0 + nsr ** 2) / 2.0 if isinstance(nsr, torch.Tensor) else np.log(1.0 + nsr ** 2) / 2.0,
    },
    # ----- VE SDE -----
    "NCSN": {
        "scale_fn": NotImplementedError,
        "scale_deriv_fn": NotImplementedError,
        "sigma_fn": NotImplementedError,
        "sigma_deriv_fn": NotImplementedError,
        "nsr_inv_fn": NotImplementedError,
    },
    "EDM": {
        "scale_fn": lambda t: torch.ones_like(t) if isinstance(t, torch.Tensor) else 1.0,
        "scale_deriv_fn": lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else 0.0,
        "sigma_fn": lambda t: t,
        "sigma_deriv_fn": lambda t: torch.ones_like(t) if isinstance(t, torch.Tensor) else 1.0,
        "nsr_inv_fn": lambda nsr: nsr,
    },
    # ----- Flow -----
    "Rectified Flow": {
        "scale_fn": lambda t: 1.0 - t,
        "scale_deriv_fn": lambda t: -torch.ones_like(t) if isinstance(t, torch.Tensor) else -1.0,
        "sigma_fn": lambda t: t,
        "sigma_deriv_fn": lambda t: torch.ones_like(t) if isinstance(t, torch.Tensor) else 1.0,
        "nsr_inv_fn": lambda nsr: nsr / (1.0 + nsr),
    },
}
