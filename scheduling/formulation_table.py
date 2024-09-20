from typing import Callable

import numpy as np
import torch

SCHEDULER_FORMULATION_TABLE: dict[str, dict[str, Callable[[int | float | np.ndarray | torch.Tensor], int | float | np.ndarray | torch.Tensor]]] = {
    # ----- VP SDE -----
    "DDPM": {
        "scale": NotImplementedError,
        "scale_deriv": NotImplementedError,
        "sigma": NotImplementedError,
        "sigma_deriv": NotImplementedError,
        "nsr_inv": NotImplementedError,
    },
    "Ornstien-Uhlenbeck": {
        "scale": lambda t: torch.exp(-t) if isinstance(t, torch.Tensor) else np.exp(-t),
        "scale_deriv": lambda t: torch.exp(-t) if isinstance(t, torch.Tensor) else np.exp(-t),
        "sigma": lambda t: torch.sqrt(1.0 - torch.exp(-2.0 * t)) if isinstance(t, torch.Tensor) else np.sqrt(1.0 - np.exp(-2.0 * t)),
        "sigma_deriv": lambda t: torch.exp(-2.0 * t) / torch.sqrt(1.0 - torch.exp(-2.0 * t)) if isinstance(t, torch.Tensor) else np.exp(-2.0 * t) / np.sqrt(1.0 - np.exp(-2.0 * t)),
        "nsr_inv": lambda nsr: torch.log(1.0 + nsr ** 2) / 2.0 if isinstance(nsr, torch.Tensor) else np.log(1.0 + nsr ** 2) / 2.0,
    },
    # ----- VE SDE -----
    "NCSN": {
        "scale": NotImplementedError,
        "scale_deriv": NotImplementedError,
        "sigma": NotImplementedError,
        "sigma_deriv": NotImplementedError,
        "nsr_inv": NotImplementedError,
    },
    "EDM": {
        "scale": lambda t: torch.ones_like(t) if isinstance(t, torch.Tensor) else 1.0,
        "scale_deriv": lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else 0.0,
        "sigma": lambda t: t,
        "sigma_deriv": lambda t: torch.ones_like(t) if isinstance(t, torch.Tensor) else 1.0,
        "nsr_inv": lambda nsr: nsr,
    },
    # ----- Flow -----
    "Rectified Flow": {
        "scale": lambda t: 1.0 - t,
        "scale_deriv": lambda t: -torch.ones_like(t) if isinstance(t, torch.Tensor) else -1.0,
        "sigma": lambda t: t,
        "sigma_deriv": lambda t: torch.ones_like(t) if isinstance(t, torch.Tensor) else 1.0,
        "nsr_inv": lambda nsr: nsr / (1.0 + nsr),
    },
}
