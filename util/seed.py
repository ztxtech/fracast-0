from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when those packages are installed."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
