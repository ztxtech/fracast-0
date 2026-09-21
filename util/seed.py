from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """固定 Python 随机种子；若已安装 numpy/torch 则一并固定。"""
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
