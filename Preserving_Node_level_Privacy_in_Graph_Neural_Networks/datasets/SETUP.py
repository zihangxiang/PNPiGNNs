"""Global setup: data location, seeding and device selection."""
import random
from pathlib import Path

import numpy as np
import torch


def get_dataset_data_path():
    """Root for raw datasets and all caches: ``GRAPH_DATA/`` at the top of the repository."""
    return Path(__file__).parent.parent.parent / 'GRAPH_DATA'


def setup_seed(seed):
    print('\n\n==> Setting seed = ', seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        print('\n==> using cuda')
        return torch.device("cuda:0")
    print('\n==> using CPU')
    return torch.device("cpu")
