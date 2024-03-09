import torch.nn as nn
import math
import torch
import random
import numpy as np


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataset_data_path():
    from pathlib import Path
    path = str( Path(__file__).parent.parent.parent) + '/GRAPH_DATA'
    return path

def setup_seed(seed):
    print('\n\n==> Setting seed = ', seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

def init_model_para(model):
    # return
    # setup_seed(1234)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
        elif isinstance(layer, nn.Conv2d):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            nn.init.normal_(layer.weight, 0., math.sqrt(2. / n))

def get_device():
    if torch.cuda.is_available():
        print('\n==> using cuda')
        device = torch.device("cuda:0")
    else:
        print('\n==> using CPU')
        device = torch.device("cpu")
    return device