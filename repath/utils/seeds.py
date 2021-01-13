import torch
import random
import numpy as np



def set_seed(global_seed):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)

