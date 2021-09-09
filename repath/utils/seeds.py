import torch
import random
import numpy as np



def set_seed(global_seed):
    torch.manual_seed(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    # np random number generators???
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

    torch.cuda.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    set_seed(worker_seed)