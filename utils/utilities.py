import torch
import os
import os.path as osp

def get_device():
    cuda_available = torch.cuda.is_available()
    print("Cuda available? {}.".format(cuda_available))
    if cuda_available:
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)

def makedirs_if_needed(path):
    if not osp.exists(path):
        os.makedirs(path)
    return path