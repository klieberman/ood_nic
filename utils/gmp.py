import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# ------ GMP Layers ----- #
class GMPConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curr_sparsity_rate = nn.Parameter(torch.ones(1), requires_grad=False)

    def set_curr_sparsity_rate(self, curr_sparsity_rate):
        self.curr_sparsity_rate = nn.Parameter(curr_sparsity_rate.clone().detach(), requires_grad=False)

    def forward(self, x):
        w = GMP_GetSubnet(self.weight, self.curr_sparsity_rate)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


class GlobalGMPConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_threshold = nn.Parameter(torch.zeros(1), requires_grad=False)

    def set_prune_threshold(self, prune_threshold):
        self.prune_threshold = nn.Parameter(prune_threshold, requires_grad=False)

    def forward(self, x):
        w = GMP_GetGlobalSubnet(self.weight, self.prune_threshold)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


class GMPConvTranspose(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curr_sparsity_rate = nn.Parameter(torch.ones(1), requires_grad=False)

    def set_curr_sparsity_rate(self, curr_sparsity_rate):
        self.curr_sparsity_rate = nn.Parameter(curr_sparsity_rate.clone().detach(), requires_grad=False)

    def forward(self, x):
        w = GMP_GetSubnet(self.weight, self.curr_sparsity_rate)
        x = F.conv_transpose2d(
            x, w, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation,
        )

        return x


class GlobalGMPConvTranspose(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_threshold = nn.Parameter(torch.zeros(1), requires_grad=False)

    def set_prune_threshold(self, prune_threshold):
        self.prune_threshold = nn.Parameter(prune_threshold.clone().detach(), requires_grad=False)

    def forward(self, x):
        w = GMP_GetGlobalSubnet(self.weight, self.prune_threshold)
        x = F.conv_transpose2d(
            x, w, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation,
        )

        return x


class GMPLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curr_sparsity_rate = nn.Parameter(torch.ones(1), requires_grad=False)

    def set_curr_sparsity_rate(self, curr_sparsity_rate):
        self.curr_sparsity_rate = nn.Parameter(curr_sparsity_rate.clone().detach(), requires_grad=False)

    def forward(self, x):
        w = GMP_GetSubnet(self.weight, self.curr_sparsity_rate)

        x = F.linear(
            x, w, self.bias
        )

        return x


class GlobalGMPLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_threshold = nn.Parameter(torch.zeros(1), requires_grad=False)

    def set_prune_threshold(self, prune_threshold):
        self.prune_threshold = nn.Parameter(prune_threshold.clone().detach(), requires_grad=False)

    def forward(self, x):
        w = GMP_GetGlobalSubnet(self.weight, self.prune_threshold)
        x = F.linear(
            x, w, self.bias
        )

        return x


# Functions for GMP (do not require scores)
def GMP_GetSubnet(weight, curr_sparsity_rate):
    output = weight.clone()
    _, idx = weight.flatten().abs().sort()
    p = int(curr_sparsity_rate * weight.numel())
    # flat_oup and output access the same memory.
    flat_oup = output.flatten()
    flat_oup[idx[:p]] = 0
    return output


def GMP_GetGlobalSubnet(weight, prune_threshold):
    output = weight.clone()
    cond = torch.abs(output) > prune_threshold
    cond = cond.to(output.device)
    z = torch.as_tensor(0.0, device = output.device)
    # Only keep weights that are above threshold (in absolute value)
    output = torch.where(cond, output, z)
    return output


def get_curr_sparsity_gmp(curr_epoch, prune_epochs, final_sparsity, initial_sparsity=0.):
    if curr_epoch < prune_epochs[0]:
        return initial_sparsity
    elif curr_epoch < prune_epochs[1]:
        total_prune_epochs = prune_epochs[1] - prune_epochs[0] + 1
        prune_decay = (1 - ((curr_epoch - prune_epochs[0]) / total_prune_epochs))**3
        return final_sparsity + ((initial_sparsity - final_sparsity) * prune_decay)
    else:
        return final_sparsity


# --- Setting Prune Thresholds --- #

def global_set_prune_thresholds(model, curr_sparsity, device):
    # Get all weights, sort them, and find correct threshold to prune at
    p_list = []
    for n, m in model.named_modules():
        if hasattr(m, 'set_prune_threshold'):
            p_list.append(m.weight.clone().abs().flatten())
    z = torch.cat(p_list)
    z,_ = z.sort()
    p_idx = int(curr_sparsity * z.numel())
    prune_threshold = z[p_idx]
    prune_threshold = torch.tensor([prune_threshold]).to(device)
    # Loop over all model parameters to update prune_threshold
    for n, m in model.named_modules():
      if hasattr(m,'set_prune_threshold'):
        m.set_prune_threshold(prune_threshold)
        # print(f'Set {n} to prune_threshold {prune_threshold}.')

    return


def layerwise_set_sparsity_rates(model, curr_sparsity, device):
    curr_sparsity = torch.tensor([curr_sparsity]).to(device)
    for n, m in model.named_modules():
        if hasattr(m, 'set_curr_sparsity_rate'):
            m.set_curr_sparsity_rate(curr_sparsity)
            # print(f'Set {n} to current sparsity {curr_sparsity}.')
    return


def check_sparsity(model, desired_sparsity, layerwise=False):
    all_weights, nonzero_weights = 0, 0
    for n, m in model.named_modules():
        if not layerwise and hasattr(m, 'prune_threshold'):
            w = GMP_GetGlobalSubnet(m.weight, m.prune_threshold)
            all_weights += w.numel()
            nonzero_weights += torch.count_nonzero(w.detach())
        elif layerwise and hasattr(m, 'curr_sparsity_rate'):
            w = GMP_GetSubnet(m.weight, m.curr_sparsity_rate)
            all_weights += w.numel()
            nonzero_weights += torch.count_nonzero(w.detach())

    actual_sparsity = 1 - (nonzero_weights/all_weights)
    assert (desired_sparsity - actual_sparsity).abs() < 0.01, f'Desired sparsity ({desired_sparsity}) different than actual sparsity ({actual_sparsity})'
    
    return actual_sparsity


