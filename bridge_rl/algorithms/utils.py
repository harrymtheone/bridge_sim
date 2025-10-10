from __future__ import annotations

import torch
from torch import nn


def make_linear_layers(*shape, activation_func=None, output_activation=True):
    if activation_func is None:
        raise ValueError('activation_func cannot be None!')

    layers = nn.Sequential()

    for l1, l2 in zip(shape[:-1], shape[1:]):
        layers.append(nn.Linear(l1, l2))
        layers.append(activation_func)

    if not output_activation:
        layers.pop(-1)

    return layers


def recurrent_wrapper(func: callable, tensor: torch.Tensor, ):
    n_seq = tensor.size(0)
    return func(tensor.flatten(0, 1)).unflatten(0, (n_seq, -1))


def masked_MSE(input_, target, mask):
    return ((input_ - target) * mask).square().sum() / (input_.numel() / mask.numel() * mask.sum())


def masked_L1(input_, target, mask):
    return ((input_ - target) * mask).abs().sum() / (input_.numel() / mask.numel() * mask.sum())


def masked_mean(input_, mask):
    return (input_ * mask).sum() / (input_.numel() / mask.numel() * mask.sum())
