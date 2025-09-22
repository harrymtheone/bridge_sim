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


def recurrent_wrapper(func: callable,
                      obs: torch.Tensor | dict[str, torch.Tensor],
                      **kwargs):
    if isinstance(obs, torch.Tensor):
        seq_batch = obs.shape[:2]
        rtn = func(obs.flatten(0, 1), **kwargs)
    elif isinstance(obs, dict):
        seq_batch = next(iter(obs.values())).shape[:2]
        rtn = func({k: v.flatten(0, 1) for k, v in obs.items()}, **kwargs)
    else:
        raise TypeError('obs must be actions torch.Tensor or actions dict!')

    if type(rtn) is tuple:
        return [r.unflatten(0, seq_batch) for r in rtn]
    else:
        return rtn.unflatten(0, seq_batch)


