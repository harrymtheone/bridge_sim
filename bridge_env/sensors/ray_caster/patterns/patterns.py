from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from . import patterns_cfg


def grid_pattern(cfg: patterns_cfg.GridPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # check valid arguments
    if cfg.ordering not in ["xy", "yx"]:
        raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{cfg.ordering}'.")

    # resolve mesh grid indexing (note: torch meshgrid is different from numpy meshgrid)
    # check: https://github.com/pytorch/pytorch/issues/15301
    indexing = cfg.ordering if cfg.ordering == "xy" else "ij"
    # define grid pattern
    x = torch.linspace(start=-cfg.size[0] / 2, end=cfg.size[0] / 2 + 1.0e-9, steps=cfg.shape[0], device=device)
    y = torch.linspace(start=-cfg.size[1] / 2, end=cfg.size[1] / 2 + 1.0e-9, steps=cfg.shape[1], device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)

    # store into ray starts
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    # define ray-cast directions
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(cfg.direction), device=device)

    return ray_starts, ray_directions
