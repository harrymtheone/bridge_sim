from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

import torch
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


def grid_pattern(cfg: GridPatternCfg, device: str) -> torch.Tensor:
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

    return ray_starts


@configclass
class GridPatternCfg(PatternBaseCfg):
    """Configuration for the grid pattern for ray-casting.

    Defines actions 2D grid of rays in the coordinates of the sensor.

    .. attention::
        The points are ordered based on the :attr:`ordering` attribute.

    """

    func = grid_pattern

    shape: tuple[int, int] = MISSING
    """Grid shape (x, y)."""

    size: tuple[float, float] = MISSING
    """Grid size (length, width) (in meters)."""

    ordering: Literal["xy", "yx"] = "xy"
    """Specifies the ordering of points in the generated grid. Defaults to ``"xy"``.

    Consider actions grid pattern with points at :math:`(x, y)` where :math:`x` and :math:`y` are the grid indices.
    The ordering of the points can be specified as "xy" or "yx". This determines the inner and outer loop order
    when iterating over the grid points.

    * If "xy" is selected, the points are ordered with inner loop over "x" and outer loop over "y".
    * If "yx" is selected, the points are ordered with inner loop over "y" and outer loop over "x".

    For example, the grid pattern points with :math:`X = (0, 1, 2)` and :math:`Y = (3, 4)`:

    * "xy" ordering: :math:`[(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (2, 4)]`
    * "yx" ordering: :math:`[(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]`
    """
