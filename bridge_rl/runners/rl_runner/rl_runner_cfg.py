from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from bridge_rl.algorithms.ppo import PPOCfg


@configclass
class RLRunnerCfg:
    algorithm_cfg: PPOCfg = MISSING

    max_iterations: int = MISSING

    num_steps_per_env: int = 24

    save_interval: int = 100

    logger_backend: str = "tensorboard"
