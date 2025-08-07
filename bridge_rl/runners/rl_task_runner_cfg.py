from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from bridge_rl.runners.rl_task_runner import RLRunner

if TYPE_CHECKING:
    from bridge_rl.algorithms.ppo import PPOCfg


@configclass
class RLTaskCfg:
    env_cfg: ManagerBasedRLEnvCfg = MISSING

    algorithm_cfg: PPOCfg = MISSING

    max_iterations: int = MISSING

    num_steps_per_env: int = 24

    save_interval: int = 100

    logger_backend: str = "tensorboard"

    class_type: type = RLRunner

    def __post_init__(self):
        self.env_cfg.observations = self.algorithm_cfg.observation_cfg
