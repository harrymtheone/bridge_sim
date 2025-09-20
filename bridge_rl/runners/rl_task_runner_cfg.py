from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from bridge_env.envs import BridgeEnvCfg
from . import RLRunner

if TYPE_CHECKING:
    from bridge_rl.algorithms import PPOCfg


@configclass
class RLTaskCfg:
    class_type: type = RLRunner

    env: BridgeEnvCfg = MISSING

    algorithm: PPOCfg = MISSING

    max_iterations: int = MISSING

    num_steps_per_update: int = 24

    save_interval: int = 100

    logger_backend: str = "tensorboard"

    log_root_dir: str = MISSING

    project_name: str = MISSING

    exptid: str = MISSING

    resume_id: str = None

    checkpoint: int = -1

    def __post_init__(self):
        self.env.observations = self.algorithm.observations

        self.algorithm.num_steps_per_update = self.num_steps_per_update
