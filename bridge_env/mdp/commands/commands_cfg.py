from dataclasses import MISSING
from typing import List

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .phase_command import PhaseCommand


@configclass
class PhaseCommandCfg(CommandTermCfg):
    class_type: type = PhaseCommand

    asset_name: str = MISSING

    base_command_cfg: CommandTermCfg = MISSING

    num_phase_clock: int = MISSING

    period_s: float = MISSING

    phase_bias: List[float] = MISSING

    randomize_start_phase: bool = MISSING

    stand_walk_switch: bool = MISSING

    def __post_init__(self):
        assert self.num_phase_clock == len(self.phase_bias)
