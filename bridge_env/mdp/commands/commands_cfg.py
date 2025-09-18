from dataclasses import MISSING
from typing import List

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .phase_command import PhaseCommand


@configclass
class PhaseCommandCfg(CommandTermCfg):
    class_type: type = PhaseCommand

    # base_command_name: str = MISSING

    num_clocks: int = MISSING

    period_s: float = MISSING

    clock_bias: List[float] = MISSING

    randomize_start_phase: bool = MISSING

    stand_walk_switch: bool = MISSING

    air_ratio: float = MISSING

    delta_t: float = MISSING

    resampling_time_range = (1e5, 1e5)
    """ Phase command will never resample """
