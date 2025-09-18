from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from . import PhaseCommandCfg


class PhaseCommand(CommandTerm):
    cfg: PhaseCommandCfg

    def __init__(self, cfg: PhaseCommandCfg, env: ManagerBasedRLEnv):
        self.env: ManagerBasedRLEnv = env

        super().__init__(cfg, env)

        self.num_phase_clock = self.cfg.num_clocks
        self.period_s = self.cfg.period_s
        self.phase_bias = torch.tensor(self.cfg.clock_bias, device=env.device).unsqueeze(0)
        self.air_ratio = cfg.air_ratio
        self.delta_t = cfg.delta_t

        self.phase_length_buf = torch.zeros(env.num_envs, device=env.device)
        self.start_phase = torch.zeros(env.num_envs, device=env.device)
        self.phase = torch.zeros(env.num_envs, device=env.device)

        self.phase_cmd = torch.zeros(env.num_envs, self.num_phase_clock, device=env.device)

    def __str__(self) -> str:
        msg = "PhaseCommand:\n"
        msg += f"\tNumber of phase clock: {self.num_phase_clock}\n"
        msg += f"\tPeriod: {self.period_s}s\n"
        msg += f"\tPhase bias: {self.cfg.clock_bias}\n"
        return msg

    """
    Properties
    """

    def _update_metrics(self):
        return

    @property
    def command(self) -> torch.Tensor:
        return torch.sin(2 * torch.pi * self.phase_cmd)

    @property
    def stance_mask(self) -> torch.Tensor:
        phase = self.get_phase()
        return (phase >= self.air_ratio + self.delta_t) & (phase < (1. - self.delta_t))

    @property
    def swing_mask(self) -> torch.Tensor:
        phase = self.get_phase()
        return (phase >= self.delta_t) & (phase < (self.air_ratio - self.delta_t))

    def get_phase(self) -> torch.Tensor:
        return (self.phase[:, None] + self.phase_bias) % 1.0

    def _resample_command(self, env_ids: Sequence[int]):
        self.phase_length_buf[env_ids] = 0.

        if self.cfg.randomize_start_phase:
            self.start_phase[env_ids] = torch.rand(len(env_ids), device=self.start_phase.device)

    def _update_command(self):
        self.phase_length_buf[:] += self.env.step_dt
        self.phase[:] = (self.phase_length_buf / self.period_s + self.start_phase) % 1.0

        self.phase_cmd[:] = self.get_phase()

        # if self.cfg.stand_walk_switch:  TODO
        #     self.phase_cmd[:] *= ~self.is_standing_env.unsqueeze(-1)
