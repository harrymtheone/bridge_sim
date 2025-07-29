from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from . import PhaseCommandCfg


class PhaseCommand(CommandTerm):
    cfg: PhaseCommandCfg

    def __init__(self, cfg: PhaseCommandCfg, env: ManagerBasedEnv):
        self.env: ManagerBasedEnv = env
        self.robot: Articulation = env.scene[cfg.base_command_cfg.asset_name]
        self.base_command_cfg = cfg.base_command_cfg
        self.base_command = self.base_command_cfg.class_type(self.base_command_cfg, env)

        super().__init__(cfg, env)

        self.num_phase_clock = self.cfg.num_phase_clock
        self.period_s = self.cfg.period_s
        self.phase_bias = torch.tensor(self.cfg.phase_bias, device=env.device).unsqueeze(0)

        self.phase_length_buf = torch.zeros(env.num_envs, device=env.device)
        self.start_phase = torch.zeros(env.num_envs, device=env.device)
        self.phase = torch.zeros(env.num_envs, device=env.device)

        self.phase_cmd = torch.zeros(env.num_envs, self.num_phase_clock, device=env.device)

        self.metrics = self.base_command.metrics

    def __str__(self) -> str:  # TODO: add more details
        base_command_name = str(self.base_command).split(":")[0]
        msg = f"PhaseCommand warpped with {base_command_name}:\n"
        msg += f"\tBase command dimension: {self.base_command.command.size(1)}\n"
        msg += f"\tNumber of phase clock: {self.num_phase_clock}\n"
        msg += f"\tPeriod: {self.period_s}s\n"
        msg += f"\tPhase bias: {self.cfg.phase_bias}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return torch.cat([self.base_command.command, self.phase_cmd], dim=1)

    @property
    def is_standing_env(self) -> torch.Tensor:
        assert hasattr(self.base_command, "is_standing_env"), "base command must have is_standing_env attribute"
        return self.base_command.is_standing_env

    @property
    def stance_mask(self) -> torch.Tensor:
        return self.get_clocks() >= 0 | self.is_standing_env[:, None]

    def get_clocks(self) -> torch.Tensor:
        return torch.sin(2 * torch.pi * (self.phase[:, None] + self.phase_bias))

    def _update_metrics(self):
        self.base_command._update_metrics()

    def _resample_command(self, env_ids: Sequence[int]):
        self.base_command._resample_command(env_ids)

        self.phase_length_buf[env_ids] = 0.

        if self.cfg.randomize_start_phase:
            self.start_phase[env_ids] = torch.rand(len(env_ids), device=self.start_phase.device)

    def _update_command(self):
        self.base_command._update_command()

        self.phase_length_buf[:] += self.env.step_dt
        self.phase[:] = (self.phase_length_buf / self.period_s + self.start_phase) % 1.0

        self.phase_cmd[:] = self.get_clocks()

        if self.cfg.stand_walk_switch:
            self.phase_cmd[:] *= ~self.is_standing_env.unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        return self.base_command._set_debug_vis_impl(debug_vis)

    def _debug_vis_callback(self, event):
        return self.base_command._debug_vis_callback(event)
