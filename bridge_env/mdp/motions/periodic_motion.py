from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from bridge_env.mdp.commands import PhaseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stepper_with_air_ratio(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        phase_command_name: str,
        motion_scale: float,
):
    """ asset_cfg should provide necessary joint names, in the sequence of
    l_hip_pitch, l_knee_pitch, l_ankle_pitch,
    r_hip_pitch, r_knee_pitch, r_ankle_pitch,
     """

    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(phase_command_name)

    if not isinstance(command, PhaseCommand):
        raise TypeError(f"Stepper motion generator requires PhaseCommand for clock input!"
                        f" Got [{phase_command_name}] with type [{type(command)}]")

    air_ratio = command.air_ratio
    delta_t = command.delta_t

    clocks = command.get_clocks()
    phase_swing = torch.clip((clocks - delta_t / 2) / (air_ratio - delta_t), min=0., max=1.)
    clocks = torch.sin(torch.pi * phase_swing)

    ref_dof_pos = torch.zeros(env.num_envs, robot.num_joints, device=robot.device)

    # left motion
    ref_dof_pos[:, asset_cfg.joint_ids[0]] = -clocks[:, 0] * motion_scale
    ref_dof_pos[:, asset_cfg.joint_ids[1]] = clocks[:, 0] * motion_scale * 2
    ref_dof_pos[:, asset_cfg.joint_ids[2]] = -clocks[:, 0] * motion_scale

    # right motion
    ref_dof_pos[:, asset_cfg.joint_ids[3]] = -clocks[:, 1] * motion_scale
    ref_dof_pos[:, asset_cfg.joint_ids[4]] = clocks[:, 1] * motion_scale * 2
    ref_dof_pos[:, asset_cfg.joint_ids[5]] = -clocks[:, 1] * motion_scale

    return ref_dof_pos + robot.data.default_joint_pos
