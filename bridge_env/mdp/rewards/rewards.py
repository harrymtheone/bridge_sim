from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

from bridge_env.envs import BridgeEnv
from bridge_env.mdp.commands.phase_command import PhaseCommand
from bridge_env.sensors.ray_caster import RayCasterV2

"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
        env: ManagerBasedRLEnv,
        tracking_sigma: float,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error * tracking_sigma)


def track_ang_vel_z_exp(
        env: ManagerBasedRLEnv,
        tracking_sigma: float,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error * tracking_sigma)


"""
gait rewards.
"""


def track_ref_dof_pos_T1(
        env: BridgeEnv,
        motion_name: str,
        tracking_sigma: float = 5.,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ref_motion = env.motion_generator.get_motion(motion_name)

    diff = torch.norm(ref_motion - robot.data.joint_pos, dim=1)
    return torch.exp(-diff * tracking_sigma) - 0.2 * diff.clamp(0., 0.5)


def feet_contact_accordance(
        env: ManagerBasedRLEnv,
        command_name: str,
        contact_sensor_cfg: SceneEntityCfg,
        contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    command = env.command_manager.get_term(command_name)

    if not isinstance(command, PhaseCommand):
        raise TypeError(f"feet_contact_accordance reward requires PhaseCommand for clock input!"
                        f" Got [{command_name}] with type [{type(command)}]")

    swing = command.swing_mask
    stance = command.stance_mask

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_force_threshold

    rew = torch.zeros_like(swing, dtype=torch.float)
    # Case 1: swing
    # Case 2: stance
    # Case 3: ~swing & ~stance → reward already 0
    rew[swing] = torch.where(is_contact[swing], -0.3, 1.0)
    rew[stance] = torch.where(is_contact[stance], 1.0, -0.3)
    return torch.mean(rew, dim=1)


def feet_clearance_masked(
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_l_cfg: SceneEntityCfg,
        sensor_r_cfg: SceneEntityCfg,
        feet_height_correction: float,
        feet_height_target: float,
) -> torch.Tensor:
    scanner_l: RayCasterV2 = env.scene.sensors[sensor_l_cfg.name]
    scanner_r: RayCasterV2 = env.scene.sensors[sensor_r_cfg.name]
    command = env.command_manager.get_term(command_name)

    if not isinstance(command, PhaseCommand):
        raise TypeError(f"feet_clearance_masked reward requires PhaseCommand for clock input!"
                        f" Got [{command_name}] with type [{type(command)}]")

    foothold_pts_height = torch.stack([
        scanner_l.data.ray_starts_w[:, :, 2] - scanner_l.data.ray_hits_w[:, :, 2] + feet_height_correction,
        scanner_r.data.ray_starts_w[:, :, 2] - scanner_r.data.ray_hits_w[:, :, 2] + feet_height_correction,
    ], dim=1)

    rew = (foothold_pts_height.mean(dim=2) / feet_height_target).clip(min=-1, max=1)
    rew[command.stance_mask] = 0.

    return rew.sum(dim=1)


def feet_air_time(
        env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than actions threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take actions step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= ~env.command_manager.get_term(command_name).is_standing_env
    return reward


def link_distance_xy(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        target_distance_range: tuple[float, float],
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]

    feet_pos_xy = robot.data.body_link_pos_w[:, robot_cfg.body_ids, :2]
    feet_dist_xy = torch.norm(feet_pos_xy[:, 0] - feet_pos_xy[:, 1], dim=1)
    penalty_min = torch.clamp(target_distance_range[0] - feet_dist_xy, 0., 0.5)
    penalty_max = torch.clamp(feet_dist_xy - target_distance_range[1], 0., 0.5)
    return penalty_min + penalty_max


def link_orientation_euler(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]

    num_links = len(robot_cfg.body_ids)
    link_quat = robot.data.body_link_quat_w[:, robot_cfg.body_ids]
    link_euler_xyz = euler_xyz_from_quat(link_quat.flatten(0, 1))
    link_euler_xy = torch.stack(link_euler_xyz[:2], dim=1).unflatten(0, (-1, num_links))
    return link_euler_xy.square().sum(dim=[1, 2])


def feet_slip(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, robot_cfg.body_ids], dim=-1), dim=1)[0] > contact_force_threshold

    feet_lin_vel = torch.norm(robot.data.body_lin_vel_w[:, robot_cfg.body_ids, :2], dim=2)
    feet_ang_vel = torch.abs(robot.data.body_ang_vel_w[:, robot_cfg.body_ids, 2])
    return torch.sum(is_contact * (feet_lin_vel + feet_ang_vel), dim=1)


def foothold(
        env: ManagerBasedRLEnv,
        scanner_l_cfg: SceneEntityCfg,
        scanner_r_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        foothold_contact_thresh: float = 0.01,
        contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    scanner_l: RayCasterV2 = env.scene.sensors[scanner_l_cfg.name]
    scanner_r: RayCasterV2 = env.scene.sensors[scanner_r_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    foothold_pts_height = torch.stack([
        scanner_l.ray_starts_w[:, :, 2] - scanner_l.data.ray_hits_w[:, :, 2] + scanner_l.cfg.reading_bias_z,
        scanner_r.ray_starts_w[:, :, 2] - scanner_r.data.ray_hits_w[:, :, 2] + scanner_r.cfg.reading_bias_z,
    ], dim=1)

    valid_foothold_perc = (foothold_pts_height < foothold_contact_thresh).sum(2) / foothold_pts_height.size(2)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_force_threshold
    rew = (1 - valid_foothold_perc) * is_contact
    return rew.sum(dim=1)


"""
Action penalties.
"""


def action_rate_l2_v2(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    action_dict = env.action_manager.action
    prev_action_dict = env.action_manager.prev_action
    
    if action_name is None:
        current_actions = torch.cat(list(action_dict.values()), dim=1)
        prev_actions = torch.cat(list(prev_action_dict.values()), dim=1)
        return torch.sum(torch.square(current_actions - prev_actions), dim=1)
    else:
        current_action = action_dict[action_name]
        prev_action = prev_action_dict[action_name]
        return torch.sum(torch.square(current_action - prev_action), dim=1)
