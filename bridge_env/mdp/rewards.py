import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from bridge_env.mdp.commands.phase_command import PhaseCommand

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
Reference gait tracking rewards.
"""


def track_ref_dof_pos_T1(
        env: ManagerBasedRLEnv,
        command_name: str,
        ref_motion_scale: float,
        tracking_sigma: float = 5.,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]

    cmd_term: PhaseCommand = env.command_manager.get_term(command_name)
    clocks = cmd_term.get_clocks()
    swing_ratio = cmd_term.air_ratio
    delta_t = cmd_term.delta_t

    ref_dof_pos = torch.zeros_like(robot.data.joint_pos)
    scale_1 = ref_motion_scale
    scale_2 = 2 * scale_1

    phase_swing = torch.clip((clocks - delta_t / 2) / (swing_ratio - delta_t), min=0., max=1.)
    clock = torch.sin(torch.pi * phase_swing)

    # left motion
    ref_dof_pos[:, 1] = -clock[:, 0] * scale_1
    ref_dof_pos[:, 4] = clock[:, 0] * scale_2
    ref_dof_pos[:, 5] = -clock[:, 0] * scale_1

    # right motion
    ref_dof_pos[:, 7] = -clock[:, 1] * scale_1
    ref_dof_pos[:, 10] = clock[:, 1] * scale_2
    ref_dof_pos[:, 11] = -clock[:, 1] * scale_1

    ref_dof_pos[:] += robot.data.default_joint_pos

    diff = torch.norm(ref_dof_pos - robot.data.joint_pos, dim=1)
    return torch.exp(-diff * tracking_sigma) - 0.2 * diff.clamp(0., 0.5)


def feet_contact_accordance(
        env: ManagerBasedRLEnv,
        command_name: str,
        contact_sensor_cfg: SceneEntityCfg,
        contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    cmd_term: PhaseCommand = env.command_manager.get_term(command_name)

    swing = cmd_term.swing_mask
    stance = cmd_term.stance_mask

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_force_threshold

    rew = torch.zeros_like(swing, dtype=torch.float)
    # Case 1: swing
    # Case 2: stance
    # Case 3: ~swing & ~stance → reward already 0
    rew[swing] = torch.where(is_contact[swing], -0.3, 1.0)
    rew[stance] = torch.where(is_contact[stance], 1.0, -0.3)
    return torch.mean(rew, dim=1)


def feet_clearance_exp_period(
        env: ManagerBasedRLEnv,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_force_threshold: float = 1.0,
        desired_clearance_height: float = 0.08,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cmd_term: PhaseCommand = env.command_manager.get_term(command_name)
    stance = cmd_term.stance_mask

    feet_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    feet_height = feet_pos_w[:, :, 2]  # Z-coordinate
    pos_error = torch.square(feet_height - desired_clearance_height) * stance
    
    return -torch.sum(pos_error, dim=1)


def feet_air_time(
        env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
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
