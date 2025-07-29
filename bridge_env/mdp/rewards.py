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
        robot_cfg: SceneEntityCfg,
        command_name: str,
        ref_motion_scale: float,
        double_support_phase: float = -0.3,
        tracking_sigma: float = 5.,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]

    cmd_term: PhaseCommand = env.command_manager.get_term(command_name)
    clocks = cmd_term.get_clocks()
    clock_l, clock_r = clocks[:, 0], clocks[:, 1]

    scale_1 = ref_motion_scale
    scale_2 = 2 * scale_1
    ref_dof_pos = torch.zeros_like(robot.data.joint_pos)

    # left swing (with double support phase)
    clock_l[clock_l > double_support_phase] = 0
    ref_dof_pos[:, 10 + 1] = clock_l * scale_1
    ref_dof_pos[:, 10 + 4] = -clock_l * scale_2
    ref_dof_pos[:, 10 + 5] = clock_l * scale_1

    # right swing (with double support phase)
    clock_r[clock_r > double_support_phase] = 0
    ref_dof_pos[:, 10 + 7] = clock_r * scale_1
    ref_dof_pos[:, 10 + 10] = -clock_r * scale_2
    ref_dof_pos[:, 10 + 11] = clock_r * scale_1

    ref_dof_pos[:] += robot.data.default_joint_pos

    diff = torch.norm(ref_dof_pos - robot.data.joint_pos, dim=1)
    return torch.exp(-diff * tracking_sigma) - 0.2 * diff.clamp(0., 0.5)


def feet_contact_accordance(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        command_name: str,
        contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    cmd_term: PhaseCommand = env.command_manager.get_term(command_name)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, robot_cfg.body_ids], dim=-1), dim=1)[0] > contact_force_threshold

    rew = torch.where(is_contact == cmd_term.stance_mask, 1., -0.3)
    return torch.mean(rew, dim=1)


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
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
