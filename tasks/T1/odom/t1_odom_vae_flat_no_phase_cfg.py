import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from bridge_env.envs import BridgeEnvCfg
from bridge_rl.algorithms import OdomVAECfg
from bridge_rl.runners import RLTaskCfg
from tasks.T1 import T1ActionsCfg, T1CommandsCfg, T1EventCfg, T1MotionGeneratorCfg, T1SceneCfg, T1TerminationsCfg


@configclass
class RewardsCfg:
    # -- task
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.rew.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.rew.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewardTermCfg(
        func=mdp.rew.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewardTermCfg(
        func=mdp.rew.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # -- penalties
    flat_orientation_l2 = RewardTermCfg(func=mdp.rew.flat_orientation_l2, weight=0.0)
    lin_vel_z_l2 = RewardTermCfg(func=mdp.rew.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.rew.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewardTermCfg(func=mdp.rew.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewardTermCfg(func=mdp.rew.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewardTermCfg(func=mdp.rew.action_rate_l2_v2, weight=-0.01)

    # Penalize ankle joint limits
    dof_pos_limits = RewardTermCfg(
        func=mdp.rew.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewardTermCfg(
        func=mdp.rew.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Roll", ".*_Hip_Yaw"])},
    )

    termination_penalty = RewardTermCfg(func=mdp.rew.is_terminated, weight=-200.0)


@configclass
class T1FlatEnvCfg(BridgeEnvCfg):
    episode_length_s = 20.0

    decimation = 4

    sim = SimulationCfg(dt=0.005, render_interval=4)

    scene = T1SceneCfg(
        num_envs=4096,
        env_spacing=2.0,

    )
    scene.terrain = TerrainImporterCfg(
        prim_path="/World/defaultGroundPlane",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    scene.robot.init_state.joint_pos = {
        'Waist': 0.,
        'Left_Hip_Pitch': -0.2,
        'Left_Hip_Roll': 0.,
        'Left_Hip_Yaw': 0.,
        'Left_Knee_Pitch': 0.4,
        'Left_Ankle_Pitch': -0.25,
        'Left_Ankle_Roll': 0.,
        'Right_Hip_Pitch': -0.2,
        'Right_Hip_Roll': 0.,
        'Right_Hip_Yaw': 0.,
        'Right_Knee_Pitch': 0.4,
        'Right_Ankle_Pitch': -0.25,
        'Right_Ankle_Roll': 0.,
    }

    curriculum = None

    events = T1EventCfg()

    motion_generators = T1MotionGeneratorCfg()

    commands = T1CommandsCfg()

    actions = T1ActionsCfg()

    rewards = RewardsCfg()

    terminations = T1TerminationsCfg()


@configclass
class T1OdomVAEFlatNoPhaseTaskCfg(RLTaskCfg):
    env: T1FlatEnvCfg = T1FlatEnvCfg()

    algorithm = OdomVAECfg()
    algorithm.observations.scan.scan.params = dict(sensor_cfg=SceneEntityCfg("scanner"), offset=-0.7)
    algorithm.observations.proprio.phase_command = None
    algorithm.observations.critic_obs.phase_command = None

    max_iterations: int = 10000
