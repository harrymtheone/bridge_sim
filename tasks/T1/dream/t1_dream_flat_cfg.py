from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationTermCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from bridge_env.envs import BridgeEnvCfg
from bridge_rl.algorithms import DreamWaQCfg
from bridge_rl.runners import RLTaskCfg
from tasks.T1 import T1SceneCfg, T1ActionsCfg, T1TerminationsCfg, T1EventCfg, T1MotionGeneratorCfg, T1CommandsCfg


@configclass
class RewardsCfg:
    # ##################################### gait #####################################
    track_ref_joint_pos = RewardTermCfg(
        func=mdp.rew.track_ref_dof_pos_T1,
        weight=2.0,
        params=dict(
            motion_name='ref_motion',
        )
    )

    contact_accordance = RewardTermCfg(
        func=mdp.rew.feet_contact_accordance,
        weight=1.2,
        params=dict(
            command_name='phase',
            contact_sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        )
    )

    feet_clearance = RewardTermCfg(
        func=mdp.rew.feet_clearance_masked,
        weight=0.6,
        params=dict(
            command_name="phase",
            sensor_l_cfg=SceneEntityCfg("left_feet_scanner"),
            sensor_r_cfg=SceneEntityCfg("right_feet_scanner"),
            feet_height_correction=-0.03,
            feet_height_target=0.04,
        )
    )

    # feet_air_time = RewardTermCfg(
    #     func=mdp.feet_air_time,
    #     weight=0.125,
    #     params=dict(
    #         sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*foot.*"),
    #         command_name="base_velocity",
    #         threshold=0.5,
    #     ),
    # )

    feet_distance = RewardTermCfg(
        func=mdp.rew.link_distance_xy,
        weight=-1.,
        params=dict(
            robot_cfg=SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link")),
            target_distance_range=(0.25, 0.5),
        )
    )

    knee_distance = RewardTermCfg(
        func=mdp.rew.link_distance_xy,
        weight=-1.,
        params=dict(
            robot_cfg=SceneEntityCfg("robot", body_names=("Ankle_Cross_Left", "Ankle_Cross_Right")),
            target_distance_range=(0.25, 0.5),
        )
    )

    feet_rotation = RewardTermCfg(
        func=mdp.rew.link_orientation_euler,
        weight=-0.3,
        params=dict(
            robot_cfg=SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link")),
        )
    )

    # ##################################### task #####################################
    track_lin_vel_exp = RewardTermCfg(
        func=mdp.rew.track_lin_vel_xy_exp,
        weight=1.0,
        params=dict(command_name="base_velocity", tracking_sigma=5)
    )

    track_ang_vel_exp = RewardTermCfg(
        func=mdp.rew.track_ang_vel_z_exp,
        weight=0.5,
        params=dict(command_name="base_velocity", tracking_sigma=5)
    )

    # ##################################### contact #####################################
    feet_slip = RewardTermCfg(
        func=mdp.rew.feet_slip,
        weight=-0.1,
        params=dict(
            robot_cfg=SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link")),
            contact_sensor_cfg=SceneEntityCfg("contact_forces")
        )
    )

    # foothold = RewardTermCfg(
    #     func=mdp.rew.foothold,
    #     weight=-0.1,
    #     params=dict(
    #         scanner_l_cfg=SceneEntityCfg("left_feet_scanner"),
    #         scanner_r_cfg=SceneEntityCfg("right_feet_scanner"),
    #         contact_sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*foot.*"),
    #         foothold_contact_thresh=0.03 + 0.01,
    #     )
    # )

    # ##################################### regularization #####################################
    joint_pos_deviation = RewardTermCfg(
        func=mdp.rew.joint_deviation_l1,
        weight=-0.04,
        params=dict(asset_cfg=SceneEntityCfg("robot", joint_names=".*")),
    )

    # joint_yr_pos_deviation = RewardTermCfg(
    #     func=mdp.rew.joint_deviation_l1,
    #     weight=-0.5,
    #     params=dict(asset_cfg=SceneEntityCfg("robot", joint_names=(".*Yaw", ".*Roll"))),
    # )

    base_orientation = RewardTermCfg(func=mdp.rew.flat_orientation_l2, weight=-10.0)
    lin_vel_z = RewardTermCfg(func=mdp.rew.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewardTermCfg(func=mdp.rew.ang_vel_xy_l2, weight=-0.05)

    # ##################################### energy #####################################
    action_rate = RewardTermCfg(func=mdp.rew.action_rate_l2_v2, weight=-0.01)
    dof_torques = RewardTermCfg(func=mdp.rew.joint_torques_l2, weight=-1.0e-5)
    # dof_acc = RewardTermCfg(func=mdp.rew.joint_acc_l2, weight=-2.5e-7)

    undesired_contacts = RewardTermCfg(
        func=mdp.rew.undesired_contacts,
        weight=-1.0,
        params=dict(
            sensor_cfg=SceneEntityCfg("contact_forces", body_names="Trunk"),
            threshold=1.0
        ),
    )

    # dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0)
    termination_penalty = RewardTermCfg(func=mdp.rew.is_terminated, weight=-200.0)


@configclass
class T1FlatEnvCfg(BridgeEnvCfg):
    episode_length_s = 20.0

    decimation = 10

    sim = SimulationCfg(dt=0.002, render_interval=10)

    scene = T1SceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        terrain=AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    )

    curriculum = None

    events = T1EventCfg()

    motion_generators = T1MotionGeneratorCfg()

    commands = T1CommandsCfg()

    actions = T1ActionsCfg()

    rewards = RewardsCfg()

    terminations = T1TerminationsCfg()


@configclass
class T1DreamWaqFlatTaskCfg(RLTaskCfg):
    env: T1FlatEnvCfg = T1FlatEnvCfg()

    # only_positive_reward = True
    # only_positive_reward_until = 500

    algorithm = DreamWaQCfg()
    algorithm.observations.scan.scan = ObservationTermCfg(func=mdp.obs.height_scan, params={"sensor_cfg": SceneEntityCfg("scanner")})

    max_iterations: int = 10000
