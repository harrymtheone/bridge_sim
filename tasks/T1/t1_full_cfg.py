from dataclasses import MISSING

from isaaclab import terrains
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.managers import RecorderManagerBaseCfg, ActionTermCfg, ObservationGroupCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, CurriculumTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from . import T1SceneCfg, ActionsCfg, TerminationsCfg, EventCfg, CommandsCfg


@configclass
class T1FullEnvCfg:
    """Base configuration of the environment."""

    # simulation settings
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""

    # ui settings
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow
    """The class type of the UI window. Default is None.

    If None, then no UI window is created.

    Note:
        If you want to make your own UI window, you can create a class that inherits from
        from :class:`isaaclab.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        this attribute to your class type.
    """

    # general settings
    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed is not set.

    Note:
      The seed is set at the beginning of the environment initialization. This ensures that the environment
      creation is deterministic and behaves similarly across different runs.
    """

    decimation: int = MISSING
    """Number of control action updates @ sim dt per policy dt.

    For instance, if the simulation dt is 0.01s and the policy dt is 0.1s, then the decimation is 10.
    This means that the control action is updated every 10 simulation steps.
    """

    # general settings
    is_finite_horizon: bool = False
    """Whether the learning task is treated as a finite or infinite horizon problem for the agent.
    Defaults to False, which means the task is treated as an infinite horizon problem.

    This flag handles the subtleties of finite and infinite horizon tasks:

    * **Finite horizon**: no penalty or bootstrapping value is required by the the agent for
      running out of time. However, the environment still needs to terminate the episode after the
      time limit is reached.
    * **Infinite horizon**: the agent needs to bootstrap the value of the state at the end of the episode.
      This is done by sending a time-limit (or truncated) done signal to the agent, which triggers this
      bootstrapping calculation.

    If True, then the environment is treated as a finite horizon problem and no time-out (or truncated) done signal
    is sent to the agent. If False, then the environment is treated as an infinite horizon problem and a time-out
    (or truncated) done signal is sent to the agent.

    Note:
        The base :class:`ManagerBasedRLEnv` class does not use this flag directly. It is used by the environment
        wrappers to determine what type of done signal to send to the corresponding learning agent.
    """

    episode_length_s: float = MISSING
    """Duration of an episode (in seconds).

    Based on the decimation rate and physics time step, the episode length is calculated as:

    .. code-block:: python

        episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))

    For example, if the decimation rate is 10, the physics time step is 0.01, and the episode length is 10 seconds,
    then the episode length in steps is 100.
    """

    # environment settings
    scene: InteractiveSceneCfg = MISSING
    """Scene settings.

    Please refer to the :class:`isaaclab.scene.InteractiveSceneCfg` class for more details.
    """

    recorders: dict[str, RecorderManagerBaseCfg] = RecorderManagerBaseCfg()
    """Recorder settings. Defaults to recording nothing.

    Please refer to the :class:`isaaclab.managers.RecorderManager` class for more details.
    """

    observations: dict[str, ObservationGroupCfg] = MISSING
    """Observation space settings.

    Please refer to the :class:`isaaclab.managers.ObservationManager` class for more details.
    """

    actions: dict[str, ActionTermCfg] = MISSING
    """Action space settings.

    Please refer to the :class:`isaaclab.managers.ActionManager` class for more details.
    """

    events: dict = MISSING
    """Event settings. Defaults to the basic configuration that resets the scene to its default state.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """

    rerender_on_reset: bool = False
    """Whether a render step is performed again after at least one environment has been reset.
    Defaults to False, which means no render step will be performed after reset.

    * When this is False, data collected from sensors after performing reset will be stale and will not reflect the
      latest states in simulation caused by the reset.
    * When this is True, an extra render step will be performed to update the sensor data
      to reflect the latest states from the reset. This comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset.

    """

    wait_for_textures: bool = True
    """True to wait for assets to be loaded completely, False otherwise. Defaults to True."""

    xr: XrCfg | None = None
    """Configuration for viewing and interacting with the environment through an XR device."""

    # environment settings
    rewards: dict = MISSING
    """Reward settings.

    Please refer to the :class:`isaaclab.managers.RewardManager` class for more details.
    """

    terminations: dict = MISSING
    """Termination settings.

    Please refer to the :class:`isaaclab.managers.TerminationManager` class for more details.
    """

    curriculum: dict | None = None
    """Curriculum settings. Defaults to None, in which case no curriculum is applied.

    Please refer to the :class:`isaaclab.managers.CurriculumManager` class for more details.
    """

    commands: dict | None = None
    """Command settings. Defaults to None, in which case no commands are generated.

    Please refer to the :class:`isaaclab.managers.CommandManager` class for more details.
    """


@configclass
class T1FullEnv1Cfg(ManagerBasedRLEnvCfg):
    episode_length_s = 20.0

    decimation = 10

    sim = SimulationCfg(dt=0.002, render_interval=10)

    scene = T1SceneCfg(
        num_envs=4096,
        env_spacing=2.0,

        ground=terrains.TerrainImporterCfg(
            prim_path="/World/defaultGroundPlane",
            terrain_generator=terrains.TerrainGeneratorCfg(
                curriculum=True,
                size=(8, 8),
                border_width=8.,
                border_height=0.,
                num_rows=20,
                num_cols=20,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                sub_terrains={
                    'plane': terrains.MeshPlaneTerrainCfg(
                        proportion=0.5,
                    ),
                    'stairs_up': terrains.MeshPyramidStairsTerrainCfg(
                        proportion=0.25,
                        border_width=0.5,
                        step_height_range=(0.05, 0.13),
                        step_width=0.25,
                        platform_width=2,
                    ),
                    'stairs_down': terrains.MeshInvertedPyramidStairsTerrainCfg(
                        proportion=0.25,
                        border_width=0.5,
                        step_height_range=(0.05, 0.13),
                        step_width=0.25,
                        platform_width=2,
                    ),
                },
            )
        )
    )

    curriculum = {
        "terrain_levels": CurriculumTermCfg(func=mdp.curr.terrain_levels_vel)
    }

    events = EventCfg()

    commands = CommandsCfg()

    actions = ActionsCfg()

    rewards = dict(
        # ##################################### gait #####################################
        track_ref_joint_pos=RewardTermCfg(
            func=mdp.rew.track_ref_dof_pos_T1,
            weight=0.6,
            params=dict(
                command_name='base_velocity',
                ref_motion_scale=0.3,
                robot_cfg=SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "Left_Hip_Pitch", "Left_Knee_Pitch", "Left_Ankle_Pitch",
                        "Right_Hip_Pitch", "Right_Knee_Pitch", "Right_Ankle_Pitch"
                    ],
                    preserve_order=True
                )
            )
        ),
        contact_accordance=RewardTermCfg(
            func=mdp.rew.feet_contact_accordance,
            weight=0.3,
            params=dict(
                command_name='base_velocity',
                contact_sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            )
        ),
        feet_clearance=RewardTermCfg(
            func=mdp.rew.feet_clearance_masked,
            weight=0.5,
            params=dict(
                command_name="base_velocity",
                sensor_l_cfg=SceneEntityCfg("left_feet_scanner"),
                sensor_r_cfg=SceneEntityCfg("right_feet_scanner"),
                feet_height_target=0.04
            )
        ),
        # "feet_air_time": RewardTermCfg(
        #     func=mdp.feet_air_time,
        #     weight=0.125,
        #     params=dict(
        #         sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        #         command_name="base_velocity",
        #         threshold=0.5,
        #     ),
        # ),
        feet_distance=RewardTermCfg(
            func=mdp.rew.link_distance_xy,
            weight=-1.,
            params=dict(
                robot_cfg=SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link")),
                target_distance_range=(0.25, 0.5),
            )
        ),
        knee_distance=RewardTermCfg(
            func=mdp.rew.link_distance_xy,
            weight=-1.,
            params=dict(
                robot_cfg=SceneEntityCfg("robot", body_names=("Ankle_Cross_Left", "Ankle_Cross_Right")),
                target_distance_range=(0.25, 0.5),
            )
        ),
        feet_rotation=RewardTermCfg(
            func=mdp.rew.link_orientation_euler,
            weight=-0.3,
            params=dict(
                robot_cfg=SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link")),
            )
        ),
        # ##################################### task #####################################
        track_lin_vel_exp=RewardTermCfg(
            func=mdp.rew.track_lin_vel_xy_exp,
            weight=2.5,
            params=dict(command_name="base_velocity", tracking_sigma=5)
        ),
        track_ang_vel_exp=RewardTermCfg(
            func=mdp.rew.track_ang_vel_z_exp,
            weight=2.,
            params=dict(command_name="base_velocity", tracking_sigma=5)
        ),
        # ##################################### contact #####################################
        feet_slip=RewardTermCfg(
            func=mdp.rew.feet_slip,
            weight=-0.1,
            params=dict(
                robot_cfg=SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link")),
                contact_sensor_cfg=SceneEntityCfg("contact_forces")
            )
        ),
        foothold=RewardTermCfg(
            func=mdp.rew.foothold,
            weight=-0.1,
            params=dict(
                scanner_l_cfg=SceneEntityCfg("left_feet_scanner"),
                scanner_r_cfg=SceneEntityCfg("right_feet_scanner"),
                contact_sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            )
        ),
        # ##################################### regularization #####################################
        joint_pos_deviation=RewardTermCfg(
            func=mdp.rew.joint_deviation_l1,
            weight=-0.04,
            params=dict(asset_cfg=SceneEntityCfg("robot", joint_names=".*")),
        ),
        joint_yr_pos_deviation=RewardTermCfg(
            func=mdp.rew.joint_deviation_l1,
            weight=-0.5,
            params=dict(asset_cfg=SceneEntityCfg("robot", joint_names=(".*Yaw", ".*Roll"))),
        ),
        base_orientation=RewardTermCfg(func=mdp.rew.flat_orientation_l2, weight=-10.0),
        lin_vel_z=RewardTermCfg(func=mdp.rew.lin_vel_z_l2, weight=-2.0),
        ang_vel_xy=RewardTermCfg(func=mdp.rew.ang_vel_xy_l2, weight=-0.05),

        # ##################################### energy #####################################
        action_rate=RewardTermCfg(func=mdp.rew.action_rate_l2, weight=-0.01),
        dof_torques=RewardTermCfg(func=mdp.rew.joint_torques_l2, weight=-1.0e-5),
        dof_acc=RewardTermCfg(func=mdp.rew.joint_acc_l2, weight=-2.5e-7),
        undesired_contacts=RewardTermCfg(
            func=mdp.rew.undesired_contacts,
            weight=-1.0,
            params=dict(
                sensor_cfg=SceneEntityCfg("contact_forces", body_names="Trunk"),
                threshold=1.0
            ),
        ),
        # "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0)
    )

    terminations = TerminationsCfg()
