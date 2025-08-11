from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bridge_rl.algorithms import DreamWaQCfg
from bridge_rl.algorithms.odom.odom_cfg import OdomCfg
from bridge_rl.runners import RLTaskCfg
from . import T1FlatEnvCfg, T1ParkourEnvCfg


@configclass
class T1FlatDreamWaqTaskCfg(RLTaskCfg):
    env_cfg: T1FlatEnvCfg = T1FlatEnvCfg()

    algorithm_cfg = DreamWaQCfg()

    max_iterations: int = 10000

    def __post_init__(self):
        self.algorithm_cfg.observation_cfg.critic_obs.foot_scan_l.params = {
            "sensor_cfg": SceneEntityCfg("left_feet_scanner")
        }

        self.algorithm_cfg.observation_cfg.critic_obs.foot_scan_r.params = {
            "sensor_cfg": SceneEntityCfg("right_feet_scanner")
        }

        self.algorithm_cfg.observation_cfg.critic_obs.feet_is_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*")
        }

        self.algorithm_cfg.observation_cfg.scan.scan.params = {
            "sensor_cfg": SceneEntityCfg("scan_scanner")
        }

        super().__post_init__()


@configclass
class T1ParkourDreamWaqTaskCfg(RLTaskCfg):
    env_cfg: T1ParkourEnvCfg = T1ParkourEnvCfg()

    algorithm_cfg = DreamWaQCfg()

    max_iterations: int = 10000

    def __post_init__(self):
        self.algorithm_cfg.observation_cfg.critic_obs.foot_scan_l.params = {
            "sensor_cfg": SceneEntityCfg("left_feet_scanner")
        }

        self.algorithm_cfg.observation_cfg.critic_obs.foot_scan_r.params = {
            "sensor_cfg": SceneEntityCfg("right_feet_scanner")
        }

        self.algorithm_cfg.observation_cfg.critic_obs.feet_is_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*")
        }

        self.algorithm_cfg.observation_cfg.scan.scan.params = {
            "sensor_cfg": SceneEntityCfg("scan_scanner")
        }

        super().__post_init__()


@configclass
class T1ParkourOdomPPOTaskCfg(RLTaskCfg):
    env_cfg: T1ParkourEnvCfg = T1ParkourEnvCfg()

    algorithm_cfg = OdomCfg(
        continue_from_last_std=False,
        init_noise_std=0.5
    )

    max_iterations: int = 10000

    def __post_init__(self):
        self.algorithm_cfg.observation_cfg.critic_obs.foot_scan_l.params = {
            "sensor_cfg": SceneEntityCfg("left_feet_scanner")
        }

        self.algorithm_cfg.observation_cfg.critic_obs.foot_scan_r.params = {
            "sensor_cfg": SceneEntityCfg("right_feet_scanner")
        }

        self.algorithm_cfg.observation_cfg.critic_obs.feet_is_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*")
        }

        self.algorithm_cfg.observation_cfg.scan.scan.params = {
            "sensor_cfg": SceneEntityCfg("scan_scanner")
        }

        # self.algorithm_cfg.observation_cfg.depth.d435.params = {
        #     "sensor_cfg": SceneEntityCfg("depth_d435")
        # }

        super().__post_init__()
