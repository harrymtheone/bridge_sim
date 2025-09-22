from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from bridge_env.sensors import RayCasterV2
from bridge_env.sensors.ray_caster.patterns import GridPatternV2Cfg


def height_scan_1d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    sensor: RayCasterV2 = env.scene.sensors[sensor_cfg.name]
    measurement = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    measurement[measurement.isinf()] = sensor.cfg.max_distance  # TODO: why????
    return measurement


def height_scan_2d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    sensor: RayCasterV2 = env.scene.sensors[sensor_cfg.name]
    assert isinstance(sensor.cfg.pattern_cfg, GridPatternV2Cfg)

    measurement = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    measurement[measurement.isinf()] = sensor.cfg.max_distance  # TODO: why????
    return measurement.unflatten(1, sensor.cfg.pattern_cfg.shape)


# def foothold_1d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     scanner: FootholdRayCaster = env.scene.sensors[sensor_cfg.name]
#
#     measurement = scanner.ray_starts_w[:, :, 2] - scanner.data.ray_hits_w[:, :, 2] + scanner.cfg.reading_bias_z
#     measurement[measurement.isinf()] = scanner.cfg.max_distance  # TODO: why????
#     return measurement


def link_is_contact(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 5.) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact.float()  # noqa


"""
Actions.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_v2(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned (concatenated from all terms).
    """
    action_dict = env.action_manager.action

    if action_name is None:
        return torch.cat(list(action_dict.values()), dim=1)
    else:
        return action_dict[action_name]
