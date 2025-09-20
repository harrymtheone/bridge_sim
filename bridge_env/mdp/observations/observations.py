from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


# def height_scan_1d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
#     sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
#     measurement = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
#
#     measurement[measurement.isinf()] = sensor.cfg.max_distance  # TODO: why????
#     return measurement
#
#
# def height_scan_2d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
#     sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
#     assert isinstance(sensor.cfg.pattern_cfg, GridPatternV2Cfg)
#
#     measurement = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
#
#     measurement[measurement.isinf()] = sensor.cfg.max_distance  # TODO: why????
#     return measurement.unflatten(1, sensor.cfg.pattern_cfg.shape)
#
#
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
