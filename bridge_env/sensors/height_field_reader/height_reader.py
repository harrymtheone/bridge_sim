from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

import torch
from isaaclab.sensors import SensorBase


class HeightReader(SensorBase):
    _cfg: HeightScannerCfg

    @property
    def data_raw(self):
        return self._height_map_output

    @property
    def num_points(self):
        return self._scan_points_pos_b.size(1)

    def _init_buffers(self):
        super()._init_buffers()

        self._horizontal_scale = self._sim.terrain.cfg.horizontal_scale

        if self.cfg.use_guidance:
            self._global_height_map = torch.from_numpy(self._sim.terrain.height_map_guidance).to(self.device)
        else:
            self._global_height_map = torch.from_numpy(self._sim.terrain.height_map).to(self.device)

        pattern_cfg: PatternBaseCfg = self.cfg.pattern_cfg
        self._scan_points_pos_b = pattern_cfg.func(pattern_cfg, self.device).repeat(self.num_envs, 1, 1)

        self._scan_points_pos_w = self._zero_tensor(self.num_envs, self.num_points, 3)
        self._height_map_output = self._zero_tensor(self.num_envs, self.num_points)

        # Store map coordinates to avoid recomputation in derived classes
        self._map_px = self._zero_tensor(self.num_envs * self.num_points, dtype=torch.long)
        self._map_py = self._zero_tensor(self.num_envs * self.num_points, dtype=torch.long)

    def _render(self):
        # rotate the sensor
        if self.cfg.alignment == "base":
            self._scan_points_pos_w[:] = transform_by_quat(
                self._scan_points_pos_b,
                self.sensor_quat_sim[:, None, :].repeat(1, self.num_points, 1)
            ).unflatten(0, (self.num_envs, -1))

        elif self.cfg.alignment == "yaw":
            sensor_yaw = quat_to_xyz(self.sensor_quat_sim)[..., 2:3]
            self._scan_points_pos_w[:] = transform_by_yaw(
                self._scan_points_pos_b,
                sensor_yaw.repeat(1, self.num_points)
            ).unflatten(0, (self.num_envs, -1))

        else:
            raise ValueError(f"Unknown alignment {self.cfg.alignment}")

        # convert to world frame
        self._scan_points_pos_w[:] += self.sensor_pos_sim[:, None, :]

        # convert to map indices
        points = (self._scan_points_pos_w / self._horizontal_scale).long()
        self._map_px[:] = torch.clip(points[:, :, 0].flatten(), 0, self._global_height_map.size(0) - 2)
        self._map_py[:] = torch.clip(points[:, :, 1].flatten(), 0, self._global_height_map.size(1) - 2)

        # read height from global height map
        heights1 = self._global_height_map[self._map_px, self._map_py]
        heights2 = self._global_height_map[self._map_px + 1, self._map_py]
        heights3 = self._global_height_map[self._map_px, self._map_py + 1]
        heights4 = self._global_height_map[self._map_px + 1, self._map_py + 1]

        if self.cfg.interpolation == "average" or self.cfg.use_guidance:
            heights = (heights1 + heights2 + heights3 + heights4) / 4

        elif self.cfg.interpolation == "minimum":
            heights = torch.minimum(heights1, heights2)
            heights = torch.minimum(heights, heights3)
            heights = torch.minimum(heights, heights4)
        else:
            raise ValueError(f"Unknown interpolation {self.cfg.interpolation}")

        self._height_map_output[:] = heights.view(self.num_envs, -1)

    def debug_visualize(self) -> list[VisualizationTask]:
        if not self.cfg.debug_vis:
            return []

        pts = self._scan_points_pos_w[self._sim.lookat_id].clone()
        pts[:, 2] = self._height_map_output[self._sim.lookat_id]

        vis_task = self.cfg.debug_vis_params.copy()
        vis_task.points = pts.cpu().numpy()
        return [vis_task]


