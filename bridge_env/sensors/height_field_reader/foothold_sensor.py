from __future__ import annotations

import torch

from bridge_env.utils import configclass
from bridge_env.utils.others import VisualizationTask
from . import HeightReader, HeightScannerCfg


class FootholdSensor(HeightReader):
    @property
    def data_raw(self):
        return self._foothold_output

    def _init_buffers(self):
        super()._init_buffers()

        self._foothold_pts_contact = self._zero_tensor(self.num_envs, self.num_points, dtype=torch.bool)
        self._foothold_output = self._zero_tensor(self.num_envs, dtype=torch.float)

    def _render(self):
        super()._render()

        self._foothold_pts_contact[:] = torch.abs(self.pos_w[:, 2:3] - self._height_map_output) < self.cfg.feet_contact_threshold
        self._foothold_output[:] = self._foothold_pts_contact.sum(dim=1) / self.num_points

    def debug_visualize(self) -> list[VisualizationTask]:
        if not self.cfg.debug_vis:
            return []

        pts = self._scan_points_pos_w[self._sim.lookat_id].clone()

        if self.cfg.show_source:
            vis_task = self.cfg.debug_vis_params.copy()
            vis_task.points = pts.cpu().numpy()
            return [vis_task]

        pts[:, 2] = self._height_map_output[self._sim.lookat_id]

        # Visualize edge vs non-edge points with different colors
        contact_mask = self._foothold_pts_contact[self._sim.lookat_id]
        pts_contact = pts[contact_mask]
        pts_non_contact = pts[~contact_mask]

        vis_task_contact = self.cfg.debug_vis_params.copy()
        vis_task_contact.points = pts_contact.cpu().numpy()
        vis_task_contact.color = (0, 1, 0)  # Red for edges

        vis_task_non_contact = self.cfg.debug_vis_params.copy()
        vis_task_non_contact.points = pts_non_contact.cpu().numpy()
        vis_task_non_contact.color = (1, 0, 0)  # Green for non-edges

        return [vis_task_contact, vis_task_non_contact]


@configclass
class FootholdSensorCfg(HeightScannerCfg):
    class_type = FootholdSensor

    feet_contact_threshold: float = 0.01

    show_source: bool = False
