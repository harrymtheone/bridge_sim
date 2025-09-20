from __future__ import annotations

import torch

from bridge_env.sensors.height_field_scanner.height_scanner import HeightScanner, HeightScannerCfg


class HeightEdgeReader(HeightScanner):
    _cfg: HeightEdgeScannerCfg

    @property
    def data_raw(self):
        return torch.stack([self._height_map_output, self._edge_map_output], dim=1)

    def _init_buffers(self):
        super()._init_buffers()

        # Initialize edge map from terrain
        self._global_edge_map = torch.from_numpy(self._sim.terrain.edge_map).to(self.device)

        # Initialize edge output buffer
        self._edge_map_output = self._zero_tensor(self.num_envs, self.num_points, dtype=torch.bool)

    def _render(self):
        super()._render()

        # Read edge mask from global edge map
        self._edge_map_output[:] = self._global_edge_map[self._map_px, self._map_py].view(self.num_envs, -1)

    def debug_visualize(self) -> list[VisualizationTask]:
        if not self.cfg.debug_vis:
            return []

        pts = self._scan_points_pos_w[self._sim.lookat_id].clone()

        pts[:, 2] = self._height_map_output[self._sim.lookat_id]

        # Visualize edge vs non-edge points with different colors
        edge_mask = self._edge_map_output[self._sim.lookat_id]
        pts_edge = pts[edge_mask]
        pts_non_edge = pts[~edge_mask]

        vis_task_edge = self.cfg.debug_vis_params.copy()
        vis_task_edge.points = pts_edge.cpu().numpy()
        vis_task_edge.color = (1, 0, 0)  # Red for edges

        vis_task_non_edge = self.cfg.debug_vis_params.copy()
        vis_task_non_edge.points = pts_non_edge.cpu().numpy()
        vis_task_non_edge.color = (0, 1, 0)  # Green for non-edges

        return [vis_task_edge, vis_task_non_edge]


