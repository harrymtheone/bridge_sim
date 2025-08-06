from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import torch
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply, convert_quat
from isaaclab.utils.warp import raycast_mesh
from isaacsim.core.prims import XFormPrim

if TYPE_CHECKING:
    from . import FootholdRayCasterCfg


class FootholdRayCaster(RayCaster):
    cfg: FootholdRayCasterCfg
    ray_starts_w: torch.Tensor

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self.ray_starts_w = self.ray_starts.clone()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift to ray starting position in world frame
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # check if user provided attach_yaw_only flag
        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' property will be deprecated in a future release."
                " Please use the parameter 'ray_alignment' instead."
            )
            # set ray alignment to yaw
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."
            # log the warning
            omni.log.warn(msg)
        # ray cast based on the sensor poses
        if self.cfg.ray_alignment == "foothold":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self.ray_starts_w = ray_starts_w
        self._data.ray_hits_w[env_ids] = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]

        # apply vertical drift to ray starting position in ray caster frame
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

    def _debug_vis_callback(self, event):
        # remove possible inf values
        # viz_points = self.ray_starts_w.reshape(-1, 3)
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        # show ray hit positions
        self.ray_visualizer.visualize(viz_points)
