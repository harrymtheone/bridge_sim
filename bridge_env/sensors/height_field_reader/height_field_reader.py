from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import numpy as np
import omni.log
import torch
from isaaclab.sensors import SensorBase
from isaaclab.terrains.trimesh.utils import make_plane
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, UsdPhysics

from . import HeightFieldReaderData

if TYPE_CHECKING:
    from . import HeightFieldReaderCfg


class HeightFieldReader(SensorBase):
    cfg: HeightFieldReaderCfg

    def __init__(self, cfg: HeightFieldReaderCfg):
        super().__init__(cfg)

        # Create empty variables for storing output data
        self._data = HeightFieldReaderData()

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim.")
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")

        # load the meshes by parsing the stage
        self._initialize_height_field()
        # initialize the ray start and directions
        self._initialize_rays_impl()

    def _initialize_height_field(self):
        # check if the prim is a plane - handle PhysX plane as a special case
        # if a plane exists then we need to create an infinite mesh that is a plane
        mesh_prim = sim_utils.get_first_matching_child_prim(
            self.cfg.mesh_prim_paths, lambda prim: prim.GetTypeName() == "Plane"
        )

        # if we did not find a plane then we need to read the mesh
        if mesh_prim is None:
            # obtain the mesh prim
            mesh_prim = sim_utils.get_first_matching_child_prim(
                self.cfg.mesh_prim_paths, lambda prim: prim.GetTypeName() == "Mesh"
            )

            # check if valid
            if mesh_prim is None or not mesh_prim.IsValid():
                raise RuntimeError(f"Invalid mesh prim path: {self.cfg.mesh_prim_paths}")

            # cast into UsdGeomMesh
            mesh_prim = UsdGeom.Mesh(mesh_prim)

            # read the vertices and faces
            points = np.asarray(mesh_prim.GetPointsAttr().Get())
            transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
            points = np.matmul(points, transform_matrix[:3, :3].T)
            points += transform_matrix[:3, 3]
            indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())

            # print info
            omni.log.info(
                f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
            )
        else:
            mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            points = mesh.vertices
            indices = mesh.faces

            # print info
            omni.log.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")

        self._height_field = 0  #

        # add the warp mesh to the list
        self.meshes[self.cfg.mesh_prim_paths] = wp_mesh

    # throw an error if no meshes are found
    if all([mesh_prim_path not in self.meshes for mesh_prim_path in self.cfg.mesh_prim_paths]):
        raise RuntimeError(
            f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
        )

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
