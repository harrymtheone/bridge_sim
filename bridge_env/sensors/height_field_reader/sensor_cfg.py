from dataclasses import MISSING
from typing import Literal

from isaaclab.sensors import SensorBaseCfg
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass

from . import HeightFieldReader


@configclass
class HeightFieldReaderCfg(SensorBaseCfg):
    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type = HeightFieldReader

    mesh_prim_paths: str = MISSING
    """The list of mesh primitive paths to ray cast against.

    Note:
        Currently, only a single static mesh is supported. We are working on supporting multiple
        static meshes and dynamic meshes.
    """

    alignment: Literal["base", "yaw"] = "yaw"

    pattern_cfg: PatternBaseCfg = MISSING

    offset: OffsetCfg = OffsetCfg()

    interpolation: Literal["average", "minimum"] = "minimum"

    use_guidance: bool = False

# @configclass
# class HeightEdgeScannerCfg(HeightScannerCfg):
#     class_type = HeightEdgeReader
