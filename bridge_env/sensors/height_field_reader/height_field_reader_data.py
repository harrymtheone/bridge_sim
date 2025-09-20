from dataclasses import dataclass

import torch


@dataclass
class HeightFieldReaderData:
    """Data container for the ray-cast sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.
    """
    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.
    """
    height_map: torch.Tensor = None
    """The height readings of the sensor.

    Shape is (N, B), where N is the number of sensors, B is the number of points
    in the scan pattern per sensor.
    """
