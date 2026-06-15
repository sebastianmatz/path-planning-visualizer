from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

Point = Tuple[int, int]


FloatPoint = Tuple[float, float]


Edge = Tuple[Point, Point]


OccupancyGrid = NDArray[np.bool_]
