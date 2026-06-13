"""
Path Planning Visualizer
========================

Interactive desktop application for exploring and comparing path-planning
algorithms on occupancy-grid maps.

Supported algorithms:
- Sampling-Based: RRT, RRT-Connect, BiTRRT, KPIECE, RRT*, PRM, SBL, FMT*, BIT*
- Graph Search: A*, Dijkstra
- Potential Field: APF
- Trajectory Optimization: CHOMP, STOMP, TrajOpt, ITOMP, GPMP
- Metaheuristic: PSO, Genetic

Usage:
    python -m path_planning_visualizer
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# The version lives in exactly one place: the ``version`` field of
# ``pyproject.toml``. Installing the package (``pip install -e .``) records it as
# package metadata, which we read back here so nothing else has to hardcode it.
try:
    __version__ = _pkg_version("path-planning-visualizer")
except PackageNotFoundError:  # running from a raw source tree without an install
    __version__ = "0.0.0+dev"

from .types import Point, FloatPoint, Edge, OccupancyGrid
from .geometry import (
    bilinear_sample_scalar,
    bilinear_sample_vector,
    blend_float_paths,
    clamp_point,
    compute_path_length,
    dist,
    float_polyline_collision_free,
    integrate_holonomic_state,
    iter_path_pixels,
    l1_dist,
    line_collision_free,
    linf_dist,
    make_distance_field,
    resample_float_path_fixed_count,
    resample_float_path_points,
    resample_path_points,
    round_point,
    segment_points,
    select_holonomic_input,
    shortcut_path,
    smooth_display_path,
    smooth_float_polyline,
    steer,
)
from .metrics import (
    PathMetrics,
    compute_path_mean_clearance,
    compute_path_metrics,
    compute_path_min_clearance,
    compute_path_smoothness,
)
from .mapping import (
    blank_occupancy,
    image_to_occupancy,
    occupancy_to_image,
    paint_disk,
)
from .planners.base import BasePlanner, StepResult
from .planners.registry import (
    ALGORITHM_GROUPS,
    ALGORITHM_INFO,
    ANYTIME_ALGOS,
    AVAILABLE_PLANNERS,
    SAMPLING_BASED_ALGOS,
)
from .planners import (
    APFPlanner,
    AStarPlanner,
    BiTRRTPlanner,
    BITStarPlanner,
    CHOMPPlanner,
    DijkstraPlanner,
    FMTStarPlanner,
    GeneticPlanner,
    GPMPPlanner,
    ITOMPPlanner,
    KPIECEPlanner,
    PRMPlanner,
    PSOPlanner,
    RRTConnectPlanner,
    RRTPlanner,
    RRTStarPlanner,
    SBLPlanner,
    STOMPPlanner,
    TrajOptPlanner,
)
from .gui.canvas import ImageCanvas
from .gui.main_window import MainWindow
from .app import main

__all__ = [
    "__version__",
    # types
    "Point", "FloatPoint", "Edge", "OccupancyGrid",
    # geometry
    "dist", "l1_dist", "linf_dist", "round_point", "select_holonomic_input",
    "integrate_holonomic_state", "steer", "clamp_point", "make_distance_field",
    "bilinear_sample_scalar", "bilinear_sample_vector", "segment_points",
    "line_collision_free", "float_polyline_collision_free", "iter_path_pixels",
    "compute_path_length", "resample_path_points", "resample_float_path_points",
    "resample_float_path_fixed_count", "smooth_float_polyline", "smooth_display_path",
    "blend_float_paths", "shortcut_path",
    # metrics
    "PathMetrics", "compute_path_metrics", "compute_path_min_clearance",
    "compute_path_mean_clearance", "compute_path_smoothness",
    # mapping
    "image_to_occupancy", "occupancy_to_image", "blank_occupancy", "paint_disk",
    # planners
    "BasePlanner", "StepResult", "AVAILABLE_PLANNERS", "ALGORITHM_GROUPS",
    "SAMPLING_BASED_ALGOS", "ANYTIME_ALGOS", "ALGORITHM_INFO",
    "RRTPlanner", "RRTConnectPlanner", "BiTRRTPlanner", "KPIECEPlanner",
    "RRTStarPlanner", "PRMPlanner", "SBLPlanner", "FMTStarPlanner", "BITStarPlanner",
    "AStarPlanner", "DijkstraPlanner", "APFPlanner", "CHOMPPlanner", "STOMPPlanner",
    "TrajOptPlanner", "ITOMPPlanner", "GPMPPlanner", "PSOPlanner", "GeneticPlanner",
    # gui + entry
    "ImageCanvas", "MainWindow", "main",
]
