"""
Path Planning Visualizer
========================

Interactive desktop application for exploring and comparing path-planning
algorithms on occupancy-grid maps.

Supported algorithms:
- Sampling-Based: RRT, RRT-Connect, BiTRRT, KPIECE, RRT*, PRM, sPRM, SBL, FMT*, BIT*
- Graph Search: A*, Dijkstra
- Potential Field: APF
- Trajectory Optimization: CHOMP, STOMP, TrajOpt, ITOMP, GPMP
- Metaheuristic: PSO

Usage:
    python -m path_planning_visualizer
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# The version lives in exactly one place: the ``version`` field of
# ``pyproject.toml``. Installing the package (``pip install -e .``) records it as
# package metadata, which we read back here so nothing else has to hardcode it.
try:
    __version__ = _pkg_version("path-planning-visualizer")
except PackageNotFoundError:  # running from a raw source tree without an install
    __version__ = "0.0.0+dev"

from .geometry import (
    bilinear_sample_scalar,
    bilinear_sample_scalar_batch,
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
from .mapping import (
    blank_occupancy,
    image_to_occupancy,
    occupancy_to_image,
    paint_disk,
)
from .metrics import (
    PathMetrics,
    compute_path_mean_clearance,
    compute_path_metrics,
    compute_path_min_clearance,
    compute_path_smoothness,
)
from .planners import (
    APFPlanner,
    AStarPlanner,
    BiTRRTPlanner,
    BITStarPlanner,
    CHOMPPlanner,
    ClassicPRMPlanner,
    DijkstraPlanner,
    FMTStarPlanner,
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
from .planners.base import BasePlanner, StepResult
from .planners.registry import (
    ALGORITHM_GROUPS,
    ALGORITHM_INFO,
    ANYTIME_ALGOS,
    AVAILABLE_PLANNERS,
    SAMPLING_BASED_ALGOS,
)
from .types import Edge, FloatPoint, OccupancyGrid, Point

# GUI and entry-point symbols are imported lazily (PEP 562) so that
# ``import path_planning_visualizer`` -- and the headless benchmark/library use --
# does not pull in PyQt6. They resolve on first attribute access.
_LAZY_ATTRS = {
    "ImageCanvas": (".gui.canvas", "ImageCanvas"),
    "MainWindow": (".gui.main_window", "MainWindow"),
    "main": (".app", "main"),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(target[0], __name__)
    return getattr(module, target[1])


def __dir__():
    return sorted([*globals(), *_LAZY_ATTRS])

__all__ = [
    "__version__",
    # types
    "Point", "FloatPoint", "Edge", "OccupancyGrid",
    # geometry
    "dist", "l1_dist", "linf_dist", "round_point", "select_holonomic_input",
    "integrate_holonomic_state", "steer", "clamp_point", "make_distance_field",
    "bilinear_sample_scalar", "bilinear_sample_scalar_batch", "bilinear_sample_vector", "segment_points",
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
    "RRTStarPlanner", "PRMPlanner", "ClassicPRMPlanner", "SBLPlanner", "FMTStarPlanner", "BITStarPlanner",
    "AStarPlanner", "DijkstraPlanner", "APFPlanner", "CHOMPPlanner", "STOMPPlanner",
    "TrajOptPlanner", "ITOMPPlanner", "GPMPPlanner", "PSOPlanner",
    # gui + entry
    "ImageCanvas", "MainWindow", "main",
]
