from __future__ import annotations

from typing import Dict, List, Set, Tuple, Type

from .base import BasePlanner
from .rrt import RRTPlanner
from .rrt_connect import RRTConnectPlanner
from .bitrrt import BiTRRTPlanner
from .kpiece import KPIECEPlanner
from .rrt_star import RRTStarPlanner
from .prm import PRMPlanner
from .sbl import SBLPlanner
from .fmt_star import FMTStarPlanner
from .bit_star import BITStarPlanner
from .astar import AStarPlanner
from .dijkstra import DijkstraPlanner
from .apf import APFPlanner
from .chomp import CHOMPPlanner
from .stomp import STOMPPlanner
from .trajopt import TrajOptPlanner
from .itomp import ITOMPPlanner
from .gpmp import GPMPPlanner
from .pso import PSOPlanner
from .genetic import GeneticPlanner


ALGORITHM_GROUPS: List[Tuple[str, List[str]]] = [
    ('Sampling-Based', ['RRT', 'RRT-Connect', 'BiTRRT', 'KPIECE', 'RRT*', 'PRM', 'SBL', 'FMT*', 'BIT*']),
    ('Graph Search', ['A*', 'Dijkstra']),
    ('Potential Field', ['APF']),
    ('Trajectory Optimization', ['CHOMP', 'STOMP', 'TrajOpt', 'ITOMP', 'GPMP']),
    ('Metaheuristic', ['PSO', 'Genetic']),
]


AVAILABLE_PLANNERS: Dict[str, Type[BasePlanner]] = {
    'RRT': RRTPlanner,
    'RRT-Connect': RRTConnectPlanner,
    'BiTRRT': BiTRRTPlanner,
    'KPIECE': KPIECEPlanner,
    'RRT*': RRTStarPlanner,
    'PRM': PRMPlanner,
    'SBL': SBLPlanner,
    'FMT*': FMTStarPlanner,
    'BIT*': BITStarPlanner,
    'A*': AStarPlanner,
    'Dijkstra': DijkstraPlanner,
    'APF': APFPlanner,
    'CHOMP': CHOMPPlanner,
    'STOMP': STOMPPlanner,
    'TrajOpt': TrajOptPlanner,
    'ITOMP': ITOMPPlanner,
    'GPMP': GPMPPlanner,
    'PSO': PSOPlanner,
    'Genetic': GeneticPlanner,
}


SAMPLING_BASED_ALGOS: Set[str] = {
    'RRT', 'RRT-Connect', 'BiTRRT', 'KPIECE', 'RRT*', 'PRM', 'SBL', 'FMT*', 'BIT*'
}


ANYTIME_ALGOS: Set[str] = {'RRT*', 'BIT*'}


ALGORITHM_INFO: Dict[str, Tuple[str, str]] = {
    'RRT': (
        "Rapidly-exploring Random Tree implementation after LaValle (1998) for a 2D occupancy-grid setting with configurable goal bias.",
        "LaValle, 1998"
    ),
    'RRT-Connect': (
        "Bidirectional RRT growing two trees from start and goal, connecting when they meet.",
        "Kuffner & LaValle, 2000"
    ),
    'BiTRRT': (
        "OMPL-style bidirectional Transition-based RRT with transition tests, frontier control, and a clearance-derived cost map adaptation for 2D occupancy grids.",
        "Devaurs et al., 2013 / OMPL"
    ),
    'KPIECE': (
        "Single-level geometric KPIECE adaptation using a 2D projection grid, border-cell preference, half-normal motion selection within cells, state sampling along motions, and progress-based score penalties.",
        "Sucan & Kavraki, 2008"
    ),
    'RRT*': (
        "RRT with rewiring and incremental path improvement. Uses the shrinking RGG radius min(gamma*(log n/n)^(1/2), step) by default for asymptotic optimality; a fixed search radius is available as a legacy option.",
        "Karaman & Frazzoli, 2011"
    ),
    'PRM': (
        "Probabilistic Roadmap with a query-independent learning phase and a query phase that connects start and goal to the learned roadmap before graph search.",
        "Kavraki et al., 1996"
    ),
    'SBL': (
        "Single-query bi-directional lazy roadmap planner using an L-infinity neighborhood metric, deferred segment validation, and the paper's lightweight random path optimizer.",
        "Sanchez & Latombe, 2001"
    ),
    'FMT*': (
        "Fast Marching Tree using uniform free-space sampling, open-set parent selection, and one-shot lazy collision checking per candidate connection in a 2D occupancy-grid adaptation.",
        "Janson et al., 2015"
    ),
    'BIT*': (
        "Batch Informed Trees with ordered vertex and edge queues, rewiring, informed batch sampling, and incumbent-based pruning. Connects within the full RGG radius by default (asymptotically optimal); an optional step-size cap gives a tidier visualization.",
        "Gammell et al., 2015"
    ),
    'A*': (
        "Classic heuristic graph search on the induced occupancy grid. Optimal only with respect to that grid discretization.",
        "Hart et al., 1968"
    ),
    'Dijkstra': (
        "Uniform-cost graph search on the induced occupancy grid. Optimal only with respect to that grid discretization.",
        "Dijkstra, 1959"
    ),
    'APF': (
        "Artificial Potential Field. Goal attracts, obstacles repel. Fast but can get stuck.",
        "Khatib, 1986"
    ),
    'CHOMP': (
        "Covariant trajectory optimization for a 2D point robot using a signed distance field, functional obstacle gradients, and a CHOMP-style preconditioned update.",
        "Ratliff et al., 2009"
    ),
    'STOMP': (
        "Stochastic Trajectory Optimization: smooth noisy rollouts with covariance R^-1, per-timestep probability-weighted updates, and the M smoothing projection, for a 2D point robot.",
        "Kalakrishnan et al., 2011"
    ),
    'TrajOpt': (
        "Trajectory optimization by sequential convex optimization: l1 collision penalties on the linearized signed distance, solved inside a trust region with an outer penalty loop, for a 2D point robot.",
        "Schulman et al., 2014"
    ),
    'ITOMP': (
        "Incremental covariant trajectory optimization over a receding execution horizon; the A^T A smoothness metric preconditions the obstacle gradient. Static-map adaptation (no dynamic obstacles).",
        "Park et al., 2012"
    ),
    'GPMP': (
        "Gaussian Process Motion Planning (GPMP2 style): constant-velocity LTI GP prior, GP-interpolated obstacle factors, and Gauss-Newton/Levenberg-Marquardt MAP inference, for a 2D point robot.",
        "Mukadam et al., 2016"
    ),
    'PSO': (
        "Experimental waypoint-path optimizer based on Particle Swarm Optimization.",
        "Kennedy & Eberhart, 1995"
    ),
    'Genetic': (
        "Experimental waypoint-path optimizer based on a Genetic Algorithm.",
        "Holland, 1975"
    ),
}
