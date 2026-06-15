from __future__ import annotations

from typing import Dict, List, Set, Tuple, Type

from .apf import APFPlanner
from .astar import AStarPlanner
from .base import BasePlanner
from .bit_star import BITStarPlanner
from .bitrrt import BiTRRTPlanner
from .chomp import CHOMPPlanner
from .dijkstra import DijkstraPlanner
from .fmt_star import FMTStarPlanner
from .genetic import GeneticPlanner
from .gpmp import GPMPPlanner
from .itomp import ITOMPPlanner
from .kpiece import KPIECEPlanner
from .prm import ClassicPRMPlanner, PRMPlanner
from .pso import PSOPlanner
from .rrt import RRTPlanner
from .rrt_connect import RRTConnectPlanner
from .rrt_star import RRTStarPlanner
from .sbl import SBLPlanner
from .stomp import STOMPPlanner
from .trajopt import TrajOptPlanner

ALGORITHM_GROUPS: List[Tuple[str, List[str]]] = [
    ('Sampling-Based', ['RRT', 'RRT-Connect', 'BiTRRT', 'KPIECE', 'RRT*', 'PRM', 'sPRM', 'SBL', 'FMT*', 'BIT*']),
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
    'PRM': ClassicPRMPlanner,
    'sPRM': PRMPlanner,
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
    'RRT', 'RRT-Connect', 'BiTRRT', 'KPIECE', 'RRT*', 'PRM', 'sPRM', 'SBL', 'FMT*', 'BIT*'
}


ANYTIME_ALGOS: Set[str] = {'RRT*', 'BIT*'}


ALGORITHM_INFO: Dict[str, Tuple[str, str]] = {
    'RRT': (
        "Rapidly-exploring Random Tree (LaValle 1998): the GENERATE_RRT(<i>x</i><sub>init</sub>, <i>K</i>, Δ<i>t</i>) loop specialized to the holonomic model ẋ = u (‖u‖ ≤ 1) with Euler integration, plus a single-query goal bias and goal-region stop.",
        "LaValle, 1998"
    ),
    'RRT-Connect': (
        "Bidirectional RRT (Kuffner & LaValle 2000): one tree takes a single EXTEND step toward a uniform sample over <i>C</i>, the other CONNECTs greedily (repeated EXTEND) to the new vertex and lands exactly on it; the trees swap each iteration and meet at one shared, collision-checked vertex.",
        "Kuffner & LaValle, 2000"
    ),
    'BiTRRT': (
        "Bidirectional Transition-based RRT (Devaurs et al. 2013): the adaptive-temperature transition test (deterministic 0.5 threshold, <i>T</i> scaled by base-2 powers), refinement control, and the downhill-only tree junction (attemptLink within 10·δ), with a clearance-derived cost map for 2D grids.",
        "Devaurs et al., 2013"
    ),
    'KPIECE': (
        "Single-level geometric KPIECE (Sucan & Kavraki 2009): the paper's importance log(<i>I</i>)·score / (<i>S</i>·<i>N</i>·<i>C</i>), the 2<i>n</i> interior/exterior cell rule, a fixed exterior-cell bias, half-normal motion selection, state sampling along motions, boundary-split AddMotion, and the <i>P</i> = α + β·(coverage/dist) progress penalty, on a 2D projection grid.",
        "Sucan & Kavraki, 2009"
    ),
    'RRT*': (
        "RRT* (Karaman & Frazzoli 2011, Alg. 6): ChooseParent + Rewire over the shrinking RGG radius min(γ·(log <i>n</i> / <i>n</i>)<sup>1/2</sup>, step), γ = 2(1 + 1/<i>d</i>)<sup>1/<i>d</i></sup>(μ/ζ)<sup>1/<i>d</i></sup>, for asymptotic optimality; a fixed search radius is available as a legacy option.",
        "Karaman & Frazzoli, 2011"
    ),
    'PRM': (
        "Probabilistic Roadmap, original Kavraki et al. (1996) construction step: random free configs connected within max_edge_dist (capped at <i>k</i> neighbors, increasing distance) by a straight-line local planner, but skipping edges within the same connected component (a cycle-free forest). Not asymptotically optimal; compare with sPRM.",
        "Kavraki et al., 1996"
    ),
    'sPRM': (
        "Simplified PRM (Karaman & Frazzoli 2011): the Kavraki roadmap without the same-component cycle removal, so it keeps cycles. Asymptotically optimal and returns shorter query paths than the original forest PRM.",
        "Karaman & Frazzoli, 2011"
    ),
    'SBL': (
        "Single-query bi-directional lazy planner (Sanchez & Latombe 2001): density-weighted milestone selection (π(m) ∝ 1/η(m)) with shrinking L<sub>∞</sub> neighborhoods B(m, ρ/i), lazy dyadic TEST-SEGMENT (mark safe when 2<sup>−κ</sup>λ &lt; ε), TEST-PATH ordered most-likely-to-collide first, milestone transfer on collision, and the paper's random shortcut optimizer.",
        "Sanchez & Latombe, 2001"
    ),
    'FMT*': (
        "Fast Marching Tree (Janson et al. 2015): forward dynamic-programming wavefront over uniform free-space samples within the shrinking radius <i>r</i><sub>n</sub>, with one-shot lazy collision checking on the single best parent; terminates when the goal is popped as the lowest-cost open node.",
        "Janson et al., 2015"
    ),
    'BIT*': (
        "Batch Informed Trees with ordered vertex and edge queues, rewiring, informed batch sampling, and incumbent-based pruning. Connects within the full RGG radius by default (asymptotically optimal); an optional step-size cap gives a tidier visualization.",
        "Gammell et al., 2015"
    ),
    'A*': (
        "A* (Hart, Nilsson & Raphael 1968): best-first search with <i>f</i> = <i>g</i> + <i>h</i> and an admissible, consistent Euclidean heuristic on the induced 8-connected grid; optimal with respect to that grid discretization.",
        "Hart et al., 1968"
    ),
    'Dijkstra': (
        "Dijkstra's algorithm (1959, Problem 2): uniform-cost search on the induced 8-connected occupancy grid with Euclidean edge weights; optimal with respect to that grid discretization, not the continuous plane.",
        "Dijkstra, 1959"
    ),
    'APF': (
        "Artificial Potential Field (Khatib 1986): parabolic-well attraction plus the FIRAS repulsive force within an influence limit, integrated with Khatib's velocity saturation. Pure APF stalls at local minima; an optional non-paper escape is available.",
        "Khatib, 1986"
    ),
    'CHOMP': (
        "Covariant trajectory optimization (Ratliff et al. 2009) for a 2D point robot: a signed distance field, the workspace cost <i>c</i>(x), the Eq. 4 obstacle functional gradient, and the covariant step ξ ← ξ − (1/λ)A<sup>−1</sup>g over the velocity-prior metric. Also used to post-optimize sampling-based paths.",
        "Ratliff et al., 2009"
    ),
    'STOMP': (
        "Stochastic Trajectory Optimization (Kalakrishnan et al. 2011): noisy rollouts with covariance <i>R</i><sup>−1</sup>, the paper's per-timestep probability update (Eq. 11), the <i>M</i> smoothing projection, and an obstacle cost max(ε − d, 0)·‖ẋ‖ on a signed distance field (Eq. 13), for a 2D point robot.",
        "Kalakrishnan et al., 2011"
    ),
    'TrajOpt': (
        "Trajectory optimization by sequential convex optimization (Schulman et al. 2013): the sum-of-squared-displacements objective with ℓ₁ collision penalties on the linearized signed distance, solved in a trust region (accept iff true/model improvement &gt; <i>c</i>) with an outer penalty loop, for a 2D point robot.",
        "Schulman et al., 2013"
    ),
    'ITOMP': (
        "Incremental covariant trajectory optimization (Park et al. 2012) over a receding execution horizon: acceleration smoothness ½‖AQ‖² (Eq. 6) and a signed-distance obstacle cost max(ε − d, 0)·‖ẋ‖ (Eq. 8), the AᵀA metric preconditioning the update. Static-map adaptation — the dynamic-obstacle cost (Eq. 9) is out of scope (no moving obstacles).",
        "Park et al., 2012"
    ),
    'GPMP': (
        "Gaussian Process Motion Planning (Mukadam, Yan & Boots 2016): a constant-velocity LTI GP prior with GP interpolation, optimized by the covariant gradient update ξ ← ξ − (1/η)K∇U (gradient preconditioned by the GP covariance <i>K</i>, Eq. 24), for a 2D point robot.",
        "Mukadam, Yan & Boots, 2016"
    ),
    'PSO': (
        "Particle Swarm Optimization over waypoint paths. The default velocity update is the exact Kennedy & Eberhart (1995) form v ← v + 2r₁(pbest − x) + 2r₂(gbest − x) with a V<sub>max</sub> clamp and full momentum (no inertia weight, w = 1.0); an off-by-default safeguards toggle adds adaptive inertia/social gain, diversity injection, random immigrants, and swarm restart for cluttered maps. A metaheuristic, not a complete planner.",
        "Kennedy & Eberhart, 1995"
    ),
    'Genetic': (
        "Experimental waypoint-path optimizer based on a Genetic Algorithm.",
        "Holland, 1975"
    ),
}
