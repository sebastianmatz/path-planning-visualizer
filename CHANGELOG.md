# Changelog

This file tracks release notes for published versions of the project.

## [Unreleased]

### Changed

- Reworked `RRT` into a single configurable planner with an OMPL-style `goal_bias` parameter instead of maintaining a separate goal-biased UI variant
- Updated `RRT` to use the paper's `GENERATE_RRT(x_init, K, Delta t)` structure in a clearer 2D holonomic occupancy-grid adaptation
- Improved `RRT` path presentation with a smoother display-only rendering of the current/final solution
- Clarified the `RRT` algorithm description in the UI and README, including the role of configurable goal bias
- Reworked `CHOMP` into a more faithful 2D point-robot specialization with signed-distance interpolation, CHOMP-style covariant preconditioning, and functional obstacle gradients
- Reworked `CHOMP` visualization to show full-trajectory deformation per iteration, including recent trajectory history and previous-to-current correspondence cues
- Retuned interactive `CHOMP` optimization defaults and stopping behavior for faster GUI feedback without changing the underlying objective

### Fixed

- Prevented duplicate or null `RRT` vertex insertions caused by continuous-to-grid discretization
- Fixed `RRT` goal-region handling, including immediate success when `start` already lies inside the goal region
- Fixed `RRT` vertex-budget termination so the planner now stops cleanly at the configured `K` limit
- Corrected `RRT` rejection highlighting so failed expansion attempts mark the rejected extension point more accurately
- Fixed `CHOMP` cost/gradient inconsistencies by aligning the optimized objective with the reported smoothness and obstacle terms
- Fixed `CHOMP Optimize` so the original sampled path remains visible while the optimizer runs on top of it

## [0.1.0b3] - 2026-04-09

### Added

- Added `SBL` as a bidirectional lazy roadmap-style planner for the 2D occupancy-grid setting
- Added `BiTRRT` as an OMPL-inspired bidirectional transition-based RRT with a clearance-derived cost map
- Added `KPIECE` as a single-level geometric adaptation with projection-grid cell exploration for 2D maps

### Changed

- Reworked `PRM` into a cleaner two-phase formulation with query-independent roadmap construction and query-time start/goal attachment
- Reworked `FMT*` into a cleaner 2D geometric adaptation with uniform free-space sampling, open-wavefront parent selection, and one-shot lazy collision checking
- Reworked `BIT*` with ordered vertex and edge queues, rewiring, informed batch sampling, incumbent-based pruning, smaller local connection control, and cleaner live/final rendering
- Improved `KPIECE` by adding motion-based state selection, progress-based cell penalties, cell-boundary motion splitting, and later runtime-focused incremental bookkeeping
- Clarified algorithm descriptions in the UI and README, especially for grid-optimality claims and approximate or adapted planners

### Fixed

- Corrected `SBL` toward a cleaner paper-aligned 2D adaptation, including `L-infinity` neighborhood logic and a lighter random path optimizer
- Fixed PRM query connectivity so `start` and `goal` are attached correctly after roadmap construction
- Improved BIT* visualization so rewiring history no longer accumulates as misleading permanent tree artifacts

## [0.1.0b2] - 2026-04-08

### Changed

- Expanded path-quality reporting with minimum clearance, mean clearance, and smoothness metrics
- Clarified compute-time reporting by separating time to first path from total compute time
- Improved CHOMP path selection and final trajectory smoothing for more stable post-optimization results
- Tightened A* and Dijkstra grid handling, including safer diagonal motion near obstacle corners
- Reworked GPMP into a deterministic local optimizer without external warm-start heuristics or seed-driven behavior
- Expanded the README with a short usage walkthrough and status-panel explanation

## [0.1.0b1] - 2026-04-07

Initial beta release.

### Included functionality

- Desktop application for interactive path-planning visualization on occupancy-grid maps
- Focus on 2D point-robot planning in binary occupancy-grid environments
- Click-to-place start and goal points on loaded example or user-provided map images
- Parameter configuration and visualization for sampling-based, graph-search, potential-field, trajectory-optimization, and metaheuristic planners
- Step-by-step execution, continuous playback, and live status display during planning
- Geometric path metrics for length, minimum clearance, mean clearance, and smoothness
- Compute-time-to-first-path and total-compute-time tracking, plus reproducible seeds for stochastic planners
- Bundled example maps under `assets/`
- Basic packaging and project metadata via `pyproject.toml`, `README.md`, and `.gitignore`
