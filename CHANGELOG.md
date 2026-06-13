# Changelog

This file tracks release notes for published versions of the project.

## [Unreleased]

## [0.1.0b8] - 2026-06-14

Release-readiness hardening: a single source of truth for the version, a
top-level error handler, behavior-preserving planner speedups, and GUI
integration tests.

### Added

- Top-level GUI exception handler (`app.install_excepthook`): uncaught
  exceptions are printed to stderr and shown in an error dialog (with the full
  traceback under "Show Details") instead of silently aborting the window;
  `tests/test_app.py` covers it
- Headless GUI integration tests (`tests/test_gui_integration.py`): drive the
  real `MainWindow` end to end (off-thread build -> stepping -> collision-free
  path), the map editor (paint / endpoint invalidation), and the save/load
  round-trip

### Changed

- The version is now defined only in `pyproject.toml`. `__version__` is read
  from the installed package metadata (`importlib.metadata`) and the window
  title is derived from it, so the number is no longer hand-copied across the
  package docstring, `__init__`, and the GUI title
- `BIT*` keeps its visualization tree-edge list in sync incrementally and caches
  it, instead of rebuilding the whole edge list from parent pointers on every
  step (identical edges, far less per-step work)
- `PRM` builds its sample pool from the `np.where` coordinate arrays directly
  rather than materializing a Python tuple for every free pixel; the drawn
  samples are byte-for-byte identical for a given seed

## [0.1.0b7] - 2026-06-13

Map-authoring and responsiveness release.

### Added

- Interactive map editor: toggle **Edit Map** to draw obstacles (left-drag) and
  erase them (right-drag) with an adjustable brush; **New Map** starts a blank
  grid and **Save Map** writes the current occupancy grid to a PNG
- Planner construction now runs on a background thread (`gui/worker.py`
  `PlannerBuilder`): heavy setup (PRM/FMT*/BIT* on large maps) shows a
  "Preparing…" state instead of freezing the window
- `path_planning_visualizer.mapping` helpers (`image_to_occupancy`,
  `occupancy_to_image`, `blank_occupancy`, `paint_disk`), re-exported from the
  package, with `tests/test_mapping.py` and a `tests/test_gui_worker.py`
  off-thread-build test
- `LICENSE` (MIT) and expanded `pyproject.toml` metadata (license, author,
  project URLs, classifiers, keywords)

### Changed

- Map loading and saving now share one grayscale<->occupancy convention via
  `image_to_occupancy` / `occupancy_to_image`
- Planner start (`Run`/`Step`) is routed through a single off-thread build
  chokepoint; editing the map or resetting discards any in-flight build

### Fixed

- Corrected the `CHOMP` paper citation in the algorithm info panel to the original
  Ratliff et al. (2009) reference (was Zucker et al., 2013)

## [0.1.0b6] - 2026-06-13

Engineering-hardening release: continuous integration, a correctness + performance
fix in `RRT*`, and stricter linting.

### Added

- Continuous integration via GitHub Actions (`.github/workflows/ci.yml`): runs
  `ruff`, the full `pytest` suite, and a benchmark smoke check on Python 3.11 and
  3.12 for every push and pull request; added a CI status badge to the README

### Changed

- `RRT*` now maintains an explicit children adjacency and propagates rewiring cost
  updates over the affected subtree only (was an O(n^2) scan of all parent pointers
  on every rewire)
- Enforced the full Pyflakes rule set in `ruff` (removed the `F841` ignore) and
  cleaned up the remaining unused locals in `bit_star`, `chomp`, and `rrt_connect`

### Fixed

- `RRT*` returned-path cost (and the reported `best_goal_cost`) could increase
  between steps: after an ancestor rewire lowered the goal's true cost, a later
  goal reconnection that beat the now-stale incumbent could reconnect to a worse
  path. The incumbent is now kept in sync with the goal's true cost during
  propagation, so `extract_path()` cost is monotone non-increasing

## [0.1.0b5] - 2026-06-13

This release focuses on paper fidelity and academic evidence. Several planners
now implement the defining mechanism of their source paper by default; the
changes to `RRT*` and `BIT*` alter their default behavior (see Changed).

### Added

- Shared `rgg_radius` helper (`planners/_rgg.py`) for the shrinking
  Random-Geometric-Graph connection radius, reused by `FMT*`, `BIT*`, and `RRT*`;
  a `tests/test_rgg.py` equivalence suite proves it reproduces the previous
  `FMT*`/`BIT*` formulas byte-for-byte
- Shared trajectory-optimizer utilities (`planners/_trajectory.py`): straight-line /
  obstacle-escape initialization, the finite-difference acceleration / smoothness
  metric `A^T A`, and an SDF-gradient sampler, reused by the trajectory optimizers
- Reproducible benchmark CLI `python -m path_planning_visualizer.benchmark`
  (success rate, path length, clearance, compute time, collision checks across
  planners, maps, and seeds; `--planners`, `--maps`, `--seeds`, `--csv`)
- `tests/test_optimality.py`: anytime-improvement and near-optimality evidence for
  `RRT*`/`BIT*`/`FMT*`/`PRM`, adaptive-`RRT*` determinism, and convergence /
  determinism of the four trajectory optimizers

### Changed

- `RRT*` now uses the shrinking RGG radius `min(gamma*(log n/n)^(1/2), step)` by
  default (asymptotically optimal, Karaman & Frazzoli 2011); the search-radius
  field accepts `0 = auto`, and a positive value selects the previous fixed-radius
  behavior
- `BIT*` now connects within the full RGG radius by default (asymptotically
  optimal, Gammell et al. 2015); the former step-size connection cap is now an
  optional "Cap edges at step size (visualization)" toggle, off by default
- `STOMP` reworked to the faithful algorithm (Kalakrishnan et al. 2011): smooth
  noise with covariance `R^-1`, per-timestep probability-weighted updates, and the
  `M` smoothing projection (was a single scalar cost per rollout)
- `TrajOpt` reworked into sequential convex optimization (Schulman et al. 2014):
  l1 collision penalties on the linearized signed distance, trust-region accept /
  reject, and an outer penalty loop (was plain clipped gradient descent)
- `ITOMP` reworked into a CHOMP-style covariant optimizer over a receding
  execution horizon (Park et al. 2012); static-map adaptation
- `GPMP` reworked to GPMP2 style (Mukadam et al. 2016): constant-velocity LTI GP
  prior on `[position, velocity]` states, GP-interpolated obstacle factors, and
  Gauss-Newton / Levenberg-Marquardt MAP inference (was position-only gradient
  descent)
- Regrouped the planner menu: the trajectory optimizers (`CHOMP`, `STOMP`,
  `TrajOpt`, `ITOMP`, `GPMP`) now share the "Trajectory Optimization" group, and
  `PSO`/`Genetic` move to a "Metaheuristic" group; updated the in-app algorithm
  descriptions accordingly

## [0.1.0b4] - 2026-06-09

### Added

- Automated `pytest` suite covering planner soundness (any reported path is collision-free and connects `start` to `goal`), completeness of the graph and sampling-based planners, and `A*`/`Dijkstra` cost agreement on the same induced grid
- A dependency-free `GridIndex` spatial index used for nearest-neighbor and radius queries in `RRT`, `RRT-Connect`, `RRT*`, and `BiTRRT` (replaces the per-step O(n) array rebuild + scan); exactness and seeded-determinism tests guard it

### Changed

- Restructured the single `path_planning_visualizer.py` module into an installable `path_planning_visualizer` package (`geometry`, `metrics`, `planners/`, `gui/`, `app`); the app now launches with `python -m path_planning_visualizer`
- Bundled the example maps as package data under `path_planning_visualizer/assets/`
- Added a shared `make_distance_field` helper that removes the repeated obstacle-distance-field construction across planners
- Aligned the `PRM` `k_neighbors` default with its parameter widget
- Added lower-bound version floors for `numpy`, `opencv-python-headless`, and `PyQt6` in `pyproject.toml` and `requirements.txt`
- Removed unused imports across the package (`ruff` F401) and added `ruff` to the `dev` extra, with a `[tool.ruff.lint]` config (pyflakes; `F841` deferred)
- Factored the shared arclength-resampling core in `geometry.py`, reused by the resample helpers and `CHOMP`'s path resampling
- Cut the test suite's planner budgets so it runs in roughly half the time (~3.5 min) with the same coverage
- Minor code cleanups: corrected an `Optional[str]` annotation, removed unreachable visualization-edge branches in `CHOMP`/`STOMP`, and standardized a `get_params` return annotation
- Reworked `RRT` into a single configurable planner with an OMPL-style `goal_bias` parameter instead of maintaining a separate goal-biased UI variant
- Updated `RRT` to use the paper's `GENERATE_RRT(x_init, K, Delta t)` structure in a clearer 2D holonomic occupancy-grid adaptation
- Improved `RRT` path presentation with a smoother display-only rendering of the current/final solution
- Clarified the `RRT` algorithm description in the UI and README, including the role of configurable goal bias
- Reworked `CHOMP` into a more faithful 2D point-robot specialization with signed-distance interpolation, CHOMP-style covariant preconditioning, and functional obstacle gradients
- Reworked `CHOMP` visualization to show full-trajectory deformation per iteration, including recent trajectory history and previous-to-current correspondence cues
- Retuned interactive `CHOMP` optimization defaults and stopping behavior for faster GUI feedback without changing the underlying objective

### Fixed

- `RRT-Connect` now joins its two trees at an exact, collision-checked vertex, removing a possible unchecked gap in the returned path
- `A*` / `Dijkstra` no longer report "No Path" when a free `start`/`goal` falls in a coarse cell that also contains an obstacle pixel, and their returned paths now connect to the exact clicked `start`/`goal`
- `steer` now rounds to the nearest pixel instead of truncating
- Added the missing `List`/`StepResult` imports referenced by `MainWindow` type annotations (no runtime effect under `from __future__ import annotations`, but now statically correct)
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
