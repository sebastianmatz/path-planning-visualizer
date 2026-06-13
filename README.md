# Path Planning Visualizer

[![CI](https://github.com/sebastianmatz/path-planning-visualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastianmatz/path-planning-visualizer/actions/workflows/ci.yml)

Path Planning Visualizer is a desktop application for exploring, comparing, and tuning path-planning algorithms for a 2D point robot on occupancy-grid maps.

Current release status: `0.1.0b7` (`Beta`)

## What's New in `0.1.0b7`

- Interactive map editor: draw obstacles (left-drag) and erase (right-drag) with an adjustable brush, start a blank map with **New Map**, and **Save Map** to a PNG
- Planner construction now runs off the UI thread, so heavy setup (PRM/FMT*/BIT* on large maps) shows a "Preparing…" state instead of freezing the window
- New `path_planning_visualizer.mapping` helpers (load/save/brush) with tests

## What's New in `0.1.0b6`

- Added continuous integration (GitHub Actions): `ruff` + the full `pytest` suite + a benchmark smoke run on every push and pull request
- `RRT*` cost propagation is now O(affected subtree) via a children adjacency, and the returned path cost is now monotone non-increasing (a stale-incumbent goal reconnection could previously make it rise)
- Enforced the full Pyflakes rule set (`F841` unused-locals no longer ignored)

## What's New in `0.1.0b5`

- `RRT*` now uses the shrinking RGG radius `min(gamma*(log n/n)^(1/2), step)` by default, making it asymptotically optimal (Karaman & Frazzoli 2011); set the search radius to `0` for this auto mode or a positive value for the legacy fixed radius
- `BIT*` now connects within the full RGG radius by default (asymptotically optimal, Gammell et al. 2015); the previous step-size connection cap is now an optional visualization toggle
- `STOMP`, `TrajOpt`, `ITOMP`, and `GPMP` were reworked to implement the defining mechanism of their source papers (per-timestep STOMP updates, TrajOpt sequential convex optimization, incremental covariant ITOMP, and GPMP2-style GP + Gauss-Newton) for a 2D point robot
- Added a reproducible benchmark CLI: `python -m path_planning_visualizer.benchmark`
- Added optimality/anytime-improvement evidence tests and a shared RGG-radius helper used by `FMT*`, `BIT*`, and `RRT*`

## Preview

The demo below shows the typical workflow: loading a map, placing start and goal points, adjusting planner settings, and running the visualization.

![Path Planning Visualizer demo showing map setup and planner execution](assets/demo.gif)

## Features

- Interactive map loading with click-to-set start and goal points
- Side-by-side parameter configuration for multiple planner families
- Step mode and continuous playback for algorithm visualization
- Geometric path metrics including path length, minimum clearance, mean clearance, and smoothness
- Compute-time tracking for time-to-first-path and total run time
- Reproducible seeds for stochastic planners
- Built-in example maps for quick testing

## Included Algorithms

- Sampling-Based: `RRT`, `RRT-Connect`, `BiTRRT`, `KPIECE`, `RRT*`, `PRM`, `SBL`, `FMT*`, `BIT*`
- Graph Search: `A*`, `Dijkstra`
- Potential Field: `APF`
- Trajectory Optimization: `CHOMP`, `STOMP`, `TrajOpt`, `ITOMP`, `GPMP`
- Metaheuristic: `PSO`, `Genetic`

## Scientific Scope

- The tool is best understood as a teaching and visualization environment, not as a benchmark suite.
- Core baseline planners are implemented in a comparatively direct way for 2D occupancy-grid pathfinding.
- `A*` and `Dijkstra` should be interpreted as optimal on the induced search grid, not on the continuous image plane.
- `RRT` follows the paper's `GENERATE_RRT(x_init, K, Delta t)` structure in a 2D holonomic specialization: uniform bounded workspace sampling, Euclidean nearest-neighbor search, explicit `SELECT_INPUT/NEW_STATE` semantics via `x_dot = u`, a fixed vertex budget `K`, and an optional OMPL-style `goal_bias` parameter for goal-directed sampling.
- `CHOMP` is implemented as a 2D point-robot specialization with a signed distance field, covariant preconditioned updates, and the functional obstacle-gradient structure from the original method, rather than a full articulated-robot configuration-space system.
- `BiTRRT` follows the OMPL planner structure closely, but in this application its optimization objective is adapted to a 2D occupancy grid through a clearance-derived cost field rather than an arbitrary user-supplied cost objective.
- `PRM` now follows a clearer two-phase structure with query-independent roadmap construction followed by query-time start/goal attachment.
- `SBL` is implemented as a 2D adaptation with lazy segment validation and lightweight path optimization, not as a full general-configuration-space reproduction of the original system.
- `KPIECE` is implemented here as a single-level geometric adaptation with projection-grid cell selection, state sampling along motions, and progress-based cell penalties, rather than the full multilevel kinodynamic formulation from the original paper.
- `RRT*` uses the shrinking RGG radius `min(gamma*(log n/n)^(1/2), step)` by default, which makes it asymptotically optimal in the sense of Karaman & Frazzoli (2011); a fixed search radius remains selectable for comparison.
- `FMT*` is implemented as a cleaner 2D geometric adaptation with uniform free-space samples, open-wavefront parent selection, one-shot lazy collision checking, and the paper's shrinking connection radius.
- `BIT*` connects within the full RGG radius by default (asymptotically optimal, Gammell et al. 2015) with ordered vertex/edge queues, rewiring, informed batches, and incumbent-based pruning; an optional step-size cap is available purely for tidier visualization.
- The trajectory optimizers implement the defining mechanism of their source papers for a 2D point robot: `CHOMP` (covariant gradient on a signed distance field), `STOMP` (smooth noisy rollouts with per-timestep probability-weighted updates and the M projection), `TrajOpt` (sequential convex optimization with l1 collision penalties and trust regions), `ITOMP` (incremental covariant optimization over a receding execution horizon; static-map adaptation), and `GPMP` (GPMP2-style LTI GP prior with GP-interpolated obstacle factors and Gauss-Newton inference). They are local optimizers and can fail on problems that require a large topological change from the straight-line initialization.
- The environment model is a binary occupancy map with a point robot, so results do not directly transfer to higher-dimensional robot dynamics or configuration spaces.
- For a reproducible cross-planner comparison (success rate, path length, clearance, time, collision checks) run `python -m path_planning_visualizer.benchmark`.

## Requirements

- Python 3.11+
- A desktop environment capable of running PyQt6 applications

## Setup

1. Create and activate a virtual environment:

   ```
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install the dependencies, either directly:

   ```
   pip install -r requirements.txt
   ```

   or as an editable install (this also registers the `path-planning-visualizer` console command):

   ```
   pip install -e .
   ```

## Run

Start the application with:

```
python -m path_planning_visualizer
```

If you installed it with `pip install -e .`, you can instead run the console command:

```
path-planning-visualizer
```

> Note: the project is now a package, so the old `python path_planning_visualizer.py` command is
> replaced by `python -m path_planning_visualizer`.

## Development

Install the dev extras and run the test suite:

```
pip install -e .[dev]
pytest
```

Run the cross-planner benchmark (headless, no GUI):

```
python -m path_planning_visualizer.benchmark
python -m path_planning_visualizer.benchmark --planners "A*,RRT*,BIT*" --maps all --seeds 5 --csv results.csv
```

## Basic Usage

1. Start the application. A bundled example maze is loaded automatically if available.
2. Click a free white pixel to place the `start`, then click another free white pixel to place the `goal`.
3. Choose an algorithm from the dropdown and adjust its parameters in the left panel.
4. Use `Step` to advance the planner incrementally or `Run` to let it continue automatically.
5. Watch the `Status` panel for path metrics, compute times, and the planner's current status string.
6. For sampling-based planners, you can pause after a valid path is found and trigger `CHOMP Optimize` to smooth the result.
7. Use `Reset` to clear the current run, or change the algorithm/seed to compare different planners on the same map.

To author your own map, toggle `Edit Map` and left-drag to draw obstacles or right-drag to erase (the `Brush` spin sets the radius). `New Map` starts from a blank grid and `Save Map` writes the current map to a PNG you can reload later. On large maps the planner is built in the background, so the window stays responsive and shows a brief `Preparing…` state before it starts.

## Reading the Status Panel

- `Path length`: geometric length of the final path in pixels
- `Min clearance`: smallest obstacle distance anywhere along the path in pixels
- `Mean clearance`: average obstacle distance along the full path in pixels
- `Smoothness`: average squared turning angle on a resampled path; lower is smoother
- `Compute time to first path`: planner compute time until the first valid solution is found
- `Total compute time`: total accumulated planner compute time for the current run

## Project Layout

- `path_planning_visualizer/`: application package
  - `app.py`, `__main__.py`: entry point (`python -m path_planning_visualizer`)
  - `geometry.py`, `metrics.py`: shared geometry, collision, and path-metric helpers
  - `planners/`: one module per planner, plus the planner `registry` and shared helpers (`_spatial`, `_rgg`, `_trajectory`)
  - `benchmark.py`: headless cross-planner benchmark CLI
  - `gui/`: the `ImageCanvas` and the `MainWindow`
  - `mapping.py`: occupancy-grid load/save/brush helpers used by the map editor
  - `assets/maze.png`, `assets/maze 2.png`, `assets/maze 3.png`: bundled example maps
- `tests/`: pytest suite
- `assets/demo.gif`: short application walkthrough for the README
- `pyproject.toml`: package metadata and installable script entry
- `LICENSE`: MIT license

## Notes

- The application uses `opencv-python-headless` for image processing and `PyQt6` for the UI.
- Example maps are interpreted as binary occupancy grids after grayscale thresholding.

## Versioning

- The project uses semantic versioning with Python-compatible pre-release tags.
- Beta releases follow the pattern `0.1.0b1`, `0.1.0b2`, `0.1.0b3`, `0.1.0b4`, `0.1.0b5`, `0.1.0b6`, `0.1.0b7`, and so on.
- The first stable release would be `0.1.0`.
- Bugfix releases after that would continue as `0.1.1`, `0.1.2`, etc.

## License

MIT — see [LICENSE](LICENSE).
