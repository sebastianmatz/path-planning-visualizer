# Path Planning Visualizer

Path Planning Visualizer is a desktop application for exploring, comparing, and tuning path-planning algorithms for a 2D point robot on occupancy-grid maps.

Current release status: `0.1.0b3` (`Beta`)

## What's New in `0.1.0b3`

- Added `SBL`, `BiTRRT`, and `KPIECE` as additional sampling-based planners for the 2D occupancy-grid setting
- Reworked `PRM`, `FMT*`, and `BIT*` to be cleaner and closer to their reference formulations within this application's 2D scope
- Improved BIT* live rendering with smaller local connections and smoother display paths for clearer visual feedback

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

- Core Baselines: `A*`, `Dijkstra`, `RRT`, `RRT-Connect`, `BiTRRT`, `KPIECE`, `RRT*`, `PRM`, `SBL`, `FMT*`, `BIT*`, `APF`, `CHOMP`
- Approximate / Experimental Visualizations: `STOMP`, `TrajOpt`, `ITOMP`, `GPMP`, `PSO`, `Genetic`

## Scientific Scope

- The tool is best understood as a teaching and visualization environment, not as a benchmark suite.
- Core baseline planners are implemented in a comparatively direct way for 2D occupancy-grid pathfinding.
- `A*` and `Dijkstra` should be interpreted as optimal on the induced search grid, not on the continuous image plane.
- `BiTRRT` follows the OMPL planner structure closely, but in this application its optimization objective is adapted to a 2D occupancy grid through a clearance-derived cost field rather than an arbitrary user-supplied cost objective.
- `PRM` now follows a clearer two-phase structure with query-independent roadmap construction followed by query-time start/goal attachment.
- `SBL` is implemented as a 2D adaptation with lazy segment validation and lightweight path optimization, not as a full general-configuration-space reproduction of the original system.
- `KPIECE` is implemented here as a single-level geometric adaptation with projection-grid cell selection, state sampling along motions, and progress-based cell penalties, rather than the full multilevel kinodynamic formulation from the original paper.
- `RRT*` is presented here as a practical rewiring-based planner, not as a proof-focused asymptotic-optimality reference implementation.
- `FMT*` is implemented as a cleaner 2D geometric adaptation with uniform free-space samples, open-wavefront parent selection, and one-shot lazy collision checking, rather than the earlier corridor-biased visualization variant.
- `BIT*` is implemented here as a cleaner 2D geometric adaptation with ordered vertex/edge queues, rewiring, informed batches, incumbent-based pruning, and a visualization-oriented local connection cap, rather than a full OMPL-scale implementation of the original planner.
- Several advanced planners, especially `STOMP`, `TrajOpt`, `ITOMP`, and `GPMP`, are visualization-oriented approximations inspired by the cited methods rather than paper-faithful reference implementations.
- The environment model is a binary occupancy map with a point robot, so results do not directly transfer to higher-dimensional robot dynamics or configuration spaces.

## Requirements

- Python 3.11+
- A desktop environment capable of running PyQt6 applications

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python path_planning_visualizer.py
```

To install the project with its console entry point, run:

```
pip install -e .
```

Then you can start it with:

```
path-planning-visualizer
```

## Basic Usage

1. Start the application. A bundled example maze is loaded automatically if available.
2. Click a free white pixel to place the `start`, then click another free white pixel to place the `goal`.
3. Choose an algorithm from the dropdown and adjust its parameters in the left panel.
4. Use `Step` to advance the planner incrementally or `Run` to let it continue automatically.
5. Watch the `Status` panel for path metrics, compute times, and the planner's current status string.
6. For sampling-based planners, you can pause after a valid path is found and trigger `CHOMP Optimize` to smooth the result.
7. Use `Reset` to clear the current run, or change the algorithm/seed to compare different planners on the same map.

## Reading the Status Panel

- `Path length`: geometric length of the final path in pixels
- `Min clearance`: smallest obstacle distance anywhere along the path in pixels
- `Mean clearance`: average obstacle distance along the full path in pixels
- `Smoothness`: average squared turning angle on a resampled path; lower is smoother
- `Compute time to first path`: planner compute time until the first valid solution is found
- `Total compute time`: total accumulated planner compute time for the current run

## Project Layout

- `path_planning_visualizer.py`: desktop application entry point
- `assets/demo.gif`: short application walkthrough for the README
- `assets/maze.png`, `assets/maze 2.png`: bundled example maps
- `pyproject.toml`: package metadata and installable script entry

## Notes

- The application uses `opencv-python-headless` for image processing and `PyQt6` for the UI.
- Example maps are interpreted as binary occupancy grids after grayscale thresholding.

## Versioning

- The project uses semantic versioning with Python-compatible pre-release tags.
- Beta releases follow the pattern `0.1.0b1`, `0.1.0b2`, `0.1.0b3`, and so on.
- The first stable release would be `0.1.0`.
- Bugfix releases after that would continue as `0.1.1`, `0.1.2`, etc.
