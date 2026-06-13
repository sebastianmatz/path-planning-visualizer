# Path Planning Visualizer

[![CI](https://github.com/sebastianmatz/path-planning-visualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastianmatz/path-planning-visualizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta%20(0.1.0b8)-orange)

An interactive desktop application for **exploring, comparing, and tuning path-planning algorithms** for a 2D point robot on occupancy-grid maps. It bundles 19 planners — sampling-based, graph-search, potential-field, trajectory-optimization, and metaheuristic — behind one UI, with step-through visualization, live path metrics, an interactive map editor, and a reproducible headless benchmark.

It is built as a **teaching and visualization environment**: each planner follows the defining mechanism of its source paper, adapted to a 2D occupancy grid, with the adaptations stated explicitly (see [Scientific scope](#scientific-scope)).

![Path Planning Visualizer demo showing map setup and planner execution](assets/demo.gif)

## Contents

- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Algorithms](#algorithms)
- [Scientific scope](#scientific-scope)
- [Benchmarking](#benchmarking)
- [Status panel reference](#status-panel-reference)
- [Development](#development)
- [Project layout](#project-layout)
- [Versioning](#versioning)
- [License](#license)

## Quick start

Requirements: **Python 3.11+** and a desktop environment capable of running PyQt6.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install (editable install also registers the console command)
pip install -e .

# 3. run
python -m path_planning_visualizer
```

After `pip install -e .` you can also launch it with the console command:

```bash
path-planning-visualizer
```

> The project is a package, so the old `python path_planning_visualizer.py` entry point has been
> replaced by `python -m path_planning_visualizer`. To install runtime dependencies without the
> package, use `pip install -r requirements.txt`.

## Using the app

1. A bundled example maze loads automatically on startup.
2. Click a free (white) pixel to place the **start**, then another to place the **goal**.
3. Pick an algorithm from the grouped dropdown and adjust its parameters in the left panel.
4. **Step** advances the planner incrementally; **Run** plays it continuously; **Reset** clears the run.
5. The **Status** panel shows live path metrics, compute times, and the planner's status string.
6. For sampling-based planners, pause once a path is found and click **CHOMP Optimize** to smooth it.

**Editing maps.** Toggle **Edit Map** and left-drag to draw obstacles or right-drag to erase (the
**Brush** spin sets the radius). **New Map** starts from a blank grid; **Save Map** writes the current
grid to a PNG you can reload later. On large maps the planner is built on a background thread, so the
window stays responsive and shows a brief *Preparing…* state before it starts.

## Algorithms

19 planners across five families. Each row links the planner to the source paper it is modeled on
(the same citation shown in the app's info panel).

| Family | Algorithm | Reference |
| --- | --- | --- |
| Sampling-based | `RRT` | LaValle, 1998 |
| Sampling-based | `RRT-Connect` | Kuffner & LaValle, 2000 |
| Sampling-based | `BiTRRT` | Devaurs et al., 2013 / OMPL |
| Sampling-based | `KPIECE` | Şucan & Kavraki, 2008 |
| Sampling-based | `RRT*` | Karaman & Frazzoli, 2011 |
| Sampling-based | `PRM` | Kavraki et al., 1996 |
| Sampling-based | `SBL` | Sánchez & Latombe, 2001 |
| Sampling-based | `FMT*` | Janson et al., 2015 |
| Sampling-based | `BIT*` | Gammell et al., 2015 |
| Graph search | `A*` | Hart et al., 1968 |
| Graph search | `Dijkstra` | Dijkstra, 1959 |
| Potential field | `APF` | Khatib, 1986 |
| Trajectory optimization | `CHOMP` | Ratliff et al., 2009 |
| Trajectory optimization | `STOMP` | Kalakrishnan et al., 2011 |
| Trajectory optimization | `TrajOpt` | Schulman et al., 2014 |
| Trajectory optimization | `ITOMP` | Park et al., 2012 |
| Trajectory optimization | `GPMP` | Mukadam et al., 2016 |
| Metaheuristic | `PSO` | Kennedy & Eberhart, 1995 |
| Metaheuristic | `Genetic` | Holland, 1975 |

## Scientific scope

This is a teaching and visualization tool, **not a benchmark suite**. The environment is a binary
occupancy map with a holonomic 2D point robot, so results do not directly transfer to higher
dimensional robot dynamics or configuration spaces. Graph-search planners are optimal only on the
induced search grid, and the trajectory optimizers are *local* methods that can fail when a problem
requires a large topological change away from the straight-line initialization.

Within that scope, each planner implements the defining mechanism of its paper. The adaptations are:

| Planner | Adaptation / fidelity note |
| --- | --- |
| `RRT` | Follows the paper's `GENERATE_RRT(x_init, K, Δt)` structure as a 2D holonomic specialization: uniform sampling, Euclidean nearest-neighbor, explicit `SELECT_INPUT`/`NEW_STATE` (`ẋ = u`), a fixed vertex budget `K`, and an optional OMPL-style `goal_bias`. |
| `RRT*` | Shrinking RGG radius `min(γ·(log n / n)^(1/2), step)` by default → asymptotically optimal (Karaman & Frazzoli, 2011); a fixed search radius is selectable for comparison. |
| `BIT*` | Connects within the full RGG radius by default → asymptotically optimal (Gammell et al., 2015), with ordered vertex/edge queues, rewiring, informed batches, and incumbent pruning; an optional step-size cap is purely for tidier visuals. |
| `FMT*` | 2D geometric adaptation: uniform free-space samples, open-wavefront parent selection, one-shot lazy collision checking, and the paper's shrinking connection radius. |
| `PRM` | Two-phase: query-independent roadmap construction followed by query-time start/goal attachment. |
| `SBL` | 2D adaptation with an L∞ neighborhood, lazy segment validation, and a lightweight random path optimizer (not a full configuration-space reproduction). |
| `KPIECE` | Single-level *geometric* adaptation with projection-grid cell selection and progress-based penalties (not the full multilevel kinodynamic formulation). |
| `BiTRRT` | Follows the OMPL planner structure, but its optimization objective is a clearance-derived cost field rather than an arbitrary user-supplied cost. |
| `A*` / `Dijkstra` | Optimal with respect to the induced occupancy grid, not the continuous image plane. |
| `CHOMP` | 2D point-robot specialization: signed distance field, covariant preconditioned updates, and functional obstacle gradients (not a full articulated configuration-space system). |
| `STOMP`, `TrajOpt`, `ITOMP`, `GPMP` | Each implements its paper's defining mechanism for a 2D point robot — per-timestep probability-weighted STOMP updates with the M projection; TrajOpt sequential convex optimization with ℓ1 collision penalties and a trust region; incremental covariant ITOMP over a receding horizon (static-map adaptation); and GPMP2-style LTI GP prior with GP-interpolated obstacle factors and Gauss-Newton inference. |

## Benchmarking

For a reproducible, headless cross-planner comparison (success rate, path length, clearance, compute
time, collision checks) across maps and seeds:

```bash
python -m path_planning_visualizer.benchmark
python -m path_planning_visualizer.benchmark --planners "A*,RRT*,BIT*" --maps all --seeds 5 --csv results.csv
```

## Status panel reference

| Field | Meaning |
| --- | --- |
| Path length | Geometric length of the final path, in pixels |
| Min clearance | Smallest obstacle distance anywhere along the path, in pixels |
| Mean clearance | Average obstacle distance along the full path, in pixels |
| Smoothness | Average squared turning angle on a resampled path (lower is smoother) |
| Compute time to first path | Planner compute time until the first valid solution |
| Total compute time | Total accumulated planner compute time for the current run |

## Development

Install the dev extras and run the suite (headless — Qt uses the offscreen platform):

```bash
pip install -e ".[dev]"
pytest
ruff check path_planning_visualizer tests
```

Continuous integration ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `ruff`, the full
`pytest` suite, and a benchmark smoke check on **Ubuntu, Windows, and macOS** across Python 3.11 and
3.12 for every push and pull request, and verifies that the built wheel ships its bundled maps.

## Project layout

```
path_planning_visualizer/      application package
├── __main__.py, app.py        entry point (python -m path_planning_visualizer)
├── types.py, geometry.py      shared types, geometry, and collision helpers
├── metrics.py                 path-quality metrics (length, clearance, smoothness)
├── mapping.py                 occupancy-grid load / save / brush helpers
├── resources.py               bundled-asset path resolution
├── benchmark.py               headless cross-planner benchmark CLI
├── planners/                  one module per planner, plus:
│   ├── registry.py            planner registry, groups, and in-app citations
│   └── _spatial, _rgg, _trajectory   shared helpers (spatial index, RGG radius, optimizer math)
├── gui/                       ImageCanvas, MainWindow, off-thread PlannerBuilder
└── assets/                    bundled example maps (maze.png, maze 2.png, maze 3.png)

tests/                         pytest suite (planner behavior, optimality, GUI integration)
assets/demo.gif                short walkthrough used in this README
pyproject.toml                 package metadata and installable script entry
CHANGELOG.md                   release history
LICENSE                        MIT
```

## Versioning

Semantic versioning with Python-compatible pre-release tags: betas follow `0.1.0b1`, `0.1.0b2`, …
`0.1.0b8`; the first stable release will be `0.1.0`, with bugfix releases continuing as `0.1.1`,
`0.1.2`, and so on. See [CHANGELOG.md](CHANGELOG.md) for the full release history.

## License

MIT — see [LICENSE](LICENSE).
