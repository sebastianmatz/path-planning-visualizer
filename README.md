# Path Planning Visualizer

[![CI](https://github.com/sebastianmatz/path-planning-visualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastianmatz/path-planning-visualizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta%20(0.1.0b12)-orange)
[![Download](https://img.shields.io/github/v/release/sebastianmatz/path-planning-visualizer?include_prereleases&label=download%20(Windows))](https://github.com/sebastianmatz/path-planning-visualizer/releases/latest)

An interactive desktop application for **exploring, comparing, and tuning path-planning algorithms** for a 2D point robot on occupancy-grid maps. It bundles 19 planners — sampling-based, graph-search, potential-field, trajectory-optimization, and metaheuristic — behind one UI, with step-through visualization, live path metrics, and an interactive map editor.

It is built as a **teaching and visualization environment**: each planner follows the defining mechanism of its source paper, adapted to a 2D occupancy grid, with the adaptations stated explicitly (see [Scientific scope](#scientific-scope)).

![Path Planning Visualizer demo showing map setup and planner execution](assets/demo.gif)

## Contents

- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Algorithms](#algorithms)
- [Scientific scope](#scientific-scope)
- [Status panel reference](#status-panel-reference)
- [Development](#development)
- [Project layout](#project-layout)
- [License](#license)

## Quick start

**Windows download (no Python needed).** Grab the latest `PathPlanningVisualizer.exe` from the
[**Releases**](https://github.com/sebastianmatz/path-planning-visualizer/releases/latest) page and
double-click it. It is a single self-contained file; on first launch Windows SmartScreen may warn
about an unknown publisher (the build is unsigned) — choose *More info → Run anyway*. To run on
macOS/Linux, or to develop, use the source install below.

### From source

Requirements: **Python 3.11+** and a desktop environment capable of running PyQt6. Create and
activate a virtual environment first:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

**Run it directly** — install the runtime dependencies and launch the package:

```bash
pip install -r requirements.txt
python -m path_planning_visualizer
```

**Or install the package** — this also registers a `path-planning-visualizer` console command:

```bash
pip install -e .
path-planning-visualizer        # equivalent to: python -m path_planning_visualizer
```

## Using the app

1. A bundled example maze loads automatically on startup. Use **Change Map** to switch
   between the bundled mazes (thumbnail picker), or **Map Tools → Load Image** for your own.
2. Click a free (white) pixel to place the **start**, then another to place the **goal**.
3. Pick an algorithm from the grouped dropdown and adjust its parameters in the left panel.
   The dropdown shows the reliable graph-search and sampling-based planners by default;
   tick **Show experimental algorithms** to also list the local trajectory optimizers,
   the potential field, and the metaheuristic (PSO).
4. **Step** advances the planner incrementally; **Run** plays it continuously; **Reset** clears the run.
   (Keyboard: Space = Run/Pause, `S` = Step, `R` = Reset, Esc = stop.)
5. The **Status** panel shows live path metrics, compute times, and the planner's status string.
6. For sampling-based planners, pause once a path is found and click **CHOMP Optimize** to smooth it.
7. For roadmap planners (**PRM**/**sPRM**), once a path is found you can **drag the start or goal
   marker** to re-query the *same* roadmap instantly (no re-sampling) — showing why a roadmap is worth
   building once and querying many times.

**Editing maps.** The map tools are collapsed by default behind the **Map Tools** toggle in the left
panel — click it to reveal them. Toggle **Edit Map** and left-drag to draw obstacles or right-drag to
erase (the **Brush** spin sets the radius). **New Map** starts from a blank grid; **Save Map** writes
the current grid to a PNG you can reload later, and **Save View** exports a screenshot of the rendered
map — with the tree, path, markers and legend — for figures. On large maps the planner is built on a
background thread, so the window stays responsive and shows a brief *Preparing…* state before it starts.

## Algorithms

19 planners across five families. Each row links the planner to the source paper it is modeled on
(the same citation shown in the app's info panel).

| Family | Algorithm | Reference |
| --- | --- | --- |
| Sampling-based | `RRT` | LaValle, 1998 |
| Sampling-based | `RRT-Connect` | Kuffner & LaValle, 2000 |
| Sampling-based | `BiTRRT` | Devaurs et al., 2013 |
| Sampling-based | `KPIECE` | Şucan & Kavraki, 2009 |
| Sampling-based | `RRT*` | Karaman & Frazzoli, 2011 |
| Sampling-based | `PRM` | Kavraki et al., 1996 |
| Sampling-based | `sPRM` | Karaman & Frazzoli, 2011 |
| Sampling-based | `SBL` | Sánchez & Latombe, 2001 |
| Sampling-based | `FMT*` | Janson et al., 2015 |
| Sampling-based | `BIT*` | Gammell et al., 2015 |
| Graph search | `A*` | Hart et al., 1968 |
| Graph search | `Dijkstra` | Dijkstra, 1959 |
| Potential field | `APF`* | Khatib, 1986 |
| Trajectory optimization | `CHOMP`* | Ratliff et al., 2009 |
| Trajectory optimization | `STOMP`* | Kalakrishnan et al., 2011 |
| Trajectory optimization | `TrajOpt`* | Schulman et al., 2014 |
| Trajectory optimization | `ITOMP`* | Park et al., 2012 |
| Trajectory optimization | `GPMP`* | Mukadam et al., 2016 |
| Metaheuristic | `PSO`* | Kennedy & Eberhart, 1995 |

\* **Experimental** — hidden by default; reveal them with the **Show experimental algorithms**
toggle above the dropdown. (The `*` inside `RRT*`, `FMT*`, and `BIT*` is part of the algorithm
name, not this marker — those three are default planners.)

**Default vs. experimental.** By default the dropdown lists only the **reliable** planners — the
graph-search and sampling-based families, which are complete, resolution-optimal, probabilistically
complete, or asymptotically optimal, and reliably return a path when one exists. The trajectory
optimizers (`CHOMP`, `STOMP`, `TrajOpt`, `ITOMP`, `GPMP`), the potential field (`APF`), and the
metaheuristic (`PSO`) are marked **experimental** and hidden behind the **Show
experimental algorithms** checkbox above the dropdown. They are faithful implementations of their
papers, but *local / best-effort* methods that optimize or descend from an initial guess rather than
search the configuration space — so they depend on the initialization and can stall at local minima
or fail to reach the goal on cluttered maps (which is the nature of the method, not a defect).

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
| `RRT-Connect` | Paper-exact Kuffner & LaValle (2000): EXTEND takes one step toward a uniform sample over `C` and lands exactly on it when within range; CONNECT greedily repeats EXTEND; the two trees swap and meet at one shared, collision-checked vertex. |
| `RRT*` | Paper-exact Algorithm 6 (Karaman & Frazzoli, 2011): ChooseParent + Rewire over the shrinking RGG radius `min(γ·(log n / n)^(1/2), step)` with `γ = 2(1+1/d)^(1/d)(μ/ζ_d)^(1/d)` → asymptotically optimal; a fixed search radius is selectable for comparison. |
| `BIT*` | Connects within the full RGG radius by default → asymptotically optimal (Gammell et al., 2015), with ordered vertex/edge queues, rewiring, informed batches, and incumbent pruning; an optional step-size cap is purely for tidier visuals. |
| `FMT*` | Paper-exact Janson et al. (2015): forward DP wavefront over uniform free-space samples within the shrinking radius `r_n`, one-shot lazy collision checking on the single best parent, terminating when the goal is popped as the lowest-cost open node. |
| `PRM` | Original Kavraki et al. (1996) construction step: random free configs connected within `max_edge_dist` (capped at `k`, increasing distance) by a straight-line local planner, **skipping same-connected-component edges** → a cycle-free forest. Two-phase learn/query with A\* search. Not asymptotically optimal (see `sPRM`). |
| `sPRM` | The simplified PRM (Karaman & Frazzoli, 2011): the same roadmap **without** the same-component cycle removal, so it keeps cycles. Asymptotically optimal and returns shorter query paths; this is what most libraries call "PRM". |
| `SBL` | Faithful 2D adaptation of Sánchez & Latombe (2001): density-weighted milestone selection (`π(m) ∼ 1/η(m)`), shrinking L∞ neighborhoods `B(m, ρ/i)`, the lazy dyadic `TEST-SEGMENT` (mark *safe* at `2^(−κ)·λ < ε`), `TEST-PATH` ordered most-likely-to-collide first, Fig. 4 milestone transfer on collision, and the paper's random shortcut optimizer. A finest-level exhaustive check guards the integer grid (and `ζ = ρ`). |
| `KPIECE` | Faithful single-level *geometric* adaptation of Şucan & Kavraki (2009): the paper's importance `log(𝓘)·score/(𝓢·𝓝·𝓒)`, the `2n` interior/exterior rule (`<4` neighbours in 2D), the fixed exterior-cell bias, half-normal motion selection, boundary-split AddMotion, and the `P = α + β·(ΔC/dist)` progress penalty — not the full multilevel *kinodynamic* formulation (no forward-propagation/physics; "simulated time" is traveled distance). |
| `BiTRRT` | Paper-exact Devaurs et al. (2013): adaptive-temperature transition test (deterministic 0.5, base-2 temperature updates), refinement control, and the downhill-only bidirectional junction (`attemptLink` within `10·δ`); the cost is a clearance-derived field (2D adaptation of the paper's generic cost). |
| `A*` / `Dijkstra` | Dijkstra (1959, Problem 2) uniform-cost search and A\* (Hart, Nilsson & Raphael 1968) best-first search with `f = g + h` and an admissible, consistent heuristic, on the induced 8-connected grid (Euclidean edge weights); optimal with respect to that grid discretization, not the continuous image plane. |
| `APF` | Paper-exact Khatib (1986): parabolic-well attractive force `−k(x − x_goal)`, FIRAS repulsion within `ρ₀`, integrated with velocity saturation at `V_max`. Pure APF by default, so it stalls at local minima (Khatib's documented limitation); a non-paper stochastic escape is an optional toggle. |
| `CHOMP` | Paper-exact Ratliff et al. (2009) for a 2D point robot: a true signed distance field, the workspace cost `c(x)`, the obstacle functional gradient (Eq. 4), and the covariant step `ξ ← ξ − (1/λ)A⁻¹g` over the velocity prior. Also the *CHOMP Optimize* post-processor that refines sampling-based paths (not a full articulated configuration-space system). |
| `STOMP`, `TrajOpt`, `ITOMP`, `GPMP` | Each implements its paper's defining mechanism for a 2D point robot — per-timestep probability-weighted STOMP updates (Eq. 11) with the M projection on a signed-distance obstacle cost (Eq. 13); TrajOpt (Schulman et al. 2013) sequential convex optimization — the displacement objective with ℓ1 collision penalties on the signed distance, in a trust region; incremental covariant ITOMP (Park et al. 2012) over a receding horizon with acceleration smoothness and a signed-distance obstacle cost (static-map adaptation; no dynamic obstacles); and GPMP (Mukadam, Yan & Boots 2016) — an LTI GP prior with GP interpolation, optimized by the covariant gradient update (gradient preconditioned by the GP covariance). |
| `PSO` | Paper-exact Kennedy & Eberhart (1995) by default: `v ← w·v + 2·r₁·(pbest−x) + 2·r₂·(gbest−x)` with a `Vmax` clamp and full momentum (no inertia weight, `w = 1.0`), over a fixed-waypoint path encoding (fitness = length + clearance penalty + smoothness). Pure 1995 PSO can stall on cluttered maps (like pure APF); an off-by-default *safeguards* toggle adds adaptive inertia/social gain, diversity injection, random immigrants, and swarm restart. A metaheuristic, not a complete planner. |

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
├── planners/                  one module per planner (PyQt6-free), plus:
│   ├── registry.py            planner registry, groups, and in-app citations
│   └── _spatial, _rgg, _trajectory   shared helpers (spatial index, RGG radius, optimizer math)
├── gui/                       ImageCanvas, MainWindow, param_panels, map_picker, off-thread PlannerBuilder
└── assets/                    bundled example maps (maze.png, maze 2.png, maze 3.png, maze 4.png)

tests/                         pytest suite (planner behavior, optimality, GUI integration)
assets/demo.gif                short walkthrough used in this README
pyproject.toml                 package metadata and installable script entry
CHANGELOG.md                   release history
LICENSE                        MIT
```

## License

MIT — see [LICENSE](LICENSE).
