# Path Planning Visualizer

Path Planning Visualizer is a desktop application for exploring, comparing, and tuning classical path-planning algorithms on occupancy-grid maps.

Current release status: `0.1.0b1` (`Beta`)

## Preview

The demo below shows the typical workflow: loading a map, placing start and goal points, adjusting planner settings, and running the visualization.

![Path Planning Visualizer demo showing map setup and planner execution](assets/demo.gif)

## Features

- Interactive map loading with click-to-set start and goal points
- Side-by-side parameter configuration for multiple planner families
- Step mode and continuous playback for algorithm visualization
- Built-in example maps for quick testing

## Included Algorithms

- Sampling-Based: `RRT`, `RRT-Connect`, `RRT*`, `PRM`, `FMT*`, `BIT*`
- Graph Search: `A*`, `Dijkstra`
- Potential Field: `APF`
- Trajectory Optimization: `CHOMP`, `STOMP`, `TrajOpt`, `ITOMP`, `GPMP`
- Metaheuristic: `PSO`, `Genetic`

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

## Project Layout

- `path_planning_visualizer.py`: desktop application entry point
- `assets/demo.gif`: short application walkthrough for the README
- `assets/maze.png`, `assets/maze 2.png`: bundled example maps
- `pyproject.toml`: package metadata and installable script entry

## Notes

- The application uses `opencv-python-headless` for image processing and `PyQt6` for the UI.

## Versioning

- The project uses semantic versioning with Python-compatible pre-release tags.
- Beta releases follow the pattern `0.1.0b1`, `0.1.0b2`, `0.1.0b3`, and so on.
- The first stable release would be `0.1.0`.
- Bugfix releases after that would continue as `0.1.1`, `0.1.2`, etc.
