# Changelog

This file tracks release notes for published versions of the project.

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
