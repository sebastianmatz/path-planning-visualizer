"""Reproducible micro-benchmark for the path planners.

Runs a chosen set of planners over a few built-in maps and several random seeds
and reports success rate, path length, clearance, compute time and collision
checks.  It is headless (no Qt) and uses only the standard library plus the
package itself, so it can run in CI or from a plain shell::

    python -m path_planning_visualizer.benchmark
    python -m path_planning_visualizer.benchmark --planners "A*,RRT*,BIT*" --seeds 5
    python -m path_planning_visualizer.benchmark --maps all --csv results.csv

This is a convenience comparison harness, not a formal benchmark suite: maps are
small and the per-planner budgets are tuned for a quick turnaround.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import path_planning_visualizer as ppv

Point = Tuple[int, int]


# --------------------------------------------------------------------------- maps
def _nearest_free(occ: np.ndarray, near: Point) -> Point:
    h, w = occ.shape
    nx, ny = near
    for radius in range(max(h, w)):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = nx + dx, ny + dy
                if 0 <= x < w and 0 <= y < h and not occ[y, x]:
                    return (x, y)
    raise ValueError("no free cell found")


def map_open() -> Tuple[np.ndarray, Point, Point]:
    return np.zeros((60, 60), dtype=bool), (8, 30), (52, 30)


def map_wall() -> Tuple[np.ndarray, Point, Point]:
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:45, 30] = True
    occ[0:45, 31] = True
    return occ, (8, 8), (52, 8)


def map_passage() -> Tuple[np.ndarray, Point, Point]:
    occ = np.zeros((60, 60), dtype=bool)
    occ[29:32, :] = True
    occ[29:32, 40:48] = False  # narrow gap off to one side
    return occ, (10, 8), (50, 52)


def map_maze() -> Tuple[np.ndarray, Point, Point]:
    import cv2

    from .resources import asset_path

    img = cv2.imread(str(asset_path("maze.png")), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("maze.png could not be loaded")
    img = cv2.resize(img, (90, 90), interpolation=cv2.INTER_AREA)
    occ = img < 128
    start = _nearest_free(occ, (5, 5))
    goal = _nearest_free(occ, (occ.shape[1] - 6, occ.shape[0] - 6))
    return occ, start, goal


BUILTIN_MAPS = {
    "open": map_open,
    "wall": map_wall,
    "passage": map_passage,
    "maze": map_maze,
}
DEFAULT_MAPS = ["open", "wall", "passage"]


# ----------------------------------------------------------------------- planners
# Per-planner benchmark budgets (everything else uses the planner defaults).
PARAMS: Dict[str, dict] = {
    "RRT": dict(max_vertices=8000),
    "RRT-Connect": dict(max_iters=8000),
    "BiTRRT": dict(max_iters=8000),
    "KPIECE": dict(max_iters=8000),
    "RRT*": dict(max_iters=6000),
    "PRM": dict(num_samples=600),
    "SBL": dict(max_iters=8000),
    "FMT*": dict(num_samples=800),
    "BIT*": dict(max_iters=6000),
    "APF": dict(max_iters=4000),
    "CHOMP": dict(max_iters=600),
    "STOMP": dict(max_iters=300),
    "TrajOpt": dict(max_iters=300),
    "ITOMP": dict(max_iters=600),
    "GPMP": dict(max_iters=200),
    "PSO": dict(max_iters=200),
    "Genetic": dict(max_iters=200),
}
DEFAULT_PLANNERS = ["A*", "Dijkstra", "RRT", "RRT-Connect", "RRT*", "FMT*", "PRM", "BIT*"]


def _build(name: str, occ: np.ndarray, start: Point, goal: Point, seed: int):
    cls = ppv.AVAILABLE_PLANNERS[name]
    kwargs = dict(PARAMS.get(name, {}))
    if "seed" in inspect.signature(cls.__init__).parameters:
        kwargs["seed"] = seed
    return cls(occ, start, goal, **kwargs)


def _is_seeded(name: str) -> bool:
    cls = ppv.AVAILABLE_PLANNERS[name]
    return "seed" in inspect.signature(cls.__init__).parameters


def _collision_checks(planner) -> Optional[int]:
    for attr in ("edge_collision_checks", "collision_checks"):
        value = getattr(planner, attr, None)
        if value is not None:
            return int(value)
    return None


class Result:
    __slots__ = ("success", "length", "clearance", "time", "checks")

    def __init__(self):
        self.success: List[bool] = []
        self.length: List[float] = []
        self.clearance: List[float] = []
        self.time: List[float] = []
        self.checks: List[int] = []


def _run_once(name: str, occ: np.ndarray, start: Point, goal: Point, seed: int,
              clearance_field: np.ndarray, max_steps: int):
    planner = _build(name, occ, start, goal, seed)
    t0 = time.perf_counter()
    steps = 0
    while not planner.done and steps < max_steps:
        planner.step_once()
        steps += 1
    elapsed = time.perf_counter() - t0

    if not planner.found_path:
        return False, None, None, elapsed, _collision_checks(planner)
    path = planner.extract_path()
    metrics = ppv.compute_path_metrics(path, clearance_field)
    return True, metrics.length_px, metrics.min_clearance_px, elapsed, _collision_checks(planner)


def benchmark(planners: List[str], maps: List[str], seeds: int, max_steps: int
              ) -> Dict[str, Dict[str, Result]]:
    results: Dict[str, Dict[str, Result]] = {}
    for map_name in maps:
        occ, start, goal = BUILTIN_MAPS[map_name]()
        clearance_field = ppv.make_distance_field(occ)
        results[map_name] = {}
        for name in planners:
            res = Result()
            n_runs = seeds if _is_seeded(name) else 1
            for s in range(n_runs):
                ok, length, clear, elapsed, checks = _run_once(
                    name, occ, start, goal, s + 1, clearance_field, max_steps
                )
                res.success.append(ok)
                res.time.append(elapsed)
                if ok:
                    res.length.append(length)
                    res.clearance.append(clear)
                if checks is not None:
                    res.checks.append(checks)
            results[map_name][name] = res
    return results


# ------------------------------------------------------------------------ reporting
def _fmt(value: Optional[float], spec: str) -> str:
    if value is not None:
        return format(value, spec)
    match = re.search(r"(\d+)", spec)
    return "-".rjust(int(match.group(1)) if match else 0)


def _fmt_clear(value: Optional[float], width: int = 10) -> str:
    # An obstacle-free map has unbounded clearance (distance transform ~ FLT_MAX).
    if value is None:
        return "-".rjust(width)
    if value > 1e6:
        return "inf".rjust(width)
    return format(value, f">{width}.1f")


def _print_table(results: Dict[str, Dict[str, Result]]) -> None:
    header = f"{'Planner':<13}{'Success':>9}{'MeanLen':>10}{'BestLen':>10}{'MinClear':>10}{'Time(ms)':>10}{'Checks':>9}"
    for map_name, per_planner in results.items():
        print(f"\n=== map: {map_name} ===")
        print(header)
        print("-" * len(header))
        for name, res in per_planner.items():
            runs = len(res.success)
            success_rate = sum(res.success) / runs if runs else 0.0
            mean_len = float(np.mean(res.length)) if res.length else None
            best_len = float(np.min(res.length)) if res.length else None
            min_clear = float(np.min(res.clearance)) if res.clearance else None
            mean_ms = float(np.mean(res.time)) * 1e3 if res.time else None
            mean_checks = float(np.mean(res.checks)) if res.checks else None
            print(
                f"{name:<13}{success_rate * 100:>8.0f}%"
                f"{_fmt(mean_len, '>10.1f')}{_fmt(best_len, '>10.1f')}"
                f"{_fmt_clear(min_clear)}{_fmt(mean_ms, '>10.1f')}"
                f"{_fmt(mean_checks, '>9.0f')}"
            )


def _write_csv(results: Dict[str, Dict[str, Result]], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["map", "planner", "success_rate", "mean_length", "best_length",
                         "min_clearance", "mean_time_ms", "mean_checks", "runs"])
        for map_name, per_planner in results.items():
            for name, res in per_planner.items():
                runs = len(res.success)
                writer.writerow([
                    map_name, name,
                    f"{sum(res.success) / runs:.3f}" if runs else "0",
                    f"{np.mean(res.length):.2f}" if res.length else "",
                    f"{np.min(res.length):.2f}" if res.length else "",
                    (("inf" if np.min(res.clearance) > 1e6 else f"{np.min(res.clearance):.2f}")
                     if res.clearance else ""),
                    f"{np.mean(res.time) * 1e3:.2f}" if res.time else "",
                    f"{np.mean(res.checks):.1f}" if res.checks else "",
                    runs,
                ])
    print(f"\nWrote {path}")


def _resolve(names: str, available: List[str], label: str) -> List[str]:
    if names.strip().lower() == "all":
        return list(available)
    chosen = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in chosen if n not in available]
    if unknown:
        raise SystemExit(f"unknown {label}: {', '.join(unknown)} (available: {', '.join(available)})")
    return chosen


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Path-planner micro-benchmark.")
    parser.add_argument("--planners", default=",".join(DEFAULT_PLANNERS),
                        help="comma-separated planner names, or 'all'")
    parser.add_argument("--maps", default=",".join(DEFAULT_MAPS),
                        help="comma-separated map names, or 'all' "
                             f"(available: {', '.join(BUILTIN_MAPS)})")
    parser.add_argument("--seeds", type=int, default=3, help="runs per seeded planner")
    parser.add_argument("--max-steps", type=int, default=200_000, help="hard step cap per run")
    parser.add_argument("--csv", default=None, help="optional path to write results as CSV")
    args = parser.parse_args(argv)

    planners = _resolve(args.planners, list(ppv.AVAILABLE_PLANNERS), "planner")
    maps = _resolve(args.maps, list(BUILTIN_MAPS), "map")

    results = benchmark(planners, maps, max(1, args.seeds), args.max_steps)
    _print_table(results)
    if args.csv:
        _write_csv(results, args.csv)


if __name__ == "__main__":
    main()
