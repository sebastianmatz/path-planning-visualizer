# Changelog

This file tracks release notes for published versions of the project.

## [Unreleased]

## [0.1.0b11] - 2026-06-16

### Added

- **Roadmap re-query by dragging start/goal (PRM/sPRM).** Once a roadmap planner
  finishes, the start and goal markers become draggable; dropping one re-solves on the
  **same learned roadmap** — only the start/goal attachment and graph search are redone,
  no re-sampling — so the path updates near-instantly. This showcases what makes a
  roadmap method worth its build cost (cheap repeated queries). The path updates **live
  while dragging**, and invalid drops (onto an obstacle) snap back. Once re-query mode
  is active the canvas shows the **fixed roadmap only** (the path itself shows the
  start/goal attachment), so the roadmap visibly stays put instead of appearing to
  sprout new edges at each drop point. New `PRMPlanner.requery`/`roadmap_edges`; canvas
  marker-drag interaction.

- **The active search frontier is highlighted for the rewiring/search planners.**
  RRT\*, BIT\*, A\*, Dijkstra and SBL redraw their whole tree each frame; the edges
  added/relaxed *this step* are now shown as a brief fading highlight (cyan edges + node
  dots) over the settled tree, so it is visible where the search is progressing right
  now. A\*/Dijkstra report every neighbour relaxed per expansion (not just the last) so
  the frontier reads clearly.

- **Windows executable + automated release builds.** A single-file `.exe` is produced
  with PyInstaller (bundling the example mazes and package metadata, so the in-app
  version is correct); on each pushed version tag a GitHub Actions workflow builds it on
  a Windows runner and attaches it to the GitHub Release, with a Sigstore/SLSA
  **build-provenance attestation**. The README documents the download (and the
  SmartScreen note) plus `gh attestation verify`. (`packaging/`,
  `.github/workflows/release.yml`.)

### Changed

- **PRM now builds its roadmap incrementally, matching Kavraki et al. (1996).** The
  forest variant previously connected each node to its k-nearest among *all* samples in
  one batch pass, so whichever node was processed first grabbed all k neighbours at once
  (degree-15 "mega-hub" stars). It now connects each node only to nodes *already placed*
  — Kavraki's incremental learning phase — still skipping same-component pairs. The
  result is the same cycle-free forest (verified 0 cycles) but balanced and organically
  grown (max degree 15 → 8), a faithful depiction of the algorithm. sPRM keeps its batch
  all-pairs connection per Karaman & Frazzoli (2011).

### Fixed

- **CHOMP optimization is fast again (~5× speedup, back under a second).** Its
  per-iteration best-valid/collision check converted every waypoint to a pixel with
  scalar `np.clip(np.rint(...))`, whose numpy per-call overhead dominated the profile
  (~0.9 s of pure clip machinery; the scalar `np.clip`/`np.rint` calls, not the
  collision sampling). Replaced with plain Python round/clamp — behaviour-identical
  (same rounding, same clamp, same optimized path), just fast. Measured ~3.4 s → ~0.7 s
  on a 50-point / 270-iteration optimize.
- **Collision checking no longer skips obstacle corners (could yield paths through
  walls).** The segment rasterization used fixed-rate `max(dx, dy)` sampling, which on
  a diagonal edge could round *around* the corner cell of an obstacle and report a
  colliding edge as clear. BIT\* — the one planner that relied on the coarse default
  (others pass a dense `samples=`) — returned paths that clipped wall corners, *on the
  real bundled maze*, and `test_optimality`'s collision check used the same coarse
  default so it could not catch it. `geometry.segment_points` is now an exact grid
  (voxel) traversal that visits every touched cell and never skips a corner; the
  `samples` argument is kept for compatibility but ignored. Benefits every collision
  check and clearance metric. Added `tests/test_geometry.py`.
- **A\*, Dijkstra and SBL now display their actual tree.** A\*/Dijkstra relax nodes
  still in the open set (a node's parent can change before expansion), so drawing one
  relaxation edge per pop left the displayed search tree incomplete and partly stale;
  they now expose `extract_tree_edges()` (from `came_from`) and the canvas redraws the
  whole tree once per frame (throttled like the optimizers — rebuilding it every step
  is O(tree) and would make these fast planners O(n²)). SBL is lazy and attempts
  cross-tree bridges (some in
  collision); it now redraws its authoritative, collision-free milestone tree rather
  than accumulating attempt edges (no more through-wall/stale edges in the display).
- **PRM/sPRM sampled milestones are drawn as dots, not edges.** Each sample was drawn
  as a tiny diagonal "marker edge" (not a real edge; some clipped walls). Milestones
  are now drawn as persistent node dots via a new `StepResult.node_marker` channel; the
  roadmap edges (already accurate) are unchanged.
- **Genetic no longer draws candidate segments through obstacles.** It visualized one
  segment of its current best individual unconditionally, so an unconverged candidate
  could show a segment crossing a wall; it now guards each drawn segment with a
  collision check (as PSO already did).
- **RRT\* now displays its actual tree — rewiring is shown correctly.** RRT\* rewires
  (a node's parent changes when a cheaper connection is found), but the GUI was
  *appending* each new edge and never redrawing, so rewired-away edges lingered on
  screen while the new rewired edges were never shown. Measured on a typical run,
  ~half the displayed tree was wrong (183 stale + 182 missing edges out of 362). The
  canvas now redraws RRT\*'s tree from its authoritative parent structure each step
  (new `RRTStarPlanner.extract_tree_edges`, routed through the same whole-tree redraw
  BIT\* already used); the displayed tree now matches the real tree exactly. Guarded
  by a GUI regression test.

## [0.1.0b10] - 2026-06-16

### Changed

- **Light UI polish.** The left-panel sections (Algorithm / Parameters / Animation
  Speed / Status) now render as bordered "cards" with bold titles via an app
  stylesheet; the primary **Run** action has a green accent (with hover/pressed/
  disabled states); and the canvas shows a small **color legend** as a horizontal
  strip in the margin *below* the map (off the map, not obstructing it). The legend
  uses the exact map colors on a dark chip (so the bright-yellow path is legible)
  and gains an **"Optimized"** row only when an optimized (CHOMP) path is present.
  Other widgets keep their native styling.
- **Status panel** values are right-aligned in a stable column (consistent unit
  spacing), so the numbers no longer shift — or make the panel jitter — as they
  update during a run/optimization.
- **Optimizer animations (CHOMP/GPMP) are no longer gated to 1 iteration per frame.**
  They were artificially capped at ~1 iteration/8 ms tick (~1.2 s for a CHOMP
  optimization even though it computes in ~0.2 s). The run loop now batches
  iterations per frame **scaled by the speed slider** — at MAX it runs a
  time-budgeted batch (CHOMP Optimize ~0.4 s, ~2-3× faster) and updates the
  smoothed display once per frame; dragging the slider down shows it
  iteration-by-iteration. The **algorithm is untouched** (same iterations, same
  result — the finalized path is byte-identical to the un-batched run); only the
  display cadence changed.
- **CHOMP Optimize now shows the path's evolution smoothly.** The original sampling
  path stays visible (solid) while/after CHOMP optimizes, alongside the optimized
  path. The displayed optimizer path is **eased toward each iterate** (exponential
  smoothing, reusing `blend_float_paths`) so a fast, near-convergence-oscillating
  optimizer *glides* instead of wobbling — measured per-frame motion dropped ~4×
  (mean ≈5 px → ≈1 px) with the oscillation removed (late-frame jumps >3 px: 47/48 →
  0/47). The final frame now **morphs into the converged path** instead of snapping
  (the previous "ease to final" was a no-op because the finalizing step pre-snapped
  the canvas; that step is now skipped so the tween has something to morph from),
  then **settles to a clean solid line** (the live glow and iteration trails are
  cleared) so the finished result reads cleanly.
- **Reworked the canvas to render the visualization as vectors at display
  resolution.** Previously the whole scene (map + tree + path + markers) was drawn
  at the occupancy-grid resolution and bitmap-upscaled, so lines were blurry and
  marker/line sizes scaled with the map (e.g. huge node "blobs" on small maps).
  `ImageCanvas` now paints in a `paintEvent`: the base map is scaled on draw
  (crisp/nearest when enlarging, smooth when shrinking) and tree edges, paths,
  highlights and start/goal markers are drawn in image coordinates under a scale
  transform with **cosmetic (screen-space) pen widths** and fixed-radius dots — so
  the visualization is sharp and consistently sized at any map resolution, and
  redrawing no longer rebuilds + rescales a full pixmap each step. The public
  canvas API (`draw_edge`, `draw_path`, `set_current_path`, `set_current_tree_edges`,
  markers, fades, clears) is unchanged.
- **Decoupled the algorithm layer from PyQt6.** All per-planner parameter widgets
  moved into a new GUI module `path_planning_visualizer/gui/param_panels.py`
  (with a `PARAM_PANELS` registry); the planner modules and `base.py` no longer
  import PyQt6, and the GUI/entry-point symbols (`MainWindow`, `ImageCanvas`,
  `main`) are now imported lazily in the package `__init__`. As a result
  `import path_planning_visualizer` and the headless benchmark run **without
  PyQt6 loaded** (verified by `tests/test_headless_import.py`). The dead
  `create_from_params` hook was removed from every planner.
- The trajectory optimizers' obstacle-term SDF lookups are vectorized: `TrajOpt`,
  `ITOMP`, and `GPMP` now query the signed distance field in batched calls (new
  `_trajectory.sdf_query_batch`) instead of per-waypoint Python loops (STOMP was
  already vectorized). Results are identical to the old loops (≤ ~1e-12); GPMP, in
  which the SDF lookup was ~⅔ of the run time, is markedly faster.
- Lint ruleset expanded to `F, I, B, SIM` with a configured line length; import
  order normalized. Naming (`N`) and line-length (`E501`) are intentionally not
  enabled (they would fight the deliberate paper math notation / docstrings).
- **The in-app algorithm descriptions now use proper math notation.** ASCII math in
  the registry descriptions (`x_dot`, `||u|| <= 1`, `xi <- xi - (1/lambda) A^-1 g`,
  `log(I)*score/(S*N*C)`, …) was rewritten with Unicode symbols and HTML
  sub/superscripts (ẋ, ‖u‖ ≤ 1, ξ ← ξ − (1/λ)A⁻¹g, …), rendered in the existing
  RichText info label. Wording, meaning and citations are unchanged.

### Fixed

- **RRT could report a path "through a wall."** Its goal-region rule accepted *any*
  tree vertex within the goal radius, with **no line-of-sight check** — so a vertex
  on the far side of a thin wall counted as reaching the goal (and the returned path
  did not even include the actual goal). `_goal_reached` now additionally requires a
  collision-free straight line from the vertex to the goal, and `extract_path`
  finishes the path at the goal via that verified segment. RRT now declares success
  only when a real collision-free connection exists, consistent with the other
  goal-region planners (RRT*, KPIECE, SBL…).
- **The Step button stopped working after one click.** Entering the off-thread
  "preparing" state (which the first Step, and any restart-after-done, triggers)
  disabled Step/Run, but leaving that state only restored the cursor — so Step was
  left permanently greyed out and further clicks did nothing. `_set_preparing_state`
  now re-enables Step/Run when a map and start/goal exist (the Run continuation still
  overrides to the running state). Guarded by a GUI regression test.
- **Stepping/running on a large tree could freeze the UI.** The canvas re-stroked
  *every* accumulated tree edge on *every* repaint (≈131 ms at ~4 200 edges, growing
  with the tree); combined with the 50 ms highlight-fade timer, repaints could not
  keep up and the event loop stalled. Accumulated tree edges are now baked once into
  a persistent display-resolution layer that `paintEvent` blits in O(1), and the
  transient activity highlights are capped to a bounded recent window — full repaint
  dropped to ~10 ms regardless of tree size, with the tree rendered identically.
- `GPMP` class docstring/`description` corrected to describe the covariant gradient
  update actually used (the b9 rework), not the old GPMP2 Gauss-Newton form.

### Added

- **Save View** (in the Map Tools panel): exports a screenshot of the rendered
  canvas — the map *with* the tree, path, start/goal markers and legend — as a PNG,
  distinct from **Save Map** (which writes only the bare occupancy grid). The Map
  Tools controls now wrap onto two rows (drawing tools / map file actions) so the
  buttons are no longer cramped.
- **Keyboard shortcuts:** Space = Run/Pause, `S` = Step, `R` = Reset, Esc = stop;
  shown in the button tooltips. Playback buttons no longer take keyboard focus so
  Space can't double-trigger a focused button.
- Unit tests for `metrics`, the `benchmark` harness, the `ImageCanvas` widget, and
  the headless-import guarantee (`tests/test_metrics.py`, `test_benchmark.py`,
  `test_canvas.py`, `test_headless_import.py`).

## [0.1.0b9] - 2026-06-14

### Changed — paper-fidelity audit

- `APF` made paper-exact to Khatib (1986): parabolic-well attractive force
  `-k(x - x_goal)`, the FIRAS repulsive force within the influence limit, and
  velocity-saturated integration at `V_max` (was a conic attractor with a
  unit-normalized step). Pure APF is now the default and stalls at local minima
  as the paper documents; the stochastic escape is an optional off-by-default
  toggle. Added `tests/test_apf_fidelity.py`.
- `STOMP` made paper-exact to Kalakrishnan et al. (2011): the per-timestep
  probability now uses the paper's exponent `exp(-h·(S-min)/(max-min))` (Eq. 11),
  the obstacle cost is `max(ε - d, 0)·‖ẋ‖` on a true **signed** distance field
  (Eq. 13), and the convergence metric includes the control term `½θᵀRθ`. The
  `R⁻¹` noise and M projection were already correct; exploration noise is now
  fixed (un-annealed) per the paper. Added a shared `signed_distance_field` helper
  and `tests/test_stomp_fidelity.py`. (Also fixed a latent attribute-shadowing
  bug where the new sensitivity parameter collided with the grid height.)
- `BIT*` verified line-by-line against Gammell et al. (2015) and brought to
  paper-exact structure: vertex-vertex rewiring edges are now enqueued only from
  vertices new to the current batch (Alg. 2 line 4), pruning runs at batch start
  (Alg. 1 line 5), and the edge queue breaks ties by source cost-to-come `g_T(v)`
  (Alg. 1 line 12). Added `tests/test_bit_star_fidelity.py`. Tree-vertex pruning
  (Alg. 3 lines 2-5) remains omitted as an asymptotic-optimality-preserving
  simplification (documented in `literature/fidelity/bit_star.md`).
- `RRT-Connect` made paper-exact to Kuffner & LaValle (2000): `EXTEND` now lands
  exactly on its target when within `step_size` and returns *Reached* only then
  (Fig. 2); `CONNECT` is the paper's "repeat `EXTEND` until not *Advanced*"
  (Fig. 5); and `RANDOM_CONFIG` samples uniformly over the whole space `C` instead
  of rejection-sampling free space (an occupied sample is only a steering target).
  Added `tests/test_rrt_connect_fidelity.py`.
- `FMT*` termination aligned to the paper (Janson et al. 2015, Alg. 1 line 9.2):
  the goal is now connected like any other sample and the planner terminates when
  the goal is popped as the lowest-cost node in `V_open`, rather than the instant
  it is connected. The returned path is unchanged (an FMT* node's cost is final
  when it enters `V_open`); the wavefront, lazy single-best-parent connection, and
  shrinking radius `r_n` were already faithful. Added `tests/test_fmt_star_fidelity.py`.
- `CHOMP` update rule made paper-exact to Ratliff et al. (2009, Sec. II-A): the
  backtracking line search is replaced by the covariant step
  `ξ ← ξ − (1/λ)A⁻¹g` with a single step-norm cap. The signed distance field, the
  workspace cost `c(x)`, the obstacle functional gradient (Eq. 4), and the
  velocity prior (Eq. 1, `d=1`) were already faithful. The best-valid finalize is
  kept so CHOMP still reliably returns a collision-free result, including when
  used as the *CHOMP Optimize* post-processor for sampling-based paths (same
  `CHOMPPlanner` via `init_trajectory`). Added `tests/test_chomp_fidelity.py`.
- `RRT` verified against LaValle (1998) `GENERATE_RRT`: the
  sample / nearest / `SELECT_INPUT` / `NEW_STATE` loop and the holonomic
  `ẋ = u` (`‖u‖ ≤ 1`) model with `Δt` Euler integration are faithful. The goal
  bias (the paper's Sec. 5 extension), the goal-region stop, `K` as a vertex
  budget, and the grid-discretization rejections are documented as single-query
  adaptations (`literature/fidelity/rrt.md`). Added `tests/test_rrt_fidelity.py`.
- `RRT*` verified against Karaman & Frazzoli (2011) Algorithm 6: `ChooseParent`
  and `Rewire` and the shrinking radius `min(γ·(log n/n)^(1/2), step)` with the
  paper's `γ* = 2(1+1/d)^(1/d)(μ/ζ_d)^(1/d)` are faithful (the b5/b6 hardening
  was correct). Goal handling, goal bias, and the fixed-radius option are
  documented single-query adaptations (`literature/fidelity/rrt_star.md`). Added
  `tests/test_rrt_star_fidelity.py` (radius cap, the `Cost(v)=Cost(parent)+‖·‖`
  recursion, acyclic tree).
- `PRM` verified against Kavraki et al. (1996): the two-phase learn/query
  structure, random free configs, the `N_c` candidate set (within `max_edge_dist`,
  capped at `k_neighbors`, increasing distance), the straight-line local planner,
  and the A* query are faithful. The implementation is documented as the
  asymptotically-optimal **sPRM** variant (Karaman & Frazzoli 2011) — it keeps
  cycles rather than the paper's same-component forest, which the paper itself
  notes yields shorter paths (`literature/fidelity/prm.md`). Added
  `tests/test_prm_fidelity.py`.

- `Dijkstra` verified against Dijkstra (1959, Problem 2): `closed_set` = the
  finalized set A (nodes closed in increasing distance from the start), the
  heap + `dist` map = the frontier B, `heappop` = the minimum-distance extraction
  (Step 2), and edge relaxation = Step 1, terminating when the goal is popped. The
  coarse induced grid, 8-connectivity with Euclidean edge weights, and corner-cut
  prevention are documented adaptations (`literature/fidelity/dijkstra.md`). Added
  `tests/test_dijkstra_fidelity.py`.
- `A*` verified against Hart, Nilsson & Raphael (1968): the evaluation `f = g + h`,
  the min-`f` selection, edge relaxation, and goal-pop termination are faithful,
  with an admissible + consistent Euclidean heuristic (so closed nodes are never
  reopened, per the paper's Lemma 2). Same induced-grid adaptations as Dijkstra
  (`literature/fidelity/astar.md`). Added `tests/test_astar_fidelity.py`.
- `TrajOpt` made paper-exact to Schulman et al. (2013): the objective is now the
  sum of squared **displacements** `Σ‖θ_{t+1}−θ_t‖²` (Eq. 5; was accelerations),
  the collision penalty uses a true **signed** distance field, and the trust-region
  step is accepted iff `true/model improvement > c` (Alg. 1). Citation corrected to
  the 2013 RSS paper. The convex subproblem remains a box-clipped Newton step (no
  QP solver). Added `tests/test_trajopt_fidelity.py`.
- `ITOMP` aligned to Park, Pan & Manocha (2012): the static obstacle cost is now
  the Eq. 8 hinge `max(ε − d, 0)·‖ẋ‖` on a true **signed** distance field, and the
  smoothness metric is confirmed acceleration-based (Eq. 6). The receding-horizon
  incremental optimization is faithful; the dynamic-obstacle cost (Eq. 9) is
  documented as out of scope (the tool has only static maps) — the one inherent
  ITOMP gap. Added `tests/test_itomp_fidelity.py`.
- `GPMP` **reworked** to the ICRA-2016 *Gaussian Process Motion Planning* paper
  (Mukadam, Yan & Boots) requested by the user: the factor-graph Gauss-Newton/LM
  solver (GPMP2) is replaced by the paper's **covariant gradient update**
  `ξ ← ξ − (1/η)·K·∇U` — the cost gradient preconditioned by the GP covariance `K`
  (`K⁻¹ = BᵀQ⁻¹B`). The constant-velocity LTI GP prior, GP interpolation, and a
  now-**signed** SDF are kept. Citation set to Mukadam, Yan & Boots (2016). Added
  `tests/test_gpmp_fidelity.py`.
- `BiTRRT` made paper-exact to Devaurs, Siméon & Cortés (2013): the transition
  test now updates the temperature with **base-2** powers — `T /= 2^(Δc/(0.1·costRange))`
  on an accepted uphill move and `T *= 2^(T_rate)` on a rejected one (Alg. 2, was
  base `e`); `attemptLink` only fires within `10·δ` and extends the target tree
  toward the source along **flat/downhill slopes only** (Alg. 5, was the full
  transition-test extension); and refinement control thresholds at the step size
  `δ` with the `nbRefinement > ρ·nbNodes` test (Alg. 3). The deterministic
  `exp(−Δc/T) > 0.5` acceptance and the bidirectional extend/link/swap loop were
  already faithful; the clearance-derived cost field is the documented 2D adaptation
  of the paper's generic cost. Citation set to Devaurs et al. (2013). Added
  `tests/test_bitrrt_fidelity.py`.
- `KPIECE` verified against Şucan & Kavraki (2009) and made paper-exact in cell
  selection: the importance `log(𝓘)·score/(𝓢·𝓝·𝓒)` (p. 6), the `2n` interior/
  exterior rule (`<4` axis-neighbours in 2D, p. 4), the half-normal motion
  selection, the boundary-split AddMotion (Alg. 2), and the `P = α + β·(ΔC/dist)`
  progress penalty with the `P<1 ⇒ score·=P` rule (Alg. 1 l. 16-17) were already
  faithful; the exterior-cell bias is now the paper's *fixed* 70-80% bias
  (Alg. 1 l. 5) rather than an inflated `max(border_fraction, ratio)`. Documented
  as a single-level *geometric* adaptation (no forward-propagation; "simulated
  time" is traveled distance). Citation corrected 2008→2009. Added
  `tests/test_kpiece_fidelity.py`.
- `SBL` verified against Sánchez & Latombe (2001): density-weighted milestone
  selection (`π(m) ∼ 1/η(m)` via the per-tree grid), shrinking L∞ neighborhoods
  `B(m, ρ/i)`, the lazy dyadic `TEST-SEGMENT` (mark *safe* at `2^(−κ)·λ < ε`),
  `TEST-PATH` ordered by decreasing `2^(−κ)·λ`, the Fig. 4 milestone transfer on
  collision, and the random shortcut optimizer are all faithful — no behavioral
  change. Documented the 2D-grid adaptations (`ζ = ρ`; a finest-level exhaustive
  check guards sub-`ε` pixel obstacles) in `literature/fidelity/sbl.md`. Added
  `tests/test_sbl_fidelity.py`.
- `PSO` reworked so the **default** velocity update is the exact Kennedy &
  Eberhart (1995, §3.6) form `v ← w·v + 2·r₁·(pbest−x) + 2·r₂·(gbest−x)` with a
  `Vmax` clamp: full momentum (no inertia weight, `w = 1.0`; was an adaptive
  Shi-Eberhart 1998 schedule) and acceleration constants `c1 = c2 = 2.0` (were
  `1.5`), with the full social term. The non-1995 robustness heuristics — adaptive
  inertia, adaptive social gain, diversity injection, random immigrants, and swarm
  restart — are now behind an `enable_safeguards` flag (default **off**, exposed as
  an in-app checkbox). Pure 1995 PSO can stall on cluttered maps (documented, like
  pure APF); soundness is unchanged (a reported path is still collision-free).
  Added `tests/test_pso_fidelity.py`.

### Changed

- The map-editing controls (Edit Map, Brush, New Map, Save Map) are now collapsed
  by default behind a **Map Tools** toggle in the left panel, instead of occupying
  an always-visible row; expanding the toggle reveals them in place, and entering
  edit mode auto-expands it. The editing feature is unchanged — just tucked out of
  sight by default.

### Fixed

- `CHOMP` no longer burns the full iteration cap when post-optimizing a
  sampling-based path (the "CHOMP Optimize" button). Threading an RRT path through
  a cluttered map put the optimizer in a stable obstacle-term **limit cycle** (cost
  oscillating by a fixed amount every iteration), which the step-size convergence
  test (`cost_change < 8e-4` or `update_norm < 8e-3`) can never satisfy, so it ran
  to `max_iters` (1000 from the GUI). Two changes: (1) the default `learning_rate`
  is lowered `1.0 → 0.3` to damp the obstacle-gradient overshoot that drove the
  oscillation, and (2) a stagnation stop ends the run once the best cost has not
  improved for 40 iterations. The returned (best collision-free) trajectory is
  unchanged; it just stops in ~100-200 iterations instead of 1000.
- `CHOMP` obstacle term vectorized: the per-waypoint Python loop with scalar
  bilinear sampling (dominated by ~120k scalar `np.clip` calls per run) is replaced
  by array operations over all waypoints, via a new
  `geometry.bilinear_sample_scalar_batch` helper. The math is identical (the same
  Ratliff (2009) functional and gradient; results match the loop to ~1e-12), giving
  ~5-6x faster iterations (~8 ms → ~1.4 ms/iter at 50 waypoints). Together with the
  early-stop fix, a post-optimization run drops from ~8 s to ~0.3 s.

- Split PRM into two selectable planners: **`PRM`** is now the literal Kavraki
  et al. (1996) construction step (a cycle-free forest via same-connected-component
  skipping, using union-find; not asymptotically optimal), and **`sPRM`** is the
  simplified, cycle-keeping, asymptotically-optimal variant (Karaman & Frazzoli
  2011) that was previously labeled "PRM". This lets the tool demonstrate the
  optimality difference directly. (20 planners total.)

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
