"""Qt parameter panels for the planners (GUI layer).

Each planner exposes its tunable parameters through a small ``QWidget`` defined
here rather than in the algorithm module, so the planners -- and the headless
benchmark -- import without PyQt6. ``PARAM_PANELS`` maps a planner name to its
widget class; the GUI builds one per planner and reads values via the widget's
``get_params()`` method.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QSpinBox,
    QWidget,
)

# The sampling-roadmap defaults (PRM/sPRM 500 nodes, FMT* 400) are calibrated for the
# bundled ~600 px demo maps (~290k free pixels). On a much larger map a *fixed* count
# collapses the milestone density and the roadmap fragments into disconnected pieces, so
# the recommended default scales with free-space area to hold density roughly constant.
# This only sets the panel's default; the value stays user-overridable and the planner
# itself is unchanged (sample count is the user-chosen workspace parameter N).
_REF_FREE_AREA = 290_000


def recommended_sample_count(base: int, free_area: int) -> int:
    """Scale a base sample count to a map's free area, keeping density >= the base."""
    if free_area <= 0:
        return base
    return max(base, round(base * free_area / _REF_FREE_AREA))


class RRTParamsWidget(QWidget):
    """Widget for RRT parameterization."""

    def __init__(self) -> None:
        super().__init__()
        layout = QFormLayout()

        self.spin_delta_t = QDoubleSpinBox()
        self.spin_delta_t.setRange(0.5, 200.0)
        self.spin_delta_t.setSingleStep(0.5)
        self.spin_delta_t.setValue(18.0)
        self.spin_delta_t.setToolTip(
            "Fixed integration interval Delta t for the holonomic NEW_STATE step"
        )

        self.spin_goal_radius = QDoubleSpinBox()
        self.spin_goal_radius.setRange(1.0, 200.0)
        self.spin_goal_radius.setSingleStep(1.0)
        self.spin_goal_radius.setValue(20.0)
        self.spin_goal_radius.setToolTip(
            "Goal-region radius for the single-query adaptation of the paper's tree generator"
        )

        self.spin_goal_bias = QDoubleSpinBox()
        self.spin_goal_bias.setRange(0.0, 1.0)
        self.spin_goal_bias.setSingleStep(0.01)
        self.spin_goal_bias.setDecimals(3)
        self.spin_goal_bias.setValue(0.05)
        self.spin_goal_bias.setToolTip(
            "OMPL-style probability of sampling the exact goal state; 0.05 is the typical default"
        )

        self.spin_col = QSpinBox()
        self.spin_col.setRange(10, 500)
        self.spin_col.setValue(80)
        self.spin_col.setToolTip("Raster samples used to validate each accepted edge on the occupancy grid")

        self.spin_max_vertices = QSpinBox()
        self.spin_max_vertices.setRange(2, 200000)
        self.spin_max_vertices.setValue(25000)
        self.spin_max_vertices.setToolTip(
            "Vertex budget K from GENERATE_RRT(x_init, K, Delta t), including the root"
        )

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Delta t:", self.spin_delta_t)
        layout.addRow("Goal region radius:", self.spin_goal_radius)
        layout.addRow("Goal bias:", self.spin_goal_bias)
        layout.addRow("Collision samples:", self.spin_col)
        layout.addRow("Vertex budget K:", self.spin_max_vertices)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        """Get all parameter values as a dictionary."""
        return {
            'delta_t': self.spin_delta_t.value(),
            'goal_region_radius': self.spin_goal_radius.value(),
            'goal_bias': self.spin_goal_bias.value(),
            'collision_samples': self.spin_col.value(),
            'max_vertices': self.spin_max_vertices.value(),
            'seed': self.spin_seed.value(),
        }


class RRTConnectParamsWidget(QWidget):
    """Widget for RRT-Connect parameter configuration."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 200)
        self.spin_step.setValue(18)
        self.spin_step.setToolTip("Distance for each tree expansion step")
        
        self.spin_col = QSpinBox()
        self.spin_col.setRange(10, 500)
        self.spin_col.setValue(80)
        self.spin_col.setToolTip("Number of samples for collision checking along edges")
        
        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(100, 200000)
        self.spin_maxit.setValue(25000)
        self.spin_maxit.setToolTip("Maximum number of iterations")
        
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Step size:", self.spin_step)
        layout.addRow("Collision samples:", self.spin_col)
        layout.addRow("Max iterations:", self.spin_maxit)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step.value(),
            'collision_samples': self.spin_col.value(),
            'max_iters': self.spin_maxit.value(),
            'seed': self.spin_seed.value(),
        }


class BiTRRTParamsWidget(QWidget):
    """Widget for BiTRRT parameter configuration."""

    def __init__(self) -> None:
        super().__init__()
        layout = QFormLayout()

        self.spin_range = QDoubleSpinBox()
        self.spin_range.setRange(1.0, 500.0)
        self.spin_range.setSingleStep(1.0)
        self.spin_range.setValue(24.0)
        self.spin_range.setToolTip("Maximum expansion range per tree extension")

        self.spin_temp_change = QDoubleSpinBox()
        self.spin_temp_change.setRange(0.001, 2.0)
        self.spin_temp_change.setSingleStep(0.01)
        self.spin_temp_change.setDecimals(3)
        self.spin_temp_change.setValue(0.10)
        self.spin_temp_change.setToolTip(
            "OMPL-style temperature increase factor parameter; the actual multiplier is exp(value)"
        )

        self.spin_init_temp = QDoubleSpinBox()
        self.spin_init_temp.setRange(0.001, 100000.0)
        self.spin_init_temp.setDecimals(3)
        self.spin_init_temp.setValue(100.0)
        self.spin_init_temp.setToolTip("Initial transition-test temperature")

        self.spin_frontier_threshold = QDoubleSpinBox()
        self.spin_frontier_threshold.setRange(0.0, 1000.0)
        self.spin_frontier_threshold.setDecimals(3)
        self.spin_frontier_threshold.setSpecialValueText("auto")
        self.spin_frontier_threshold.setValue(0.0)
        self.spin_frontier_threshold.setToolTip(
            "Distance threshold for frontier vs refinement expansion; 0 uses OMPL-style auto scaling"
        )

        self.spin_frontier_ratio = QDoubleSpinBox()
        self.spin_frontier_ratio.setRange(0.01, 10.0)
        self.spin_frontier_ratio.setSingleStep(0.01)
        self.spin_frontier_ratio.setDecimals(3)
        self.spin_frontier_ratio.setValue(0.10)
        self.spin_frontier_ratio.setToolTip(
            "Maximum allowed ratio of non-frontier to frontier expansions"
        )

        self.chk_cost_threshold = QCheckBox("Enable")
        self.chk_cost_threshold.setToolTip(
            "Enable an upper bound on accepted transition costs"
        )

        self.spin_cost_threshold = QDoubleSpinBox()
        self.spin_cost_threshold.setRange(0.0, 1000.0)
        self.spin_cost_threshold.setSingleStep(0.1)
        self.spin_cost_threshold.setDecimals(3)
        self.spin_cost_threshold.setValue(25.0)
        self.spin_cost_threshold.setEnabled(False)
        self.spin_cost_threshold.setToolTip(
            "Maximum motion cost accepted by the transition test when enabled"
        )
        self.chk_cost_threshold.toggled.connect(self.spin_cost_threshold.setEnabled)

        cost_threshold_widget = QWidget()
        cost_threshold_layout = QHBoxLayout(cost_threshold_widget)
        cost_threshold_layout.setContentsMargins(0, 0, 0, 0)
        cost_threshold_layout.addWidget(self.chk_cost_threshold)
        cost_threshold_layout.addWidget(self.spin_cost_threshold)

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 200000)
        self.spin_max_iters.setValue(25000)
        self.spin_max_iters.setToolTip("Maximum number of planning iterations")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Range:", self.spin_range)
        layout.addRow("Temp change factor:", self.spin_temp_change)
        layout.addRow("Initial temperature:", self.spin_init_temp)
        layout.addRow("Frontier threshold:", self.spin_frontier_threshold)
        layout.addRow("Frontier node ratio:", self.spin_frontier_ratio)
        layout.addRow("Cost threshold:", cost_threshold_widget)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'range': self.spin_range.value(),
            'temp_change_factor': self.spin_temp_change.value(),
            'init_temperature': self.spin_init_temp.value(),
            'frontier_threshold': self.spin_frontier_threshold.value(),
            'frontier_node_ratio': self.spin_frontier_ratio.value(),
            'cost_threshold': self.spin_cost_threshold.value() if self.chk_cost_threshold.isChecked() else float('inf'),
            'max_iters': self.spin_max_iters.value(),
            'seed': self.spin_seed.value(),
        }


class KPIECEParamsWidget(QWidget):
    """Widget for KPIECE parameter configuration."""

    def __init__(self) -> None:
        super().__init__()
        layout = QFormLayout()

        self.spin_range = QDoubleSpinBox()
        self.spin_range.setRange(1.0, 500.0)
        self.spin_range.setSingleStep(1.0)
        self.spin_range.setValue(18.0)
        self.spin_range.setToolTip(
            "Maximum local expansion radius used when sampling around the selected motion"
        )

        self.spin_goal_bias = QDoubleSpinBox()
        self.spin_goal_bias.setRange(0.0, 1.0)
        self.spin_goal_bias.setSingleStep(0.01)
        self.spin_goal_bias.setDecimals(3)
        self.spin_goal_bias.setValue(0.02)
        self.spin_goal_bias.setToolTip(
            "Optional goal-directed sampling probability. Small nonzero values often help in this geometric 2D adaptation."
        )

        self.spin_goal_tol = QSpinBox()
        self.spin_goal_tol.setRange(1, 200)
        self.spin_goal_tol.setValue(24)
        self.spin_goal_tol.setToolTip(
            "Distance threshold for snapping a newly added state to the goal"
        )

        self.spin_border_fraction = QDoubleSpinBox()
        self.spin_border_fraction.setRange(0.0, 1.0)
        self.spin_border_fraction.setSingleStep(0.01)
        self.spin_border_fraction.setDecimals(3)
        self.spin_border_fraction.setValue(0.80)
        self.spin_border_fraction.setToolTip(
            "Probability of expanding from a border / exterior cell rather than an interior cell"
        )

        self.spin_progress_alpha = QDoubleSpinBox()
        self.spin_progress_alpha.setRange(0.001, 2.0)
        self.spin_progress_alpha.setSingleStep(0.01)
        self.spin_progress_alpha.setDecimals(3)
        self.spin_progress_alpha.setValue(0.10)
        self.spin_progress_alpha.setToolTip(
            "Positive progress offset alpha used in P = alpha + beta * (coverage increase / simulated distance)"
        )

        self.spin_progress_beta = QDoubleSpinBox()
        self.spin_progress_beta.setRange(0.0, 5.0)
        self.spin_progress_beta.setSingleStep(0.05)
        self.spin_progress_beta.setDecimals(3)
        self.spin_progress_beta.setValue(0.90)
        self.spin_progress_beta.setToolTip(
            "Progress scaling beta used in the paper-style score penalty"
        )

        self.spin_min_valid = QDoubleSpinBox()
        self.spin_min_valid.setRange(0.01, 1.0)
        self.spin_min_valid.setSingleStep(0.01)
        self.spin_min_valid.setDecimals(3)
        self.spin_min_valid.setValue(0.20)
        self.spin_min_valid.setToolTip(
            "Minimum valid fraction required to keep a partial edge when collision stops a motion"
        )

        self.spin_cell_size = QSpinBox()
        self.spin_cell_size.setRange(2, 200)
        self.spin_cell_size.setValue(28)
        self.spin_cell_size.setToolTip(
            "Projected grid cell size in pixels for the single-level KPIECE discretization"
        )

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 200000)
        self.spin_max_iters.setValue(25000)
        self.spin_max_iters.setToolTip("Maximum number of planning iterations")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Range:", self.spin_range)
        layout.addRow("Goal bias:", self.spin_goal_bias)
        layout.addRow("Goal tolerance:", self.spin_goal_tol)
        layout.addRow("Border fraction:", self.spin_border_fraction)
        layout.addRow("Progress alpha:", self.spin_progress_alpha)
        layout.addRow("Progress beta:", self.spin_progress_beta)
        layout.addRow("Min valid fraction:", self.spin_min_valid)
        layout.addRow("Cell size:", self.spin_cell_size)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'range': self.spin_range.value(),
            'goal_bias': self.spin_goal_bias.value(),
            'goal_tolerance': self.spin_goal_tol.value(),
            'border_fraction': self.spin_border_fraction.value(),
            'progress_alpha': self.spin_progress_alpha.value(),
            'progress_beta': self.spin_progress_beta.value(),
            'min_valid_path_fraction': self.spin_min_valid.value(),
            'cell_size': self.spin_cell_size.value(),
            'max_iters': self.spin_max_iters.value(),
            'seed': self.spin_seed.value(),
        }


class RRTStarParamsWidget(QWidget):
    """Widget for RRT* parameter configuration."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 200)
        self.spin_step.setValue(18)
        self.spin_step.setToolTip("Distance for each tree expansion step")
        
        self.spin_goal_rate = QDoubleSpinBox()
        self.spin_goal_rate.setRange(0.0, 1.0)
        self.spin_goal_rate.setSingleStep(0.01)
        self.spin_goal_rate.setValue(0.10)
        self.spin_goal_rate.setToolTip("Probability of sampling the goal directly")
        
        self.spin_goal_tol = QSpinBox()
        self.spin_goal_tol.setRange(1, 200)
        self.spin_goal_tol.setValue(20)
        self.spin_goal_tol.setToolTip("Distance threshold to consider goal reached")
        
        self.spin_search_radius = QSpinBox()
        self.spin_search_radius.setRange(0, 500)
        self.spin_search_radius.setValue(0)
        self.spin_search_radius.setToolTip(
            "0 = auto: shrinking RGG radius min(gamma*(log n/n)^(1/2), step), "
            "asymptotically optimal (Karaman & Frazzoli 2011). "
            ">0 = fixed radius (legacy)."
        )
        
        self.spin_col = QSpinBox()
        self.spin_col.setRange(10, 500)
        self.spin_col.setValue(80)
        self.spin_col.setToolTip("Number of samples for collision checking along edges")
        
        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(100, 200000)
        self.spin_maxit.setValue(25000)
        self.spin_maxit.setToolTip("Maximum number of iterations")
        
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Step size:", self.spin_step)
        layout.addRow("Goal sample rate:", self.spin_goal_rate)
        layout.addRow("Goal tolerance:", self.spin_goal_tol)
        layout.addRow("Search radius (0=auto):", self.spin_search_radius)
        layout.addRow("Collision samples:", self.spin_col)
        layout.addRow("Max iterations:", self.spin_maxit)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step.value(),
            'goal_sample_rate': self.spin_goal_rate.value(),
            'goal_tolerance': self.spin_goal_tol.value(),
            'search_radius': self.spin_search_radius.value(),
            'collision_samples': self.spin_col.value(),
            'max_iters': self.spin_maxit.value(),
            'seed': self.spin_seed.value(),
        }


class PRMParamsWidget(QWidget):
    """Parameters widget for PRM planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_samples = QSpinBox()
        self.spin_num_samples.setRange(50, 10000)
        self.spin_num_samples.setValue(500)
        self._base_samples = 500
        self._auto_samples = 500
        self.spin_num_samples.setToolTip(
            "Number of random samples to generate (auto-scaled to map size; editable)"
        )
        
        self.spin_k_neighbors = QSpinBox()
        self.spin_k_neighbors.setRange(3, 50)
        self.spin_k_neighbors.setValue(15)
        self.spin_k_neighbors.setToolTip("Number of nearest neighbors to connect")
        
        self.spin_max_edge_dist = QSpinBox()
        self.spin_max_edge_dist.setRange(10, 500)
        self.spin_max_edge_dist.setValue(100)
        self.spin_max_edge_dist.setToolTip("Maximum edge length")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Num samples:", self.spin_num_samples)
        layout.addRow("K neighbors:", self.spin_k_neighbors)
        layout.addRow("Max edge dist:", self.spin_max_edge_dist)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_samples': self.spin_num_samples.value(),
            'k_neighbors': self.spin_k_neighbors.value(),
            'max_edge_dist': self.spin_max_edge_dist.value(),
            'seed': self.spin_seed.value(),
        }

    def update_for_map(self, free_area: int) -> None:
        """Set the recommended default sample count for a freshly loaded map.

        Leaves the value alone if the user has overridden it (current value differs
        from the last auto-set value). ``max_edge_dist`` is intentionally not scaled:
        holding sample density constant keeps milestone spacing constant, so the same
        connection radius stays valid across map sizes.
        """
        if self.spin_num_samples.value() != self._auto_samples:
            return
        rec = min(self.spin_num_samples.maximum(),
                  recommended_sample_count(self._base_samples, free_area))
        self.spin_num_samples.setValue(rec)
        self._auto_samples = rec


class SBLParamsWidget(QWidget):
    """Parameters widget for the SBL planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(100, 200000)
        self.spin_maxit.setValue(12000)
        self.spin_maxit.setToolTip("Maximum number of milestone expansion iterations")

        self.spin_rho = QSpinBox()
        self.spin_rho.setRange(2, 400)
        self.spin_rho.setValue(45)
        self.spin_rho.setToolTip("SBL distance threshold rho for local expansion and tree connection")

        self.spin_resolution = QSpinBox()
        self.spin_resolution.setRange(1, 50)
        self.spin_resolution.setValue(4)
        self.spin_resolution.setToolTip("Lazy segment resolution epsilon in pixels")

        self.spin_candidates = QSpinBox()
        self.spin_candidates.setRange(1, 20)
        self.spin_candidates.setValue(6)
        self.spin_candidates.setToolTip("Maximum number of shrinking-neighborhood candidates per expansion")

        self.spin_grid_cells = QSpinBox()
        self.spin_grid_cells.setRange(2, 50)
        self.spin_grid_cells.setValue(10)
        self.spin_grid_cells.setToolTip("Spatial indexing resolution per tree (cells per axis)")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Max iterations:", self.spin_maxit)
        layout.addRow("Rho:", self.spin_rho)
        layout.addRow("Lazy resolution:", self.spin_resolution)
        layout.addRow("Candidates:", self.spin_candidates)
        layout.addRow("Grid cells:", self.spin_grid_cells)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'max_iters': self.spin_maxit.value(),
            'rho': self.spin_rho.value(),
            'lazy_resolution': self.spin_resolution.value(),
            'max_candidates': self.spin_candidates.value(),
            'grid_cells': self.spin_grid_cells.value(),
            'seed': self.spin_seed.value(),
        }


class FMTStarParamsWidget(QWidget):
    """Parameters widget for FMT* planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_samples = QSpinBox()
        self.spin_num_samples.setRange(50, 5000)
        self.spin_num_samples.setValue(400)
        self._base_samples = 400
        self._auto_samples = 400
        self.spin_num_samples.setToolTip(
            "Number of random samples (higher = more robust, slower; auto-scaled to map size, editable)"
        )
        
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.0, 300.0)  # 0 = auto
        self.spin_radius.setSingleStep(10.0)
        self.spin_radius.setValue(0.0)  # Auto by default
        self.spin_radius.setToolTip("Connection radius (0 = auto-compute FMT* radius for the 2D free space)")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Num samples:", self.spin_num_samples)
        layout.addRow("Radius (0=auto):", self.spin_radius)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_samples': self.spin_num_samples.value(),
            'radius': self.spin_radius.value() if self.spin_radius.value() > 0 else None,
            'seed': self.spin_seed.value(),
        }

    def update_for_map(self, free_area: int) -> None:
        """Set the recommended default sample count for a freshly loaded map
        (skipped if the user has overridden it). FMT*'s radius already auto-scales."""
        if self.spin_num_samples.value() != self._auto_samples:
            return
        rec = min(self.spin_num_samples.maximum(),
                  recommended_sample_count(self._base_samples, free_area))
        self.spin_num_samples.setValue(rec)
        self._auto_samples = rec


class BITStarParamsWidget(QWidget):
    """Parameters widget for BIT* planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(50, 2000)
        self.spin_batch_size.setValue(200)
        self.spin_batch_size.setToolTip("Samples per batch")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 50000)
        self.spin_max_iters.setValue(10000)
        self.spin_max_iters.setToolTip("Maximum iterations")
        
        self.spin_rewire_radius = QDoubleSpinBox()
        self.spin_rewire_radius.setRange(0.0, 300.0)
        self.spin_rewire_radius.setSingleStep(5.0)
        self.spin_rewire_radius.setValue(0.0)
        self.spin_rewire_radius.setToolTip("Connection / rewiring radius (0 = auto)")

        self.spin_step_size = QDoubleSpinBox()
        self.spin_step_size.setRange(4.0, 100.0)
        self.spin_step_size.setSingleStep(2.0)
        self.spin_step_size.setValue(26.0)
        self.spin_step_size.setToolTip("Local connection length cap (only used when the cap is enabled)")

        self.check_cap_edges = QCheckBox("Cap edges at step size (visualization)")
        self.check_cap_edges.setChecked(False)
        self.check_cap_edges.setToolTip(
            "Off (default): connect within the full RGG radius (paper-faithful, "
            "asymptotically optimal). On: cap local connections at the step size "
            "for a tidier visualization."
        )

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Batch size:", self.spin_batch_size)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Rewire radius:", self.spin_rewire_radius)
        layout.addRow("Step size:", self.spin_step_size)
        layout.addRow("", self.check_cap_edges)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'batch_size': self.spin_batch_size.value(),
            'max_iters': self.spin_max_iters.value(),
            'rewire_radius': self.spin_rewire_radius.value() if self.spin_rewire_radius.value() > 0 else None,
            'step_size': self.spin_step_size.value(),
            'cap_edges_to_step': self.check_cap_edges.isChecked(),
            'seed': self.spin_seed.value(),
        }


class AStarParamsWidget(QWidget):
    """Parameters widget for A* planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_grid_size = QSpinBox()
        self.spin_grid_size.setRange(1, 20)
        self.spin_grid_size.setValue(5)
        self.spin_grid_size.setToolTip("Grid cell size (lower = finer but slower)")
        
        self.check_diagonal = QCheckBox()
        self.check_diagonal.setChecked(True)
        self.check_diagonal.setToolTip("Allow diagonal movement")
        
        layout.addRow("Grid size:", self.spin_grid_size)
        layout.addRow("Diagonal:", self.check_diagonal)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'grid_size': self.spin_grid_size.value(),
            'allow_diagonal': self.check_diagonal.isChecked(),
        }


class DijkstraParamsWidget(QWidget):
    """Parameters widget for Dijkstra planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_grid_size = QSpinBox()
        self.spin_grid_size.setRange(1, 20)
        self.spin_grid_size.setValue(5)
        self.spin_grid_size.setToolTip("Grid cell size")
        
        self.check_diagonal = QCheckBox()
        self.check_diagonal.setChecked(True)
        self.check_diagonal.setToolTip("Allow diagonal movement")
        
        layout.addRow("Grid size:", self.spin_grid_size)
        layout.addRow("Diagonal:", self.check_diagonal)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'grid_size': self.spin_grid_size.value(),
            'allow_diagonal': self.check_diagonal.isChecked(),
        }


class APFParamsWidget(QWidget):
    """Parameters widget for APF planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_step_size = QDoubleSpinBox()
        self.spin_step_size.setRange(0.5, 20.0)
        self.spin_step_size.setSingleStep(0.5)
        self.spin_step_size.setValue(5.0)
        self.spin_step_size.setToolTip("Maximum speed V_max (per-step displacement cap; Khatib Eq. 16-17)")

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 50000)
        self.spin_max_iters.setValue(5000)
        self.spin_max_iters.setToolTip("Maximum iterations")

        self.spin_goal_gain = QDoubleSpinBox()
        self.spin_goal_gain.setRange(0.01, 100.0)
        self.spin_goal_gain.setSingleStep(0.1)
        self.spin_goal_gain.setValue(1.0)
        self.spin_goal_gain.setToolTip("Attractive stiffness k in F_att = -k(x - x_goal) (Khatib Eq. 12)")

        self.spin_obstacle_gain = QDoubleSpinBox()
        self.spin_obstacle_gain.setRange(1.0, 10000.0)
        self.spin_obstacle_gain.setSingleStep(100.0)
        self.spin_obstacle_gain.setValue(1000.0)
        self.spin_obstacle_gain.setToolTip("Repulsive gain eta in the FIRAS force (Khatib Eq. 20)")

        self.spin_obstacle_dist = QSpinBox()
        self.spin_obstacle_dist.setRange(5, 100)
        self.spin_obstacle_dist.setValue(30)
        self.spin_obstacle_dist.setToolTip("Obstacle influence limit rho_0 (FIRAS); no repulsion beyond it")

        self.chk_escape = QCheckBox("Enable local-minimum escape (non-paper)")
        self.chk_escape.setChecked(False)
        self.chk_escape.setToolTip(
            "Off (default): pure APF; the robot stalls at local minima as Khatib documents. "
            "On: add a stochastic kick to try to escape (a heuristic not in the paper)."
        )

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for the optional escape perturbations")

        layout.addRow("Max speed (V_max):", self.spin_step_size)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Attractive gain (k):", self.spin_goal_gain)
        layout.addRow("Repulsive gain (eta):", self.spin_obstacle_gain)
        layout.addRow("Influence dist (rho_0):", self.spin_obstacle_dist)
        layout.addRow("", self.chk_escape)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step_size.value(),
            'max_iters': self.spin_max_iters.value(),
            'goal_gain': self.spin_goal_gain.value(),
            'obstacle_gain': self.spin_obstacle_gain.value(),
            'obstacle_dist': self.spin_obstacle_dist.value(),
            'enable_escape': self.chk_escape.isChecked(),
            'seed': self.spin_seed.value(),
        }


class CHOMPParamsWidget(QWidget):
    """Parameter widget for CHOMP planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 500)
        self.spin_num_points.setValue(50)
        self.spin_num_points.setToolTip("Number of waypoints in trajectory")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(10, 50000)
        self.spin_max_iters.setValue(1000)
        self.spin_max_iters.setToolTip("Maximum optimization iterations")
        
        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setRange(0.001, 10.0)
        self.spin_learning_rate.setSingleStep(0.1)
        self.spin_learning_rate.setValue(0.3)
        self.spin_learning_rate.setToolTip("Base step size (1/lambda) for the covariant CHOMP update; lower damps obstacle-term overshoot")
        
        self.spin_smoothness_weight = QDoubleSpinBox()
        self.spin_smoothness_weight.setRange(0.0, 100.0)
        self.spin_smoothness_weight.setSingleStep(0.1)
        self.spin_smoothness_weight.setValue(1.0)
        self.spin_smoothness_weight.setToolTip("Weight for the CHOMP smoothness prior")
        
        self.spin_obstacle_weight = QDoubleSpinBox()
        self.spin_obstacle_weight.setRange(0.0, 1000.0)
        self.spin_obstacle_weight.setSingleStep(1.0)
        self.spin_obstacle_weight.setValue(100.0)
        self.spin_obstacle_weight.setToolTip("Weight for the obstacle functional")
        
        self.spin_obstacle_epsilon = QSpinBox()
        self.spin_obstacle_epsilon.setRange(1, 100)
        self.spin_obstacle_epsilon.setValue(20)
        self.spin_obstacle_epsilon.setToolTip("Distance field epsilon (obstacle influence range)")

        self.spin_path_length_weight = QDoubleSpinBox()
        self.spin_path_length_weight.setRange(0.0, 10.0)
        self.spin_path_length_weight.setSingleStep(0.05)
        self.spin_path_length_weight.setValue(0.0)
        self.spin_path_length_weight.setToolTip("Optional extra arc-length penalty beyond standard CHOMP")
        
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Learning rate:", self.spin_learning_rate)
        layout.addRow("Smoothness weight:", self.spin_smoothness_weight)
        layout.addRow("Obstacle weight:", self.spin_obstacle_weight)
        layout.addRow("Obstacle epsilon:", self.spin_obstacle_epsilon)
        layout.addRow("Path length weight:", self.spin_path_length_weight)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'learning_rate': self.spin_learning_rate.value(),
            'smoothness_weight': self.spin_smoothness_weight.value(),
            'obstacle_weight': self.spin_obstacle_weight.value(),
            'obstacle_epsilon': self.spin_obstacle_epsilon.value(),
            'path_length_weight': self.spin_path_length_weight.value(),
        }


class STOMPParamsWidget(QWidget):
    """Parameters widget for STOMP planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 200)
        self.spin_num_points.setValue(50)
        self.spin_num_points.setToolTip("Number of waypoints in trajectory")

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(10, 20000)
        self.spin_max_iters.setValue(500)
        self.spin_max_iters.setToolTip("Maximum optimization iterations")

        self.spin_num_rollouts = QSpinBox()
        self.spin_num_rollouts.setRange(5, 100)
        self.spin_num_rollouts.setValue(20)
        self.spin_num_rollouts.setToolTip("Number K of noisy trajectory rollouts per iteration")

        self.spin_noise_std = QDoubleSpinBox()
        self.spin_noise_std.setRange(0.1, 50.0)
        self.spin_noise_std.setSingleStep(1.0)
        self.spin_noise_std.setValue(10.0)
        self.spin_noise_std.setToolTip("Magnitude of the exploration noise (the only open STOMP parameter)")

        self.spin_epsilon = QDoubleSpinBox()
        self.spin_epsilon.setRange(0.0, 100.0)
        self.spin_epsilon.setSingleStep(1.0)
        self.spin_epsilon.setValue(10.0)
        self.spin_epsilon.setToolTip("Obstacle clearance margin epsilon in the cost max(eps - d, 0) (Eq. 13)")

        self.spin_h = QDoubleSpinBox()
        self.spin_h.setRange(1.0, 100.0)
        self.spin_h.setSingleStep(1.0)
        self.spin_h.setValue(10.0)
        self.spin_h.setToolTip("Cost sensitivity h in the probability exponent (Eq. 11; paper uses 10)")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Num rollouts (K):", self.spin_num_rollouts)
        layout.addRow("Noise std:", self.spin_noise_std)
        layout.addRow("Clearance margin (eps):", self.spin_epsilon)
        layout.addRow("Cost sensitivity (h):", self.spin_h)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'num_rollouts': self.spin_num_rollouts.value(),
            'noise_std': self.spin_noise_std.value(),
            'epsilon': self.spin_epsilon.value(),
            'h': self.spin_h.value(),
            'seed': self.spin_seed.value(),
        }


class TrajOptParamsWidget(QWidget):
    """Parameters widget for TrajOpt planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 200)
        self.spin_num_points.setValue(50)
        self.spin_num_points.setToolTip("Number of waypoints")

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(10, 20000)
        self.spin_max_iters.setValue(1000)
        self.spin_max_iters.setToolTip("Maximum iterations")

        self.spin_trust_region = QDoubleSpinBox()
        self.spin_trust_region.setRange(1.0, 100.0)
        self.spin_trust_region.setSingleStep(5.0)
        self.spin_trust_region.setValue(20.0)
        self.spin_trust_region.setToolTip("Initial trust-region box size")

        self.spin_collision_weight = QDoubleSpinBox()
        self.spin_collision_weight.setRange(1.0, 1000.0)
        self.spin_collision_weight.setSingleStep(10.0)
        self.spin_collision_weight.setValue(100.0)
        self.spin_collision_weight.setToolTip("Initial collision penalty coefficient")

        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Trust region:", self.spin_trust_region)
        layout.addRow("Collision weight:", self.spin_collision_weight)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'trust_region': self.spin_trust_region.value(),
            'collision_weight': self.spin_collision_weight.value(),
        }


class ITOMPParamsWidget(QWidget):
    """Parameters widget for ITOMP planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 100)
        self.spin_num_points.setValue(30)
        self.spin_num_points.setToolTip("Trajectory waypoints")

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 10000)
        self.spin_max_iters.setValue(1000)
        self.spin_max_iters.setToolTip("Maximum iterations")

        self.spin_replan_interval = QSpinBox()
        self.spin_replan_interval.setRange(1, 100)
        self.spin_replan_interval.setValue(20)
        self.spin_replan_interval.setToolTip("Iterations between execution-horizon advances")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed (kept for interface compatibility; optimizer is deterministic)")

        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iters:", self.spin_max_iters)
        layout.addRow("Replan interval:", self.spin_replan_interval)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'replan_interval': self.spin_replan_interval.value(),
            'seed': self.spin_seed.value(),
        }


class GPMPParamsWidget(QWidget):
    """Parameters widget for GPMP planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 100)
        self.spin_num_points.setValue(25)
        self.spin_num_points.setToolTip("Number of GP support states")

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(50, 10000)
        self.spin_max_iters.setValue(200)
        self.spin_max_iters.setToolTip("Maximum Gauss-Newton iterations")

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 50.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(6.0)
        self.spin_sigma.setToolTip("GP process-noise scale. Higher = softer prior.")

        layout.addRow("Support states:", self.spin_num_points)
        layout.addRow("Max iters:", self.spin_max_iters)
        layout.addRow("Prior sigma:", self.spin_sigma)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'sigma': self.spin_sigma.value(),
        }


class PSOParamsWidget(QWidget):
    """Parameters widget for PSO planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_particles = QSpinBox()
        self.spin_num_particles.setRange(10, 200)
        self.spin_num_particles.setValue(30)
        self.spin_num_particles.setToolTip("Number of particles")
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(5, 100)
        self.spin_num_points.setValue(30)
        self.spin_num_points.setToolTip("Waypoints per path")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(50, 5000)
        self.spin_max_iters.setValue(1200)
        self.spin_max_iters.setToolTip("Maximum iterations")

        self.chk_safeguards = QCheckBox("Enable safeguards (non-1995)")
        self.chk_safeguards.setChecked(False)
        self.chk_safeguards.setToolTip(
            "Off = exact Kennedy & Eberhart (1995) update: v += 2*r1*(pbest-x) + 2*r2*(gbest-x), "
            "Vmax clamp, full momentum (w=1.0).\n"
            "On = non-1995 robustness: adaptive inertia/social gain, diversity injection, "
            "random immigrants, and swarm restart for cluttered maps."
        )

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Particles:", self.spin_num_particles)
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iters:", self.spin_max_iters)
        layout.addRow(self.chk_safeguards)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'num_particles': self.spin_num_particles.value(),
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'enable_safeguards': self.chk_safeguards.isChecked(),
            'seed': self.spin_seed.value(),
        }


PARAM_PANELS: dict[str, type[QWidget]] = {
    'RRT': RRTParamsWidget,
    'RRT-Connect': RRTConnectParamsWidget,
    'BiTRRT': BiTRRTParamsWidget,
    'KPIECE': KPIECEParamsWidget,
    'RRT*': RRTStarParamsWidget,
    'PRM': PRMParamsWidget,
    'sPRM': PRMParamsWidget,
    'SBL': SBLParamsWidget,
    'FMT*': FMTStarParamsWidget,
    'BIT*': BITStarParamsWidget,
    'A*': AStarParamsWidget,
    'Dijkstra': DijkstraParamsWidget,
    'APF': APFParamsWidget,
    'CHOMP': CHOMPParamsWidget,
    'STOMP': STOMPParamsWidget,
    'TrajOpt': TrajOptParamsWidget,
    'ITOMP': ITOMPParamsWidget,
    'GPMP': GPMPParamsWidget,
    'PSO': PSOParamsWidget,
}
