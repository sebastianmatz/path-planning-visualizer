from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..types import Edge, OccupancyGrid, Point


@dataclass
class StepResult:
    """Result of a single planning step.
    
    Attributes:
        edge: Single edge added this step (from, to)
        edges: Multiple edges for batch algorithms
        rejected_point: Point that was rejected (collision)
        done: Whether planning is complete
        found_path: Whether a valid path was found
        path_improved: Whether path quality improved this step (for anytime algorithms)
    """
    edge: Optional[Edge] = None
    edges: Optional[List[Edge]] = None
    rejected_point: Optional[Point] = None
    done: bool = False
    found_path: bool = False
    path_improved: bool = False


class BasePlanner(ABC):
    """Abstract base class for all path planning algorithms.
    
    This class defines the interface that all planners must implement.
    Subclasses provide specific planning algorithms.
    
    Attributes:
        occ: Occupancy grid (True = obstacle)
        h: Grid height
        w: Grid width
        start: Start position
        goal: Goal position
        done: Whether planning is complete
        found_path: Whether a valid path was found
        iteration: Current iteration count
    """
    
    name: str = "Base Planner"
    description: str = "Abstract base class"
    
    def __init__(
        self, 
        occ: OccupancyGrid, 
        start: Point, 
        goal: Point
    ) -> None:
        """Initialize the planner.
        
        Args:
            occ: Occupancy grid (True = obstacle)
            start: Start position
            goal: Goal position
        """
        self.occ = occ
        self.h, self.w = occ.shape
        self.start = start
        self.goal = goal
        self.done = False
        self.found_path = False
        self.iteration = 0
    
    def is_free(self, p: Point) -> bool:
        """Check if a point is in free space.
        
        Args:
            p: Point to check
            
        Returns:
            True if point is within bounds and not on an obstacle
        """
        x, y = p
        return (0 <= x < self.w) and (0 <= y < self.h) and (not self.occ[y, x])
    
    @abstractmethod
    def step_once(self) -> StepResult:
        """Execute one step of the algorithm.
        
        Returns:
            StepResult containing information about this step
        """
        pass
    
    @abstractmethod
    def extract_path(self) -> List[Point]:
        """Extract the found path.
        
        Returns:
            List of points from start to goal, or empty if no path found
        """
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """Return current status string for display.
        
        Returns:
            Human-readable status string
        """
        pass
    
    