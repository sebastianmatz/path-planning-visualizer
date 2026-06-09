"""Pytest configuration.

Makes the top-level ``path_planning_visualizer`` module importable when tests
run from the repository root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
