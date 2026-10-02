"""
Project Chronos - Optimal Execution Engine
Implements Almgren-Chriss Framework with Transient & Permanent Market Impact.
"""
from .almgren_chriss import AlmgrenChrissExecutor, AlmgrenChrissTrajectory, ExecutionSlice

__all__ = ["AlmgrenChrissExecutor", "AlmgrenChrissTrajectory", "ExecutionSlice"]
