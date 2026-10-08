"""Central entry point for available time-integration methods.

Add future solver imports and dictionary entries here.
"""

from explicit_Euler import explicit_euler

solvers = {"explicit_euler": explicit_euler}
