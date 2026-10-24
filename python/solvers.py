"""Central entry point for available time-integration methods.

Add future solver imports and dictionary entries here.
"""

from explicit_Euler import explicit_euler
from rk4 import rk4
from implicit_Euler import implicit_euler

solvers = {"explicit_euler": explicit_euler,
           "rk4": rk4,
           "implicit_euler": implicit_euler}
