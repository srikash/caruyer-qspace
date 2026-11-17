"""Shared test configuration."""
import matplotlib

# Use a non-interactive backend so plotting tests never open a window.
matplotlib.use("Agg")
