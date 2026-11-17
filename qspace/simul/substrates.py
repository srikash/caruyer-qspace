"""Tissue substrates for Monte-Carlo diffusion simulations with Camino.

A substrate describes the geometry the walkers diffuse in. Each class exposes
``camino_options``, a dictionary of Camino command-line options that
:class:`qspace.simul.camino.CaminoSimulation` forwards to ``datasynth``.
"""
from abc import ABC
import numpy as np


_PACKINGS = ("hexagonal", "square")


class Substrate(ABC):
    """Base class of simulation substrates.

    Attributes
    ----------
    name : str
        Human-readable name of the substrate family.
    camino_options : dict of str to str
        Camino options describing the geometry.
    """

    name = "Substrate"


class Free(Substrate):
    """Free diffusion, with no obstacles."""

    def __init__(self):
        self.camino_options = {"-substrate": "empty"}

    def __str__(self):
        return "free"


class CylindersLattice(Substrate):
    """Infinite parallel cylinders arranged on a 2d lattice.

    Exactly one of ``separation`` (centre-to-centre distance between nearest
    cylinders) or ``density`` (intra-axonal volume fraction) must be given. The
    other is derived from the packing geometry.

    Parameters
    ----------
    radius : float
        Cylinder radius, in m.
    packing : {"hexagonal", "square"}
        Lattice packing. Case-insensitive.
    separation : float, optional
        Distance between the axes of nearest cylinders, in m. Must be at least
        ``2 * radius``.
    density : float, optional
        Volume fraction occupied by the cylinders, in ``(0, 1]``.

    Raises
    ------
    ValueError
        If the packing is unknown, if both or neither of ``separation`` and
        ``density`` are given, or if the separation is smaller than
        ``2 * radius``.
    """

    def __init__(self, radius, packing="hexagonal", separation=None,
                 density=None):
        packing = packing.lower()
        if packing not in _PACKINGS:
            raise ValueError("packing is either 'hexagonal' or 'square'.")
        if separation is not None and density is not None:
            raise ValueError("provide either separation or density, not both.")
        if separation is None and density is None:
            raise ValueError("provide either separation or density.")
        if density is not None and not 0 < density <= 1:
            raise ValueError("density must lie in (0, 1].")
        if separation is not None and separation <= 0:
            raise ValueError("separation must be positive.")

        self.camino_options = {"-geometry": "cylinder"}
        self.radius = radius
        self.packing = packing
        if separation is not None:
            self.separation = separation
            self.density = _cylinder_density_from_separation(
                radius, separation, packing)
        else:
            self.separation = _cylinder_separation_from_density(
                radius, density, packing)
            self.density = density
        if self.separation < 2 * self.radius:
            raise ValueError("cylinders separation should be >= 2*radius.")

        self.camino_options["-packing"] = "hex" if packing == "hexagonal" \
            else "square"
        self.camino_options["-cylinderrad"] = "%.3e" % self.radius
        self.camino_options["-cylindersep"] = "%.3e" % self.separation

    def __str__(self):
        return (f"CylindersLattice_{self.packing}_r={self.radius:.3e}_"
                f"d={self.density:.2f}")


def _cylinder_separation_from_density(radius, density, packing="hexagonal"):
    """Return the nearest-neighbour axis separation giving ``density``.

    Hexagonal: ``s = sqrt(2 pi r^2 / (sqrt(3) d))``. Square: ``s = sqrt(pi r^2 / d)``.
    """
    if packing == "hexagonal":
        return np.sqrt(2 * np.pi * radius**2 / (np.sqrt(3) * density))
    elif packing == "square":
        return np.sqrt(np.pi * radius**2 / density)
    raise ValueError("Packing %s not implemented." % packing)


def _cylinder_density_from_separation(radius, separation, packing="hexagonal"):
    """Return the volume fraction of a lattice with nearest-neighbour axis
    separation ``separation``. Inverse of
    :func:`_cylinder_separation_from_density`."""
    if packing == "hexagonal":
        return (2 * np.pi * radius**2) / (np.sqrt(3) * separation**2)
    elif packing == "square":
        return np.pi * radius**2 / separation**2
    raise ValueError("Packing %s not implemented." % packing)


def substrate(name, **params):
    """Create a substrate from its class name and parameters.

    Parameters
    ----------
    name : {"Free", "CylindersLattice"}
        Substrate family.
    **params
        Passed to the constructor of the chosen class. For
        ``"CylindersLattice"``: ``radius`` and one of ``density`` or
        ``separation``, with optional ``packing``.

    Returns
    -------
    Substrate

    Raises
    ------
    ValueError
        If ``name`` is not a known substrate.
    """
    if name == "Free":
        return Free()
    if name == "CylindersLattice":
        return CylindersLattice(**params)
    raise ValueError("Unknown substrate %r; expected 'Free' or "
                     "'CylindersLattice'." % name)
