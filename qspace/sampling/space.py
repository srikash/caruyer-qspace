"""Conversions between Cartesian and spherical coordinates for 3D points.

Convention: polar angle ``theta`` in ``[0, pi]`` measured from the z-axis,
azimuth ``phi`` measured from the x-axis towards the y-axis.
"""
import numpy as np

from qspace.sampling import sphere


def to_cartesian(r, theta, phi):
    """Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : array-like, shape (K,)
        Radii.
    theta : array-like, shape (K,)
        Polar angles.
    phi : array-like, shape (K,)
        Azimuthal angles.

    Returns
    -------
    points : ndarray, shape (K, 3)
        Cartesian coordinates ``(x, y, z)`` of each point.
    """
    K = r.shape[0]
    points = np.zeros((K, 3))
    points[:, 0] = r * np.sin(theta) * np.cos(phi)
    points[:, 1] = r * np.sin(theta) * np.sin(phi)
    points[:, 2] = r * np.cos(theta)
    return points


def to_spherical(points):
    """Convert Cartesian points to spherical coordinates.

    Points at the origin have radius 0. Their angles are returned as 0, since
    the direction is undefined.

    Parameters
    ----------
    points : array-like, shape (K, 3)
        Cartesian coordinates ``(x, y, z)``.

    Returns
    -------
    r : ndarray, shape (K,)
        Radii (Euclidean norms).
    theta : ndarray, shape (K,)
        Polar angles in ``[0, pi]``.
    phi : ndarray, shape (K,)
        Azimuthal angles.
    """
    points = np.asarray(points, dtype=float)
    r = np.sqrt(np.sum(points**2, 1))
    # Replace zero vectors by a fixed unit vector so that normalisation is
    # defined; their radius is still 0 in the output.
    unit_vectors = np.zeros_like(points)
    nonzero = r > 0
    unit_vectors[nonzero] = points[nonzero] / r[nonzero, np.newaxis]
    unit_vectors[~nonzero] = [0.0, 0.0, 1.0]
    theta, phi = sphere.to_spherical(unit_vectors)
    return r, theta, phi
