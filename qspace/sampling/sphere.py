"""Sampling of directions on the unit sphere.

The functions here generate sets of antipodally symmetric directions for
diffusion gradient schemes: :func:`jones` (optimized electrostatic repulsion,
for small sets), :func:`koay` (analytic, for large sets) and
:func:`random_uniform` (random).
"""
import os

import numpy as np

from qspace.sampling import multishell as ms


__location__ = os.path.dirname(__file__)


def jones(K):
    """Return K directions minimizing Jones' electrostatic repulsion energy.

    For ``K <= 256`` the points are read from precomputed tables shipped with
    the package. For larger ``K`` they are optimized on the fly with
    :func:`qspace.sampling.multishell.optimize`, which is slow.

    Parameters
    ----------
    K : int
        Number of directions.

    Returns
    -------
    ndarray, shape (K, 3)
        Unit vectors.
    """
    if K > 256:
        nb_shells = 1
        nb_points_per_shell = [K]
        weights = np.ones((1, 1))
        return ms.optimize(nb_shells, nb_points_per_shell, weights)
    filename = os.path.join(__location__, "data", "jones_%03d.txt" % K)
    dirs = np.loadtxt(filename)
    return dirs.reshape((K, 3))


def koay(K):
    """Return K antipodally symmetric directions generated analytically.

    Uses Koay's method, which places points on latitude circles with a number
    of points proportional to the circle's length.

    Parameters
    ----------
    K : int
        Number of directions.

    Returns
    -------
    ndarray, shape (K, 3)
        Unit vectors in the upper hemisphere.

    References
    ----------
    Koay, C. G. (2011). A simple scheme for generating nearly uniform
    distribution of antipodally symmetric points on the unit sphere. Journal of
    Computational Science, 2(4), 377-381.
    """
    nb_latitudes = int(np.sqrt(K * np.pi / 8))
    ks = np.zeros(nb_latitudes, dtype=int)
    latitudes = (0.5 + np.arange(nb_latitudes)) * np.pi / (2 * nb_latitudes)
    ratios = 2 * np.sin(latitudes) * np.sin(np.pi / (4 * nb_latitudes))
    thetas = np.zeros(K)
    phis = np.zeros(K)
    current_index = 0
    for n in range(nb_latitudes):
        if n < nb_latitudes - 1:
            ks[n] = int(np.rint(ratios[n] * K))
        else:
            ks[n] = K - np.sum(ks)
        thetas[current_index:current_index + ks[n]] = latitudes[n]
        phis[current_index:current_index + ks[n]] = (0.5 + np.arange(ks[n])) \
            * 2 * np.pi / ks[n]
        current_index += ks[n]
    return to_cartesian(thetas, phis)


def to_cartesian(theta, phi):
    """Convert polar and azimuthal angles of unit vectors to Cartesian.

    Parameters
    ----------
    theta : array-like, shape (K,)
        Polar angles.
    phi : array-like, shape (K,)
        Azimuthal angles.

    Returns
    -------
    points : ndarray, shape (K, 3)
        Unit vectors.
    """
    K = theta.shape[0]
    points = np.zeros((K, 3))
    points[:, 0] = np.sin(theta) * np.cos(phi)
    points[:, 1] = np.sin(theta) * np.sin(phi)
    points[:, 2] = np.cos(theta)
    return points


def to_spherical(points):
    """Convert unit vectors to polar and azimuthal angles.

    Parameters
    ----------
    points : array-like, shape (K, 3)
        Unit vectors. A ValueError is raised if any vector is not of unit norm.

    Returns
    -------
    theta : ndarray, shape (K,)
        Polar angles in ``[0, pi]``.
    phi : ndarray, shape (K,)
        Azimuthal angles.

    Raises
    ------
    ValueError
        If the input vectors are not of unit norm.
    """
    norms = np.sqrt(np.sum(points**2, 1))
    if not np.allclose(norms, 1.0):
        raise ValueError("sphere.to_spherical expects unit vectors.")
    theta = np.arccos(points[:, 2])
    phi = np.arctan2(points[:, 1], points[:, 0])
    return theta, phi


def random_uniform(K):
    """Return K pseudo-random unit vectors, uniform on the whole sphere.

    Parameters
    ----------
    K : int
        Number of directions to generate.

    Returns
    -------
    ndarray, shape (K, 3)
        Unit vectors drawn from the global NumPy random state.
    """
    phi = 2 * np.pi * np.random.rand(K)

    r = 2 * np.sqrt(np.random.rand(K))
    theta = 2 * np.arcsin(r / 2)

    points = np.zeros((K, 3))
    points[:, 0] = np.sin(theta) * np.cos(phi)
    points[:, 1] = np.sin(theta) * np.sin(phi)
    points[:, 2] = np.cos(theta)

    return points
