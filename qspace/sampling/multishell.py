"""Optimized multiple-shell sampling schemes with uniform angular coverage.

Directions are placed on several shells by minimizing a Coulomb-like repulsion
energy between all pairs of points, with antipodal symmetry. Pairs of shells
can be coupled with weights, so that points of different shells are spread
jointly rather than independently.

This implements the method of Caruyer, Lenglet, Sapiro and Deriche (2013),
Magnetic Resonance in Medicine 69(6), 1534-1540.
"""
import numpy as np
from scipy import optimize as scopt


def equality_constraints(vects, *args):
    """Spherical equality constraint: zero when every vector lies on the unit
    sphere.

    Parameters
    ----------
    vects : array-like, shape (N * 3,)
        Flattened vectors.

    Returns
    -------
    ndarray, shape (N,)
        Squared norm minus one, for each vector.
    """
    N = vects.shape[0] // 3
    vects = vects.reshape((N, 3))
    return (vects ** 2).sum(1) - 1.0


def grad_equality_constraints(vects, *args):
    """Gradient of :func:`equality_constraints` (the surface normals).

    Parameters
    ----------
    vects : array-like, shape (N * 3,)
        Flattened vectors.

    Returns
    -------
    ndarray, shape (N, N * 3)
        ``grad[i, j]`` is the partial derivative of constraint ``i`` with
        respect to coordinate ``j``.
    """
    N = vects.shape[0] // 3
    vects = vects.reshape((N, 3))
    vects = (vects.T / np.sqrt((vects ** 2).sum(1))).T
    grad = np.zeros((N, N * 3))
    for i in range(3):
        grad[:, i * N:(i + 1) * N] = np.diag(vects[:, i])
    return grad


def f(vects, weight_matrix, antipodal=True, alpha=1.0):
    r"""Electrostatic-repulsion energy of a set of vectors.

    The pairwise energy is ``w_ij / |x_i - x_j|^(2 alpha)``. With
    ``antipodal=True``, the image ``-x_j`` is also counted.

    Parameters
    ----------
    vects : array-like, shape (N * 3,)
        Flattened vectors.
    weight_matrix : array-like, shape (N, N)
        Contribution weight of each pair of points.
    antipodal : bool
        If True, include interactions with antipodal images.
    alpha : float
        Power of the repulsion. The pairwise term is ``1 / d^(2 alpha)``, where
        ``d`` is the Euclidean distance, so the default ``alpha=1`` gives an
        inverse-square (``1/r^2``) energy, not Coulomb's ``1/r``.

    Returns
    -------
    float
        Total energy, summed over all pairs.
    """
    epsilon = 1e-9
    N = vects.shape[0] // 3
    vects = vects.reshape((N, 3))
    energy = 0.0
    for i in range(N):
        indices = (np.arange(N) > i)
        diffs = ((vects[indices] - vects[i]) ** 2).sum(1) ** alpha
        energy += (weight_matrix[i, indices] * (1.0 / (diffs + epsilon))).sum()
        if antipodal:
            sums = ((vects[indices] + vects[i]) ** 2).sum(1) ** alpha
            energy += (weight_matrix[i, indices] *
                       (1.0 / (sums + epsilon))).sum()
    return energy


def grad_f(vects, weight_matrix, antipodal=True, alpha=1.0):
    """Gradient of :func:`f` with respect to the vector coordinates.

    Parameters
    ----------
    vects : array-like, shape (N * 3,)
        Flattened vectors.
    weight_matrix : array-like, shape (N, N)
        Contribution weight of each pair of points.
    antipodal : bool
        Whether antipodal images are included, as in :func:`f`.
    alpha : float
        Power of the repulsion.

    Returns
    -------
    ndarray, shape (N * 3,)
        Gradient, flattened in the same order as ``vects``.
    """
    N = vects.shape[0] // 3
    vects = vects.reshape((N, 3))
    grad = np.zeros((N, 3))
    for i in range(N):
        indices = (np.arange(N) != i)
        diffs = ((vects[indices] - vects[i]) ** 2).sum(1) ** (alpha + 1)
        grad[i] = (- 2 * alpha * weight_matrix[i, indices] *
                   (vects[i] - vects[indices]).T / diffs).sum(1)
        if antipodal:
            sums = ((vects[indices] + vects[i]) ** 2).sum(1) ** (alpha + 1)
            grad[i] += (- 2 * alpha * weight_matrix[i, indices] *
                        (vects[i] + vects[indices]).T / sums).sum(1)
    grad = grad.reshape(N * 3)
    return grad


def _expand_weights(nb_points_per_shell, weights):
    """Build the per-point weight matrix from the shell-level weights."""
    K = np.sum(nb_points_per_shell)
    S = len(nb_points_per_shell)
    indices = np.cumsum(nb_points_per_shell).tolist()
    indices.insert(0, 0)
    weight_matrix = np.zeros((K, K))
    for s1 in range(S):
        for s2 in range(S):
            weight_matrix[indices[s1]:indices[s1 + 1],
                          indices[s2]:indices[s2 + 1]] = weights[s1, s2]
    return weight_matrix


def cost(vects, S, Ks, weights, antipodal=True):
    """Multiple-shell objective: the energy of all points with per-shell
    coupling weights.

    Parameters
    ----------
    vects : array-like, shape (N * 3,)
        Flattened vectors, shells stored consecutively.
    S : int
        Number of shells.
    Ks : list of int, length S
        Number of points on each shell.
    weights : array-like, shape (S, S)
        Coupling weight between each pair of shells.
    antipodal : bool
        Whether antipodal images are included.

    Returns
    -------
    float
    """
    weight_matrix = _expand_weights(Ks, weights)
    return f(vects, weight_matrix, antipodal)


def grad_cost(vects, S, Ks, weights, antipodal=True):
    """Gradient of :func:`cost`. Arguments as in :func:`cost`.

    Returns
    -------
    ndarray, shape (N * 3,)
    """
    weight_matrix = _expand_weights(Ks, weights)
    return grad_f(vects, weight_matrix, antipodal)


def optimize(nb_shells, nb_points_per_shell, weights, max_iter=100,
             antipodal=True, init_points=None):
    """Compute a set of directions on several shells by energy minimization.

    Parameters
    ----------
    nb_shells : int
        Number of shells. Only ``len(nb_points_per_shell)`` is used.
    nb_points_per_shell : list of int, length nb_shells
        Number of points on each shell.
    weights : array-like, shape (nb_shells, nb_shells)
        Coupling weights between shells.
    max_iter : int
        Maximum number of SLSQP iterations.
    antipodal : bool
        Whether antipodal images are included in the energy.
    init_points : array-like, shape (K, 3), optional
        Starting directions. Random uniform directions are used if omitted.
        Pass a seeded set for reproducible results.

    Returns
    -------
    ndarray, shape (K, 3)
        Unit vectors, with shells stored consecutively.
    """
    nb_shells = len(nb_points_per_shell)

    # Total number of points.
    K = np.sum(nb_points_per_shell)

    # Initialized with random directions, unless the caller provides some.
    if init_points is None:
        init_points = random_uniform_on_sphere(K)
    vects = np.asarray(init_points, dtype=float).reshape(K * 3)

    vects = scopt.fmin_slsqp(cost, vects.reshape(K * 3),
                             f_eqcons=equality_constraints, fprime=grad_cost,
                             iter=max_iter, acc=1.0e-9,
                             args=(nb_shells, nb_points_per_shell, weights,
                                   antipodal),
                             iprint=0)
    vects = vects.reshape((K, 3))
    vects = (vects.T / np.sqrt((vects ** 2).sum(1))).T
    return vects


def write(vects, nb_shells, nb_points_per_shell, filename):
    """Write a multiple-shell scheme to a text file.

    The file has a header line ``#shell-id x y z`` followed by one line per
    direction, with the shell index (starting at 0).

    Parameters
    ----------
    vects : array-like, shape (K, 3)
        Unit vectors, shells stored consecutively.
    nb_shells : int
        Number of shells.
    nb_points_per_shell : list of int, length nb_shells
        Number of points on each shell.
    filename : str
        Output path.
    """
    with open(filename, "w") as datafile:
        datafile.write("#shell-id\tx\ty\tz\n")
        k = 0
        for s in range(nb_shells):
            for n in range(nb_points_per_shell[s]):
                datafile.write("%d\t%f\t%f\t%f\n" %
                               (s, vects[k, 0], vects[k, 1], vects[k, 2]))
                k += 1


def random_uniform_on_sphere(K):
    """Return K pseudo-random unit vectors, uniform on the whole sphere.

    Draws from the global NumPy random state.

    Parameters
    ----------
    K : int
        Number of vectors.

    Returns
    -------
    ndarray, shape (K, 3)
    """
    phi = 2 * np.pi * np.random.rand(K)

    r = 2 * np.sqrt(np.random.rand(K))
    theta = 2 * np.arcsin(r / 2)

    vects = np.zeros((K, 3))
    vects[:, 0] = np.sin(theta) * np.cos(phi)
    vects[:, 1] = np.sin(theta) * np.sin(phi)
    vects[:, 2] = np.cos(theta)

    return vects


def compute_weights(nb_shells, nb_points_per_shell, shell_groups, alphas):
    """Build the shell coupling matrix from groups of shells to couple.

    Each group of shells is coupled with weight ``alpha`` and normalized by the
    total number of points in the group, so that groups of different sizes
    contribute comparably.

    Parameters
    ----------
    nb_shells : int
        Number of shells.
    nb_points_per_shell : list of int
        Number of points on each shell.
    shell_groups : list of list of int
        Each inner list gives the shell indices that are coupled together.
    alphas : list of float
        Coupling strength for each group, same length as ``shell_groups``.

    Returns
    -------
    ndarray, shape (nb_shells, nb_shells)
    """
    weights = np.zeros((nb_shells, nb_shells))
    for shell_group, alpha in zip(shell_groups, alphas):
        total_nb_points = 0
        for shell_id in shell_group:
            total_nb_points += nb_points_per_shell[shell_id]
        for i in shell_group:
            for j in shell_group:
                weights[i, j] += alpha / total_nb_points**2
    return weights
