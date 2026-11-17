"""Real spherical harmonics basis on the unit sphere.

Functions on the sphere are expanded in real spherical harmonics ``Y_lm``. The
flattened index ``j`` runs over the basis elements; :func:`index_j`,
:func:`index_l` and :func:`index_m` convert between ``j`` and ``(l, m)``.

Two conventions are supported, selected with ``symmetric``:

* ``symmetric=True`` (default): antipodally symmetric functions, only even
  degrees ``l`` are kept. Rank ``R`` (even) gives ``(R+1)(R+2)/2`` terms.
* ``symmetric=False``: all degrees ``0 <= l <= R``. Gives ``(R+1)^2`` terms.
"""
import numpy as np
from scipy.special import lpmv, factorial, legendre


_default_rank = 4


class SphericalHarmonics:
    """A real function on the sphere written as a spherical-harmonics series.

    Parameters
    ----------
    coefficients : array-like, shape (D,)
        Coefficients of the series. ``D`` must be the dimension of a basis of
        some rank (see :func:`dimension`); the rank is inferred from ``D``.
    symmetric : bool, optional
        If True (default), the function is antipodally symmetric and only even
        degrees are used.

    Attributes
    ----------
    rank : int
        Truncation rank of the basis, inferred from the number of coefficients.
    coefficients : ndarray, shape (D,)
        The coefficients, in the flattened order given by :func:`index_j`.

    Raises
    ------
    ValueError
        If the number of coefficients does not match any basis dimension.
    """

    def __init__(self, coefficients, symmetric=True):
        self.symmetric = symmetric
        self._create_from_coefficients(coefficients)

    def _create_from_coefficients(self, coefficients):
        """Infer the rank from the length of ``coefficients`` and store it."""
        rank = 0
        while True:
            dim_sh = dimension(rank, self.symmetric)
            if len(coefficients) == dim_sh:
                self.rank = rank
                self.coefficients = coefficients
                return
            elif len(coefficients) < dim_sh:
                raise ValueError("Invalid dimension for SH coefficients.")
            if self.symmetric:
                rank += 2
            else:
                rank += 1

    def get_rank(self):
        return self._rank

    def set_rank(self, value):
        if self.symmetric and value % 2 != 0:
            raise ValueError("'rank' only accepts even values.")
        self._rank = value

    rank = property(get_rank, set_rank, doc="Truncation rank of the basis.")

    def get_coefficients(self):
        return self._coefficients

    def set_coefficients(self, value):
        value = np.asarray(value, dtype=float)
        expected = dimension(self.rank, self.symmetric)
        if value.shape[0] != expected:
            raise ValueError("Coefficients shape and rank mismatch.")
        self._coefficients = value

    coefficients = property(get_coefficients, set_coefficients,
                            doc="Coefficients of the series, shape (D,).")

    def angular_function(self, theta, phi):
        """Evaluate the function at the given directions.

        Parameters
        ----------
        theta : array-like
            Polar angles, physics convention (0 at the north pole).
        phi : array-like
            Azimuthal angles, physics convention.

        Returns
        -------
        ndarray
            The function value at each ``(theta, phi)`` pair.
        """
        coefs = self.coefficients
        result = 0
        rank = self.rank
        if self.symmetric:
            step = 2
        else:
            step = 1
        for l in range(0, rank + 1, step):
            for m in range(-l, l + 1):
                j = index_j(l, m, self.symmetric)
                if coefs[j] != 0.0:
                    if m < 0:
                        result += coefs[j] * np.sqrt(2) \
                            * np.sqrt((2 * l + 1) * factorial(l + m)
                                      / (4 * np.pi * factorial(l - m))) \
                            * (-1) ** (-m) \
                            * lpmv(-m, l, np.cos(theta)) * np.cos(m * phi)
                    if m == 0:
                        result += coefs[j] \
                            * np.sqrt((2 * l + 1) * factorial(l - m)
                                      / (4 * np.pi * factorial(l + m))) \
                            * lpmv(m, l, np.cos(theta))
                    if m > 0:
                        result += coefs[j] * np.sqrt(2) \
                            * np.sqrt((2 * l + 1) * factorial(l - m)
                                      / (4 * np.pi * factorial(l + m))) \
                            * lpmv(m, l, np.cos(theta)) * np.sin(m * phi)
        return result


def dimension(rank, symmetric=True):
    """Return the number of basis functions for a given truncation rank.

    Parameters
    ----------
    rank : int
        Truncation rank.
    symmetric : bool
        Antipodally symmetric (even degrees only) if True.

    Returns
    -------
    int
        ``(rank+1)(rank+2)/2`` if symmetric, else ``(rank+1)^2``.
    """
    if symmetric:
        return int((rank + 1) * (rank + 2) / 2)
    return (rank + 1) * (rank + 1)


def index_j(l, m, symmetric=True):
    """Return the flattened index ``j`` of the harmonic of degree ``l`` and
    order ``m``.

    Raises
    ------
    ValueError
        If ``|m| > l``.
    """
    if np.abs(m) > l:
        raise ValueError("SphericalHarmonics.index_j: m must lie in [-l, l]")
    if symmetric:
        return int(l + m + (2 * np.array(range(0, l, 2)) + 1).sum())
    return l * (l + 1) + m


def index_l(j, symmetric=True):
    """Return the degree ``l`` of the harmonic with flattened index ``j``."""
    l = 0
    while dimension(l, symmetric) - 1 < j:
        if symmetric:
            l += 2
        else:
            l += 1
    return l


def index_m(j, symmetric=True):
    """Return the order ``m`` of the harmonic with flattened index ``j``."""
    l = index_l(j, symmetric)
    return j - dimension(l, symmetric) + l + 1


def matrix(theta, phi, rank=_default_rank, symmetric=True):
    """Return the observation matrix of the basis at a set of directions.

    Parameters
    ----------
    theta : array-like, shape (K,)
        Polar angles of the directions.
    phi : array-like, shape (K,)
        Azimuthal angles of the directions.
    rank : int
        Truncation rank of the basis.
    symmetric : bool
        Use the antipodally symmetric basis if True.

    Returns
    -------
    H : ndarray, shape (K, D)
        ``H[k, j]`` is basis function ``j`` evaluated at direction ``k``.
    """
    dim_sh = dimension(rank, symmetric)
    sh = SphericalHarmonics(np.zeros(dim_sh), symmetric)
    N = theta.shape[0]
    H = np.zeros((N, dim_sh))
    for j in range(dim_sh):
        sh.coefficients[:] = 0
        sh.coefficients[j] = 1.0
        H[:, j] = sh.angular_function(theta, phi)
    return H


def L(rank=_default_rank):
    """Return the Laplace-Beltrami regularization matrix (diagonal).

    The entry for harmonic ``(l, m)`` is ``-l(l+1)``. Only the symmetric
    basis is supported.

    Parameters
    ----------
    rank : int
        Truncation rank of the basis.
    """
    dim_sh = dimension(rank)
    L = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        l = index_l(j)
        L[j, j] = -(l * (l + 1))
    return L


def P(rank=_default_rank):
    """Return the Funk-Radon operator matrix (diagonal).

    The entry for harmonic of degree ``l`` is ``2 pi P_l(0)``, where ``P_l``
    is the Legendre polynomial. Only the symmetric basis is supported.

    Parameters
    ----------
    rank : int
        Truncation rank of the basis.
    """
    dim_sh = dimension(rank)
    P = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        l = index_l(j)
        P[j, j] = 2 * np.pi * legendre(l)(0)
    return P
