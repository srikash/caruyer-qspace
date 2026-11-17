"""Dual spherical polar Fourier (dSPF) basis.

The dual SPF basis is the Fourier-domain counterpart of the SPF basis of
:mod:`qspace.bases.spf`, following Cheng et al. It is used to write the
Fourier transform of a function that is expressed in the SPF basis.

.. warning::
   The port of this module to Python 3 fixed undefined names and a missing
   import, and smoke-tests (finite output, correct shapes) pass. The
   mathematical formulas have not been independently validated against a
   reference implementation. Treat results as provisional.
"""
import numpy as np
from scipy.special import gamma, factorial, hyp1f1

from . import sh
from . import spf
from . import utils


# Default parameter values, shared with the SPF basis.
_default_radial_order = spf._default_radial_order
_default_angular_rank = sh._default_rank
_default_zeta = spf._default_zeta


class DualSphericalPolarFourier:
    """A function expressed in the dual SPF basis.

    Parameters
    ----------
    radial_order : int
        Radial truncation order of the dSPF basis.
    angular_rank : int
        Truncation rank of the angular part (even).
    zeta : float
        Scale parameter of the basis.

    Attributes
    ----------
    coefficients : ndarray, shape (radial_order, sh.dimension(angular_rank))
        Basis coefficients, indexed as ``coefficients[n, j]``.
    """

    def __init__(self, radial_order=_default_radial_order,
                 angular_rank=_default_angular_rank, zeta=_default_zeta):
        self.radial_order = radial_order
        self.angular_rank = angular_rank
        self.zeta = zeta
        self.coefficients = np.zeros((self.radial_order,
                                      sh.dimension(self.angular_rank)))

    def get_angular_rank(self):
        return self._angular_rank

    def set_angular_rank(self, value):
        if value % 2 != 0:
            raise ValueError("'angular_rank' only accepts even values.")
        self._angular_rank = value

    angular_rank = property(get_angular_rank, set_angular_rank,
                            doc="Truncation rank of the angular part (even).")

    def spherical_function(self, r, theta, phi):
        """Evaluate the function at points given in spherical coordinates.

        Parameters
        ----------
        r, theta, phi : array-like, shape (K,)
            Spherical coordinates of the q-space points.

        Returns
        -------
        f : ndarray, shape (K,)
            The function value at each point.
        """
        Y = sh.matrix(theta, phi, rank=self.angular_rank)
        F = np.zeros(Y.shape)
        for n in range(self.radial_order):
            if (np.abs(self.coefficients[n])).max() > 0.0:
                for l in range(0, self.angular_rank + 1, 2):
                    f_n_l = radial_function(r, n, l, self.zeta)
                    for m in range(-l, l + 1):
                        j = sh.index_j(l, m)
                        F[:, j] += self.coefficients[n, j] * f_n_l
        return (F * Y).sum(1)


def matrix(r, theta, phi, radial_order=_default_radial_order,
           angular_rank=_default_angular_rank, zeta=_default_zeta):
    """Return the dSPF observation matrix at a set of q-space points.

    Parameters
    ----------
    r, theta, phi : array-like, shape (K,)
        Spherical coordinates of the q-space points.
    radial_order : int
        Radial truncation order.
    angular_rank : int
        Truncation rank of the angular part.
    zeta : float
        Scale parameter of the basis.

    Returns
    -------
    H : ndarray, shape (K, dimension(radial_order, angular_rank))
        ``H[k, i]`` is basis function ``i`` evaluated at point ``k``.
    """
    K = r.shape[0]
    H = np.zeros((K, radial_order, sh.dimension(angular_rank)))
    b_n_j = DualSphericalPolarFourier(radial_order=radial_order,
                                      angular_rank=angular_rank, zeta=zeta)
    for n in range(H.shape[1]):
        for j in range(H.shape[2]):
            b_n_j.coefficients[:] = 0
            b_n_j.coefficients[n, j] = 1.0
            H[:, n, j] = b_n_j.spherical_function(r, theta, phi)
    return H.reshape(K, dimension(radial_order, angular_rank))


def dimension(radial_order, angular_rank):
    """Return the dimension of the truncated dSPF basis."""
    return radial_order * sh.dimension(angular_rank)


index_i = spf.index_i
index_n = spf.index_n
index_l = spf.index_l
index_m = spf.index_m


def radial_function(r, n, l, zeta):
    """Evaluate the radial part of the dSPF basis of order ``n`` and degree
    ``l`` at radius ``r``, using a confluent hypergeometric sum."""
    result = 0.0
    for i in range(n + 1):
        result += \
            (- 1)**i * utils.binomial(n + 0.5, n - i) / \
            factorial(i) * 2**(0.5 * l + i - 0.5) * \
            gamma(0.5 * l + i + 1.5) * \
            hyp1f1((2 * i + l + 3) * 0.5, l + 1.5, - 2 * np.pi**2 * r**2 * zeta)
    result = result * 4 * (-1)**(0.5 * l) * zeta**(0.5 * l + 1.5) * \
        np.pi**(l + 1.5) * r**l / gamma(l + 1.5) * spf.kappa(zeta, n)
    return result
