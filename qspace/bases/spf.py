"""Spherical polar Fourier (SPF) basis for q-space signal reconstruction.

The SPF basis is the product of a radial Laguerre-type function of order ``n``
and a real spherical harmonic of degree ``l`` and order ``m``. A function is
written as a truncated sum over radial order ``n < radial_order`` and harmonic
index ``j < dimension(angular_rank)``. The flattened index ``i`` is
``i = n * dim_sh + j``; see :func:`index_i`.

References
----------
Cheng, J., Ghosh, A., Jiang, T., Deriche, R. (2010), used for the Tuch and
marginal ODF formulas implemented here.
"""
import numpy as np
from scipy.special import genlaguerre, gamma, factorial, legendre

from . import sh
from . import utils


# Default parameter values.
_default_radial_order = 3
_default_angular_rank = sh._default_rank
_default_zeta = 700.0


class SphericalPolarFourier:
    """A function on q-space expressed in the truncated SPF basis.

    Parameters
    ----------
    radial_order : int
        Radial truncation order (number of radial functions, ``n = 0..order-1``).
    angular_rank : int
        Truncation rank of the angular (spherical harmonic) part. Must be even.
    zeta : float
        Scale parameter of the basis (controls the radial extent).

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

    def spherical_function(self, r, theta, phi):
        """Evaluate the function at points given in spherical coordinates.

        Parameters
        ----------
        r : array-like, shape (K,)
            Radii of the q-space points.
        theta : array-like, shape (K,)
            Polar angles of the q-space points.
        phi : array-like, shape (K,)
            Azimuthal angles of the q-space points.

        Returns
        -------
        f : ndarray, shape (K,)
            The function value at each point.
        """
        result = 0.0
        for n in range(self.radial_order):
            if abs(self.coefficients[n]).max() > 0.0:
                sh_coefs = self.coefficients[n]
                spherical_harm = sh.SphericalHarmonics(sh_coefs)
                result += \
                    spherical_harm.angular_function(theta, phi) * \
                    radial_function(r, n, self.zeta)
        return result

    def get_angular_rank(self):
        return self._angular_rank

    def set_angular_rank(self, value):
        if value % 2 != 0:
            raise ValueError("'angular_rank' only accepts even values.")
        self._angular_rank = value

    angular_rank = property(get_angular_rank, set_angular_rank,
                            doc="Truncation rank of the angular part (even).")

    def odf_tuch(self):
        """Compute the Tuch-style ODF from the SPF coefficients.

        Follows Cheng, Ghosh, Jiang and Deriche (2010).

        Returns
        -------
        sh.SphericalHarmonics
            The ODF as a spherical-harmonics series of rank ``angular_rank``.
        """
        dim_sh = sh.dimension(self.angular_rank)
        sh_coefs = np.zeros(dim_sh)
        for j in range(dim_sh):
            l = sh.index_l(j)
            for n in range(self.radial_order):
                partial_sum = 0.0
                for i in range(n):
                    partial_sum += utils.binomial(i - 0.5, i) * (-1)**(n - i)
                sh_coefs[j] += partial_sum * self.coefficients[n, j] \
                    * kappa(self.zeta, n)
            sh_coefs[j] = sh_coefs[j] * legendre(l)(0)
        return sh.SphericalHarmonics(sh_coefs)

    def odf_marginal(self):
        """Compute the marginal ODF from the SPF coefficients.

        Follows Cheng, Ghosh, Jiang and Deriche (2010).

        Returns
        -------
        sh.SphericalHarmonics
            The ODF as a spherical-harmonics series of rank ``angular_rank``.
        """
        dim_sh = sh.dimension(self.angular_rank)

        sh_coefs = np.zeros(dim_sh)
        sh_coefs[0] = 1 / np.sqrt(4 * np.pi)

        for l in range(2, self.angular_rank + 1, 2):
            for m in range(-l, l + 1):
                j = sh.index_j(l, m)
                for n in range(1, self.radial_order):
                    partial_sum = 0.0
                    for i in range(1, n + 1):
                        partial_sum += (-1)**i * \
                            utils.binomial(n + 0.5, n - i) * 2**i / i
                    sh_coefs[j] += partial_sum * kappa(self.zeta, n) * \
                        self.coefficients[n, j] * \
                        legendre(l)(0) * l * (l + 1) / (8 * np.pi)
        return sh.SphericalHarmonics(sh_coefs)


def matrix(r, theta, phi, radial_order=_default_radial_order,
           angular_rank=_default_angular_rank, zeta=_default_zeta):
    """Return the SPF observation matrix at a set of q-space points.

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
    b_n_j = SphericalPolarFourier(radial_order, angular_rank, zeta)
    for n in range(H.shape[1]):
        for j in range(H.shape[2]):
            b_n_j.coefficients[:] = 0
            b_n_j.coefficients[n, j] = 1.0
            H[:, n, j] = b_n_j.spherical_function(r, theta, phi)
    return H.reshape(K, dimension(radial_order, angular_rank))


def dimension(radial_order, angular_rank):
    """Return the dimension of the truncated SPF basis."""
    return radial_order * sh.dimension(angular_rank)


def index_i(n, l, m, radial_order, angular_rank):
    """Return the flattened SPF index ``i`` for radial order ``n`` and harmonic
    ``(l, m)``."""
    dim_sh = sh.dimension(angular_rank)
    j = sh.index_j(l, m)
    return n * dim_sh + j


def index_n(i, radial_order, angular_rank):
    """Return the radial order ``n`` of the flattened index ``i``."""
    dim_sh = sh.dimension(angular_rank)
    return i // dim_sh


def index_l(i, radial_order, angular_rank):
    """Return the angular degree ``l`` of the flattened index ``i``."""
    dim_sh = sh.dimension(angular_rank)
    j = i % dim_sh
    return sh.index_l(j)


def index_m(i, radial_order, angular_rank):
    """Return the angular order ``m`` of the flattened index ``i``."""
    dim_sh = sh.dimension(angular_rank)
    j = i % dim_sh
    return sh.index_m(j)


def L(radial_order, angular_rank):
    """Return the angular regularization matrix of Assemlal et al.

    Diagonal with entry ``(l(l+1))^2`` for each basis function.
    """
    dim_sh = sh.dimension(angular_rank)
    diag_L = np.zeros((radial_order, dim_sh))
    for j in range(dim_sh):
        l = sh.index_l(j)
        diag_L[:, j] = (l * (l + 1)) ** 2
    dim_spf = dimension(radial_order, angular_rank)
    return np.diag(diag_L.reshape(dim_spf))


def N(radial_order, angular_rank):
    """Return the radial regularization matrix of Assemlal et al.

    Diagonal with entry ``(n(n+1))^2`` for each radial order ``n``.
    """
    dim_sh = sh.dimension(angular_rank)
    diag_N = np.zeros((radial_order, dim_sh))
    for n in range(radial_order):
        diag_N[n, :] = (n * (n + 1)) ** 2
    dim_spf = dimension(radial_order, angular_rank)
    return np.diag(diag_N.reshape(dim_spf))


def kappa(zeta, n):
    """Return the normalization constant of the SPF radial function of order
    ``n``."""
    return np.sqrt(2 / zeta**1.5 * factorial(n) / gamma(n + 1.5))


def radial_function(r, n, zeta):
    """Evaluate the radial part of the SPF basis of order ``n`` at radius
    ``r``."""
    return genlaguerre(n, 0.5)(r**2 / zeta) * \
        np.exp(- r**2 / (2 * zeta)) * \
        kappa(zeta, n)
