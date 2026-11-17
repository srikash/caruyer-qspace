"""Modified spherical polar Fourier (mSPF) basis.

The mSPF basis is a reparametrisation of the SPF basis that is used to
reconstruct a continuous q-space signal with Laplace regularization, following
Caruyer and Deriche (2012). Its radial functions have one fewer order than the
SPF basis: ``dimension(radial_order, angular_rank)`` uses
``radial_order - 1`` radial orders.

Reference
---------
Caruyer, E., and Deriche, R. (2012). Diffusion MRI signal reconstruction with
continuity constraint and optimal regularization. Medical Image Analysis
16(6), 1113-1120.
"""
import numpy as np
from scipy.special import genlaguerre, gamma, factorial

from . import sh
from . import spf


# Default parameter values, shared with the SPF basis.
_default_radial_order = spf._default_radial_order
_default_angular_rank = sh._default_rank
_default_zeta = spf._default_zeta


class ModifiedSphericalPolarFourier:
    """A function expressed in the mSPF basis.

    Parameters
    ----------
    radial_order : int
        Radial truncation order. The basis uses ``radial_order - 1`` radial
        functions.
    angular_rank : int
        Truncation rank of the angular part (even).
    zeta : float
        Scale parameter of the basis.

    Attributes
    ----------
    coefficients : ndarray, shape (radial_order, sh.dimension(angular_rank))
        Basis coefficients, indexed as ``coefficients[n, j]``. Only the first
        ``radial_order - 1`` rows are used.
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
        result = 0.0
        for n in range(self.radial_order - 1):
            if np.abs(self.coefficients[n]).max() > 0.0:
                sh_coefs = self.coefficients[n]
                spherical_harm = sh.SphericalHarmonics(sh_coefs)
                result += spherical_harm.angular_function(theta, phi) * \
                    radial_function(r, n, self.zeta)
        return result


def matrix(r, theta, phi, radial_order=_default_radial_order,
           angular_rank=_default_angular_rank, zeta=_default_zeta):
    """Return the mSPF observation matrix at a set of q-space points.

    Parameters
    ----------
    r, theta, phi : array-like, shape (K,)
        Spherical coordinates of the q-space points.
    radial_order : int
        Radial truncation order of the SPF basis this is paired with.
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
    H = np.zeros((K, radial_order - 1, sh.dimension(angular_rank)))
    b_n_j = ModifiedSphericalPolarFourier(radial_order, angular_rank, zeta)
    for n in range(H.shape[1]):
        for j in range(H.shape[2]):
            b_n_j.coefficients[:] = 0
            b_n_j.coefficients[n, j] = 1.0
            H[:, n, j] = b_n_j.spherical_function(r, theta, phi)
    return H.reshape(K, dimension(radial_order, angular_rank))


def to_spf_matrix(radial_order=_default_radial_order,
                  angular_rank=_default_angular_rank, zeta=_default_zeta):
    """Return the transition matrix from mSPF coefficients to SPF coefficients.

    Parameters
    ----------
    radial_order : int
        Radial truncation order.
    angular_rank : int
        Truncation rank of the angular part.
    zeta : float
        Scale parameter of the basis.

    Returns
    -------
    M : ndarray, shape (spf.dimension, dimension)
        ``M @ mspf_coefficients`` gives the equivalent SPF coefficients.
    """
    M = np.zeros((spf.dimension(radial_order, angular_rank),
                  dimension(radial_order, angular_rank)))
    for i in range(M.shape[0]):
        n_i = spf.index_n(i, radial_order, angular_rank)
        l_i = spf.index_l(i, radial_order, angular_rank)
        m_i = spf.index_m(i, radial_order, angular_rank)
        kappa_ni = spf.kappa(zeta, n_i)
        for j in range(M.shape[1]):
            n_j = index_n(j, radial_order, angular_rank)
            l_j = index_l(j, radial_order, angular_rank)
            m_j = index_m(j, radial_order, angular_rank)
            chi_nj = chi(zeta, n_j)
            if l_i == l_j and m_i == m_j:
                if n_i <= n_j:
                    M[i, j] = 3 * chi_nj / (2 * kappa_ni)
                else:
                    if n_i == n_j + 1:
                        M[i, j] = - (n_j + 1) * chi_nj / kappa_ni
    return M


def dimension(radial_order, angular_rank):
    """Return the dimension of the truncated mSPF basis."""
    return (radial_order - 1) * sh.dimension(angular_rank)


index_i = spf.index_i
index_n = spf.index_n
index_l = spf.index_l
index_m = spf.index_m


def chi(zeta, n):
    """Return the normalization constant of the mSPF radial function of order
    ``n``."""
    return np.sqrt(2 / zeta**1.5 * factorial(n) / gamma(n + 3.5))


def radial_function(r, n, zeta):
    """Evaluate the radial part of the mSPF basis of order ``n`` at radius
    ``r``."""
    return genlaguerre(n, 2.5)(r**2 / zeta) * \
        r**2 / zeta * np.exp(- r**2 / (2 * zeta)) * chi(zeta, n)


def Lambda(radial_order, angular_rank, zeta=_default_zeta):
    """Return the Laplace regularization matrix.

    The regularization term is ``(x - x0)^T Lambda (x - x0)``, where ``x0`` is
    given by :func:`v`.

    Parameters
    ----------
    radial_order : int
        Radial truncation order.
    angular_rank : int
        Truncation rank of the angular part.
    zeta : float
        Scale parameter of the basis.

    Returns
    -------
    ndarray, shape (dimension, dimension)
        Symmetric positive semi-definite matrix.
    """
    max_degree = 2 * (radial_order + 1)
    gammas = gamma(np.arange(max_degree) + 0.5)

    dim = dimension(radial_order, angular_rank)
    L = np.zeros((dim, dim))
    dim_sh = sh.dimension(angular_rank)
    for n1 in range(radial_order - 1):
        chi1 = chi(zeta, n1)
        for n2 in range(radial_order - 1):
            chi2 = chi(zeta, n2)
            for j1 in range(dim_sh):
                l1 = sh.index_l(j1)
                coeffs = _T_coeffs(n1, n2, l1)
                degree = coeffs.shape[0]
                matrix_entry = chi1 * chi2 / (2 * np.sqrt(zeta)) * \
                    np.dot(coeffs, gammas[range(degree - 1, -1, -1)])
                for j2 in range(dim_sh):
                    if j1 == j2:
                        L[n1 * dim_sh + j1, n2 * dim_sh + j2] = matrix_entry
    return L


def v(radial_order, angular_rank, zeta=_default_zeta):
    """Return the vector ``v`` of the Laplace regularization.

    The regularization minimum is at ``x0 = -Lambda^{-1} v``.

    Parameters
    ----------
    radial_order : int
        Radial truncation order.
    angular_rank : int
        Truncation rank of the angular part.
    zeta : float
        Scale parameter of the basis.

    Returns
    -------
    ndarray, shape (dimension,)
    """
    max_degree = 2 * (radial_order + 1)
    gammas = gamma(np.arange(max_degree) + 0.5)

    dim = dimension(radial_order, angular_rank)
    v = np.zeros(dim)
    dim_sh = sh.dimension(angular_rank)

    for n in range(radial_order - 1):
        chi1 = chi(zeta, n)
        coeffs = _T_coeffs(n, -1, 0)
        degree = coeffs.shape[0]
        v[n * dim_sh] = chi1 / (2 * np.sqrt(zeta)) \
            * np.dot(coeffs, gammas[range(degree - 1, -1, -1)])
    return v


def _F_n(n):
    """Return the polynomial part of ``F_n(q) = chi_n exp(-q^2/2 zeta) P_n(q)``
    with ``P_n(q) = q^2 / zeta * L_n^{5/2}(q^2 / zeta)``."""
    if n == -1:
        return np.poly1d([1.0])
    else:
        a = np.poly1d([1, 0.0, 0.0])
        return a * genlaguerre(n, 2.5)(a)


def _diff_F_n(p):
    """Return the polynomial part of the derivative of ``F_n``, i.e.
    ``-q/zeta P_n(q) + P_n'(q)``, in the same normalisation as :func:`_F_n`."""
    a = np.poly1d([-1, 0.0])
    return a * p + p.deriv()


def _h_i_poly(n, l):
    """Return the polynomial ``h_i`` of the Laplace matrix, with
    ``h_i(q) = chi_n exp(-q^2/2 zeta) * h_i_poly(q)``."""
    F0n = _F_n(n)
    F1n = _diff_F_n(F0n)
    F2n = _diff_F_n(F1n)
    a = np.poly1d([1.0, 0.0])
    b = (F0n / a)[0]  # Polynomial Euclidean division: quotient.
    return a * F2n + 2 * F1n - l * (l + 1) * b


def _T_coeffs(ni, nj, l):
    """Return the even-power coefficients of the integrand of the Laplace matrix.

    The ``(i, j)`` entry is ``chi_{n(i)} chi_{n(j)} int_0^inf
    exp(-q^2/zeta) T_{ij}(q^2/zeta) dq``; this returns the coefficients of
    ``T_{ij}``.
    """
    Tij = _h_i_poly(ni, l) * _h_i_poly(nj, l)
    degree = Tij.coeffs.shape[0]
    coeffs = Tij.coeffs[range(0, degree, 2)]
    return coeffs
