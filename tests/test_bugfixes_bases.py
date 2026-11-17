"""Regression tests for bugs in the basis modules.

Each test pins behaviour through a public function, with expected values taken
from a hand calculation rather than from the implementation.
"""
import numpy as np
import pytest
from scipy.special import binom

from qspace.bases import sh, spf, utils


def test_binomial_generalized_matches_hand_value():
    # binom(1/2, 2) = (1/2)(1/2 - 1) / 2! = -1/8
    assert utils.binomial(0.5, 2) == pytest.approx(-0.125)


def test_binomial_integer_matches_pascal():
    assert utils.binomial(5, 2) == pytest.approx(10.0)


def test_binomial_agrees_with_scipy():
    for alpha, k in [(0.5, 3), (3.5, 2), (-1.5, 4)]:
        assert utils.binomial(alpha, k) == pytest.approx(binom(alpha, k))


def test_sh_asymmetric_coefficients_are_accepted():
    # Full (non antipodally symmetric) basis of rank 3 has (3+1)^2 = 16 terms.
    harmonic = sh.SphericalHarmonics(np.zeros(16), symmetric=False)
    assert harmonic.rank == 3


def test_sh_laplace_matrix_diagonal():
    # Diagonal of -l(l+1) for rank 2 symmetric: l = 0, then l = 2 five times.
    diag = np.diag(sh.L(rank=2))
    np.testing.assert_allclose(diag, [0, -6, -6, -6, -6, -6])


def test_sh_funk_radon_diagonal():
    # 2 pi P_l(0): P_0(0) = 1, P_2(0) = -1/2.
    diag = np.diag(sh.P(rank=2))
    np.testing.assert_allclose(diag, [2 * np.pi] + [-np.pi] * 5)


def test_spf_laplace_matrix_uses_angular_degree():
    # Radial order 2, rank 2: the first radial block is l=0 then l=2 x5.
    diag = np.diag(spf.L(radial_order=2, angular_rank=2))
    np.testing.assert_allclose(diag[:6], [0, 36, 36, 36, 36, 36])


def test_spf_odf_tuch_uses_instance_zeta():
    coefs = np.zeros((2, sh.dimension(2)))
    coefs[0, 0] = 1.0
    coefs[1, 1] = 0.5
    short = spf.SphericalPolarFourier(2, 2, zeta=50.0)
    long = spf.SphericalPolarFourier(2, 2, zeta=700.0)
    short.coefficients = coefs.copy()
    long.coefficients = coefs.copy()
    odf_short = short.odf_tuch().coefficients
    odf_long = long.odf_tuch().coefficients
    assert np.all(np.isfinite(odf_short))
    assert not np.allclose(odf_short, odf_long)
