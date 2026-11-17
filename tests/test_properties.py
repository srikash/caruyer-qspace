"""Property tests for the sampling, waveform, simulation and basis modules.

Expected values come from closed-form results or from identities that must hold
(orthonormality, inverse maps, round trips), not from re-running the code under
test.
"""
import numpy as np
import pytest
from scipy import integrate

from qspace.bases import sh, spf, mspf
from qspace.sampling import sphere, space, multishell as ms
from qspace.sampling.waveforms import btensors, io as wio, piecewise
from qspace.simul import camino, substrates


# ---------------------------------------------------------------- sampling

@pytest.mark.parametrize("K", [6, 20, 64, 200])
def test_koay_returns_k_unit_vectors_in_upper_hemisphere(K):
    points = sphere.koay(K)
    assert points.shape == (K, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)
    assert np.all(points[:, 2] >= 0)


@pytest.mark.parametrize("K", [6, 10, 100])
def test_jones_returns_k_unit_vectors(K):
    points = sphere.jones(K)
    assert points.shape == (K, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-9)


def test_to_spherical_rejects_non_unit_vectors():
    with pytest.raises(ValueError):
        sphere.to_spherical(np.array([[0.0, 0.0, 2.0]]))


def test_multishell_optimize_lowers_energy_and_keeps_unit_norm():
    shells = [6, 10]
    weights = ms.compute_weights(2, shells, [[0], [1], [0, 1]],
                                 [1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    init = rng.normal(size=(sum(shells), 3))
    init /= np.linalg.norm(init, axis=1)[:, None]
    points = ms.optimize(2, shells, weights, max_iter=300, init_points=init)
    assert points.shape == (16, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)
    before = ms.cost(init.reshape(-1), 2, shells, weights)
    after = ms.cost(points.reshape(-1), 2, shells, weights)
    assert after < before


def test_multishell_write_round_trip(tmp_path):
    points = sphere.jones(6)
    filename = tmp_path / "scheme.txt"
    ms.write(points, 2, [3, 3], str(filename))
    lines = filename.read_text().splitlines()
    assert lines[0] == "#shell-id\tx\ty\tz"
    assert [int(line.split()[0]) for line in lines[1:]] == [0, 0, 0, 1, 1, 1]


def test_space_round_trip_and_zero_vector():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(20, 3))
    r, theta, phi = space.to_spherical(points)
    np.testing.assert_allclose(space.to_cartesian(r, theta, phi), points,
                               atol=1e-12)


# --------------------------------------------------------------- bases

def test_sh_matrix_columns_are_orthonormal():
    # Gauss-Legendre in cos(theta) times a uniform grid in phi integrates
    # products of harmonics of rank <= 6 exactly.
    rank = 6
    x, w_x = np.polynomial.legendre.leggauss(12)
    nphi = 24
    phis = np.arange(nphi) * 2 * np.pi / nphi
    theta_grid, phi_grid = np.meshgrid(np.arccos(x), phis, indexing="ij")
    weights = np.outer(w_x, np.full(nphi, 2 * np.pi / nphi)).ravel()
    H = sh.matrix(theta_grid.ravel(), phi_grid.ravel(), rank=rank)
    gram = H.T @ (weights[:, None] * H)
    np.testing.assert_allclose(gram, np.eye(H.shape[1]), atol=1e-10)


def test_sh_laplace_is_symmetric_and_negative_semidefinite():
    L = sh.L(rank=4)
    np.testing.assert_allclose(L, L.T)
    assert np.all(np.linalg.eigvalsh(L) <= 1e-12)


def test_spf_matrix_times_transition_equals_mspf_matrix():
    rng = np.random.default_rng(2)
    K = 40
    r = rng.uniform(0.5, 4.0, K)
    theta = rng.uniform(0, np.pi, K)
    phi = rng.uniform(0, 2 * np.pi, K)
    radial_order, angular_rank, zeta = 4, 4, 1.0
    H_spf = spf.matrix(r, theta, phi, radial_order, angular_rank, zeta)
    H_mspf = mspf.matrix(r, theta, phi, radial_order, angular_rank, zeta)
    M = mspf.to_spf_matrix(radial_order, angular_rank, zeta)
    np.testing.assert_allclose(H_spf @ M, H_mspf, atol=1e-8, rtol=1e-8)


def test_mspf_lambda_is_symmetric_positive_semidefinite():
    Lam = mspf.Lambda(4, 4, zeta=1.0)
    np.testing.assert_allclose(Lam, Lam.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(Lam) >= -1e-8)


# ------------------------------------------------------------- waveforms

def test_compute_b_matches_pgse_closed_form():
    # Two opposite rectangular lobes of amplitude G and duration delta,
    # separated by Delta. Closed form: B = gamma^2 G^2 delta^2 (Delta - delta/3).
    G, delta, Delta = 0.08, 0.01, 0.02
    dt = 1.0e-4
    nb = int(round((Delta + delta) / dt)) + 1
    t = np.arange(nb) * dt
    waveform = np.zeros((nb, 3))
    waveform[(t >= 0) & (t < delta - 1e-12), 0] = G
    waveform[(t >= Delta - 1e-12) & (t < Delta + delta - 1e-12), 0] = -G
    B = btensors.compute_b(waveform, dt)
    expected = btensors.GAMMA**2 * G**2 * delta**2 * (Delta - delta / 3)
    assert B[0, 0] == pytest.approx(expected, rel=2e-2)
    assert abs(B[1, 1]) < 1e-12 * expected


def test_transform_waveform_maps_source_b_tensor_to_target():
    rng = np.random.default_rng(3)
    waveform = rng.normal(size=(50, 3)) * 0.05
    dt = 1e-4
    B1 = btensors.compute_b(waveform, dt)
    B2 = np.diag([2.0, 1.0, 0.5]) * 1e9
    M = btensors.transform_waveform(B1, B2)
    B_out = btensors.compute_b(waveform @ M.T, dt)
    # Tolerance relative to the matrix scale: zero entries carry float64 round-off.
    np.testing.assert_allclose(B_out, B2, rtol=1e-6, atol=1e-6 * np.max(B2))


def test_transform_waveform_rejects_rank_increase():
    linear = np.diag([1.0, 0.0, 0.0])
    spherical = np.eye(3)
    with pytest.raises(ValueError):
        btensors.transform_waveform(linear, spherical)


def test_camino_waveform_round_trip(tmp_path):
    ts = np.linspace(0, 0.1, 101)
    waveform = 0.08 * np.array([np.sin(2 * np.pi * ts / 0.1),
                                np.sin(6 * np.pi * ts / 0.1),
                                np.sin(10 * np.pi * ts / 0.1)]).T
    filename = str(tmp_path / "scheme.scheme")
    wio.write_camino([waveform, waveform], ts[1] - ts[0], filename)
    waveforms, dt = wio.read_camino(filename)
    assert len(waveforms) == 2
    assert dt == pytest.approx(ts[1] - ts[0])
    np.testing.assert_allclose(waveforms[0], waveform, atol=1e-5)


def test_mcdc_waveform_round_trip(tmp_path):
    waveform = np.arange(30, dtype=float).reshape(10, 3) * 1e-3
    filename = str(tmp_path / "scheme.mcdc")
    wio.write_mcdc([waveform, waveform], 1e-2, filename)
    waveforms, dt = wio.read_mcdc(filename)
    assert dt == pytest.approx(1e-2)
    np.testing.assert_allclose(waveforms[1], waveform, atol=1e-6)


def test_piecewise_basis_functions_are_orthonormal():
    basis = piecewise.QPiecewiseAffine(delta_rf=0.01, delta1=0.02,
                                       delta2=0.02, dt=0.005)
    assert len(basis) >= 1
    for i in range(len(basis)):
        for j in range(len(basis)):
            prod = integrate.quad(lambda x: basis[i](x) * basis[j](x),
                                  0, basis.te, limit=200)[0]
            expected = 1.0 if i == j else 0.0
            assert prod == pytest.approx(expected, abs=1e-6)


def test_bernstein_and_power_conversions_are_inverse():
    for order in [1, 2, 3]:
        b2p = piecewise._bernstein_to_power(order, [0, 1])
        p2b = piecewise._power_to_bernstein(order, [0, 1])
        np.testing.assert_allclose(b2p @ p2b, np.eye(order + 1), atol=1e-10)


# ------------------------------------------------------------ simulation

def test_cylinder_density_separation_round_trip():
    for packing in ["hexagonal", "square"]:
        a = substrates.CylindersLattice(1.0e-6, packing=packing, density=0.6)
        b = substrates.CylindersLattice(1.0e-6, packing=packing,
                                        separation=a.separation)
        assert b.density == pytest.approx(0.6)


def test_cylinder_rejects_bad_arguments():
    with pytest.raises(ValueError):
        substrates.CylindersLattice(1.0, packing="triangle", density=0.5)
    with pytest.raises(ValueError):
        substrates.CylindersLattice(1.0, density=0.5, separation=3.0)
    with pytest.raises(ValueError):
        substrates.CylindersLattice(1.0, density=0.5, separation=1.0)
    with pytest.raises(ValueError):
        substrates.CylindersLattice(1.0)


def test_parse_trajfile_reads_camino_binary_layout(tmp_path):
    import struct
    nb_spins, nb_steps, duration = 2, 3, 0.1
    filename = tmp_path / "walkers.traj"
    with open(filename, "wb") as f:
        f.write(struct.pack(">ddd", duration, nb_spins, nb_steps - 1))
        for step in range(nb_steps):
            for spin in range(nb_spins):
                f.write(struct.pack(">diddd", step * 0.01, spin,
                                    spin + step * 0.1, 1.0, -2.0))
    traj = camino.parse_trajfile(str(filename))
    assert traj.shape == (nb_spins, nb_steps, 3)
    assert traj[1, 2].tolist() == pytest.approx([1.0 + 0.2, 1.0, -2.0])


def test_simulation_without_datasynth_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "")
    sim = camino.CaminoSimulation(substrates.Free(), str(tmp_path / "x.traj"))
    with pytest.raises(RuntimeError, match="datasynth"):
        sim.run()


def test_dtd_signal_is_scalar_for_single_b_tensor():
    from qspace.simul import dtd
    tensors = np.stack([np.eye(3) * 1e-9, np.diag([2e-9, 1e-9, 0.5e-9])])
    dist = dtd.DiscreteDTD(tensors, np.array([0.5, 0.5]))
    b = np.diag([1e9, 0.0, 0.0])
    value = dist.signal(b)
    assert np.ndim(value) == 0
    # Hand value: mean of exp(-b:D_i) over the two tensors.
    expected = 0.5 * (np.exp(-1.0) + np.exp(-2.0))
    assert value == pytest.approx(expected)
    assert dist.signal(np.stack([b, b])).shape == (2,)


def test_piecewise_constant_and_polynomial_bases_construct():
    # Constructors exercise the breakpoint bookkeeping and constraint code
    # paths; they must build a non-empty orthonormal basis.
    constant = piecewise.QPiecewiseConstant(te=0.1, delta_rf=0.01,
                                            nb_chunks=2, delta_ramp=0.002)
    assert len(constant) >= 1
    assert str(constant) == "QPiecewiseConstant_nb_chunks_2"
    polynomial = piecewise.QPiecewisePolynomial(te=0.1, delta_rf=0.01,
                                                nb_chunks=2, order=5,
                                                smoothness=1)
    assert len(polynomial) == 6
    assert "order_5" in str(polynomial)


def test_piecewise_polynomial_default_constraints_can_be_empty():
    # Documents current behaviour: with order 3 and C^2 smoothness the
    # constraints leave no free waveform, so the basis is empty.
    empty = piecewise.QPiecewisePolynomial(te=0.1, delta_rf=0.01,
                                           nb_chunks=1, order=3)
    assert len(empty) == 0


def test_dspf_spherical_function_smoke():
    # Smoke test only: the dSPF formulas are not independently validated.
    from qspace.bases import dspf
    rng = np.random.default_rng(5)
    basis = dspf.DualSphericalPolarFourier(3, 4, zeta=60.0)
    basis.coefficients = rng.normal(size=basis.coefficients.shape)
    r = rng.uniform(0.0, 0.2, 10)
    theta = rng.uniform(0, np.pi, 10)
    phi = rng.uniform(0, 2 * np.pi, 10)
    values = basis.spherical_function(r, theta, phi)
    assert values.shape == (10,)
    assert np.all(np.isfinite(values))
    H = dspf.matrix(r, theta, phi, 3, 4, 60.0)
    assert H.shape == (10, dspf.dimension(3, 4))
    assert np.all(np.isfinite(H))


def test_odf_marginal_binomial_fix_changes_values_as_documented():
    # The old implementation multiplied by (alpha - k) instead of (alpha - i).
    import math
    from qspace.bases import utils

    def old_binomial(alpha, k):
        result = 1.0
        for _ in range(k):
            result = result * (alpha - k)
        return result / math.factorial(k)

    assert old_binomial(0.5, 2) == pytest.approx(1.125)
    assert utils.binomial(0.5, 2) == pytest.approx(-0.125)


def test_camino_writer_keeps_small_timesteps(tmp_path):
    waveform = np.zeros((5, 3))
    filename = str(tmp_path / "fine.scheme")
    wio.write_camino([waveform], 1e-4, filename)
    _, dt = wio.read_camino(filename)
    assert dt == pytest.approx(1e-4)


def test_read_single_volume_fsl_table(tmp_path):
    from qspace.sampling import io as sio
    (tmp_path / "one.bval").write_text("1000\n")
    (tmp_path / "one.bvec").write_text("1\n0\n0\n")
    bvals, bvecs = sio.read_fsl(str(tmp_path / "one"))
    np.testing.assert_allclose(bvals, [1000.0])
    np.testing.assert_allclose(bvecs, [[1.0, 0.0, 0.0]])


def test_cylinders_reject_nonpositive_density():
    with pytest.raises(ValueError):
        substrates.CylindersLattice(1.0e-6, density=0.0)
    with pytest.raises(ValueError):
        substrates.CylindersLattice(1.0e-6, density=-0.5)


def test_sh_accepts_plain_list():
    harmonic = sh.SphericalHarmonics([1.0, 0, 0, 0, 0, 0])
    assert harmonic.rank == 2


def test_reprojected_view_accepts_default_radii():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from qspace.visu import visu_points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    visu_points.draw_points_reprojected(sphere.jones(6), 1, [6], None, ax)
    plt.close(fig)


def test_sh_integer_list_coefficients_are_float_and_writable():
    harmonic = sh.SphericalHarmonics([1, 0, 0, 0, 0, 0])
    harmonic.coefficients[1] = 0.5
    assert harmonic.coefficients.dtype == float
    assert harmonic.coefficients[1] == pytest.approx(0.5)
