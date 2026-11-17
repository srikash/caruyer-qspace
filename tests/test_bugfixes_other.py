"""Regression tests for bugs outside the basis modules: spherical conversions,
gradient-table I/O, substrates, and the create_dvs helpers."""
import numpy as np
import pytest

from qspace.sampling import space, io
from qspace.simul import substrates
from qspace.cli import create_dvs


def test_space_to_spherical_round_trip():
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    r, theta, phi = space.to_spherical(points)
    np.testing.assert_allclose(r, [2.0, 1.0, 3.0])
    np.testing.assert_allclose(theta, [0.0, np.pi / 2, np.pi / 2])
    np.testing.assert_allclose(phi, [0.0, 0.0, np.pi / 2])


def test_space_to_spherical_handles_zero_vector():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r, theta, phi = space.to_spherical(points)
    np.testing.assert_allclose(r, [0.0, 1.0])
    assert np.all(np.isfinite(theta)) and np.all(np.isfinite(phi))


def test_substrate_factory_builds_cylinders():
    sub = substrates.substrate(name="CylindersLattice", radius=1.0e-6,
                               density=0.5)
    assert isinstance(sub, substrates.CylindersLattice)
    assert sub.density == pytest.approx(0.5)


def test_substrate_factory_builds_free():
    assert isinstance(substrates.substrate(name="Free"), substrates.Free)


def test_square_packing_separation_uses_square_formula():
    # Square packing: density = pi r^2 / s^2. With r = 1 and s = 2 -> pi/4.
    sub = substrates.CylindersLattice(1.0, packing="square", separation=2.0)
    assert sub.density == pytest.approx(np.pi / 4)


def test_packing_name_is_case_insensitive():
    sub = substrates.CylindersLattice(1.0e-6, packing="Hexagonal",
                                      density=0.5)
    assert sub.camino_options["-packing"] == "hex"


def test_write_fsl_and_read_back(tmp_path):
    bvals = np.array([0.0, 1000.0, 2000.0])
    bvecs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    base = tmp_path / "scheme"
    io.write(bvals, bvecs, str(base), format="fsl")
    got_bvals, got_bvecs = io.read_fsl(str(base))
    np.testing.assert_allclose(got_bvals, bvals)
    np.testing.assert_allclose(got_bvecs, bvecs)


def test_write_mrtrix_has_four_columns(tmp_path):
    bvals = np.array([1000.0, 2000.0])
    bvecs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    filename = tmp_path / "scheme.b"
    io.write(bvals, bvecs, str(filename), format="mrtrix")
    table = np.loadtxt(filename)
    np.testing.assert_allclose(table, [[1, 0, 0, 1000], [0, 0, 1, 2000]])


def test_write_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        io.write(np.ones(2), np.eye(3)[:2], str(tmp_path / "x"),
                 format="bogus")


def test_prepend_b0_uses_input_argument():
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = create_dvs.prepend_b0(dirs, 1)
    np.testing.assert_allclose(out, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_interleave_b0_places_b0_at_batch_starts():
    # 4 directions + 2 b=0: batch size 3, so b=0 goes to rows 0 and 3.
    dirs = np.arange(12, dtype=float).reshape(4, 3) + 1
    out = create_dvs.interleave_b0(dirs, 2)
    np.testing.assert_allclose(out[0], 0)
    np.testing.assert_allclose(out[3], 0)
    np.testing.assert_allclose(out[[1, 2, 4, 5]], dirs)
