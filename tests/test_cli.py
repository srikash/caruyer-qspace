"""End-to-end tests of the command-line tools, run through their main()
functions on small inputs written to a temporary directory."""
import numpy as np
import pytest

from qspace.cli import create_dvs, multishell, visu_btable


def test_create_dvs_scales_shells_and_prepends_b0(tmp_path):
    # Two shells, one direction each; unit vectors along x and y.
    shells = tmp_path / "shells.txt"
    shells.write_text("0 1 0 0\n1 0 1 0\n")
    out = tmp_path / "out.dvs"
    status = create_dvs.main([str(shells), "1", "1000", "2000",
                              "-o", str(out)])
    assert status == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "[directions=3]"
    vectors = [line.split("(")[1].split(")")[0] for line in lines
               if line.startswith("Vector")]
    values = [[float(v) for v in s.split(",")] for s in vectors]
    # b0 first, then x scaled by sqrt(1000/2000) and y scaled by 1.
    np.testing.assert_allclose(values[0], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(values[1], [np.sqrt(0.5), 0, 0], atol=1e-6)
    np.testing.assert_allclose(values[2], [0, 1, 0], atol=1e-6)


def test_create_dvs_rejects_wrong_number_of_bvalues(tmp_path):
    shells = tmp_path / "shells.txt"
    shells.write_text("0 1 0 0\n1 0 1 0\n")
    with pytest.raises(SystemExit):
        create_dvs.main([str(shells), "0", "1000", "-o",
                         str(tmp_path / "x.dvs")])


def test_multishell_cli_writes_scheme_and_figures(tmp_path):
    base = str(tmp_path / "scheme")
    status = multishell.main(["4", "6", "--seed", "0", "--max-iter", "50",
                              "-o", base])
    assert status == 0
    lines = (tmp_path / "scheme.txt").read_text().splitlines()
    assert len(lines) == 1 + 10
    assert (tmp_path / "scheme-shells.png").stat().st_size > 0
    assert (tmp_path / "scheme-shells-reprojected.png").stat().st_size > 0


def test_multishell_cli_is_reproducible_with_seed(tmp_path):
    multishell.main(["4", "4", "--seed", "7", "--max-iter", "30",
                     "-o", str(tmp_path / "a")])
    multishell.main(["4", "4", "--seed", "7", "--max-iter", "30",
                     "-o", str(tmp_path / "b")])
    assert (tmp_path / "a.txt").read_text() == (tmp_path / "b.txt").read_text()


def test_visu_btable_saves_figure(tmp_path):
    bvals = tmp_path / "scheme.bval"
    bvecs = tmp_path / "scheme.bvec"
    bvals.write_text("0 1000 1000 2000 2000 2000\n")
    bvecs.write_text("0 1 0 0 0 1\n0 0 1 0.6 0 0\n0 0 0 0.8 1 0\n")
    out = tmp_path / "btable.png"
    assert visu_btable.main(["-b", str(bvals), "-r", str(bvecs),
                             "-o", str(out)]) == 0
    assert out.stat().st_size > 0


def test_mspf_fit_on_synthetic_volume(tmp_path):
    nib = pytest.importorskip("nibabel")
    from qspace.cli import mspf_fit

    rng = np.random.default_rng(4)
    n_b0, n_dir = 2, 16
    dirs = rng.normal(size=(n_dir, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    bvals = np.concatenate([np.zeros(n_b0), np.full(n_dir, 1000.0)])
    bvecs = np.vstack([np.tile([1.0, 0, 0], (n_b0, 1)), dirs])
    bval_path = tmp_path / "bvals.txt"
    bvec_path = tmp_path / "bvecs.txt"
    np.savetxt(bval_path, bvals[None, :], fmt="%g")
    np.savetxt(bvec_path, bvecs.T, fmt="%.8f")

    data = np.exp(-0.5 * np.ones((2, 2, 1, bvals.size)))
    data[..., :n_b0] = 1.0
    nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / "dwi.nii.gz")
    out = tmp_path / "odf.nii.gz"
    status = mspf_fit.main([str(tmp_path / "dwi.nii.gz"),
                            "-b", str(bval_path), "-r", str(bvec_path),
                            "-n", "3", "-l", "4", "-o", str(out)])
    assert status == 0
    odf = nib.load(out).get_fdata()
    assert odf.shape == (2, 2, 1, 15)
    assert np.all(np.isfinite(odf))
