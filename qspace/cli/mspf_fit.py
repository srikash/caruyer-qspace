"""Fit a modified SPF basis to a diffusion-weighted 4d image and compute the
marginal ODF.

The signal is reconstructed voxel-wise in the mSPF basis with Laplace
regularization, converted to the SPF basis, and the marginal ODF is saved as a
NIfTI image with one set of spherical-harmonic coefficients per voxel.

Requires the optional ``nibabel`` dependency (``pip install qspace[nifti]``).

Example::

    mspf_fit dwi.nii.gz -b dwi.bval -r dwi.bvec -m mask.nii.gz -o odf.nii.gz
"""
import argparse

import numpy as np

from qspace.bases import sh, spf, mspf
from qspace.sampling import io, sphere


def _require_nibabel():
    """Import and return nibabel, with an actionable error if it is missing."""
    try:
        import nibabel as nib
    except ImportError as error:  # pragma: no cover - depends on environment
        raise SystemExit("mspf_fit needs nibabel to read and write NIfTI "
                         "images. Install it with: pip install "
                         "'qspace[nifti]'") from error
    return nib


def main(argv=None):
    """Command-line entry point. Returns the process exit status."""
    description = ("Fit a mSPF basis to a 4d volume and reconstruct the "
                   "marginal ODF.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dwis",
                        help="DWIs as a 4d image in a format nibabel reads.")
    parser.add_argument("-b", dest="bvals", required=True,
                        help="Text file with the b-values.")
    parser.add_argument("-r", dest="bvecs", required=True,
                        help="Text file with the unit gradient directions.")
    parser.add_argument("-m", dest="mask",
                        help="Binary mask image in a format nibabel reads.")
    parser.add_argument("-o", dest="output", default="odf.nii.gz",
                        help="Output ODF image (%(default)s).")
    parser.add_argument("-n", dest="radial_order", type=int, default=3,
                        help="Radial truncation order of the (m)SPF basis "
                             "(%(default)s).")
    parser.add_argument("-l", dest="angular_rank", type=int, default=6,
                        help="Angular truncation rank of the (m)SPF basis "
                             "(%(default)s).")
    parser.add_argument("-t", dest="tau", type=float, default=50.0,
                        help="Diffusion time of the sequence, in ms "
                             "(%(default)s).")
    parser.add_argument("-z", dest="zeta", type=float, default=700.0,
                        help="Scale parameter of the (m)SPF basis "
                             "(%(default)s).")
    parser.add_argument("-a", dest="alpha", type=float, default=0.5,
                        help="Laplace regularization weight (%(default)s).")
    args = parser.parse_args(argv)
    nib = _require_nibabel()

    print("Reading acquisition sequence...")
    bvals, bvecs = io.read_bval_bvec(args.bvals, args.bvecs)

    print("Reading diffusion-weighted images...")
    dwis_img = nib.load(args.dwis)
    dwis = dwis_img.get_fdata()
    dim_x, dim_y, dim_z = dwis.shape[:3]
    affine = dwis_img.affine
    if dwis.shape[-1] != bvals.shape[0]:
        parser.error("the image has %d volumes but the gradient table has %d "
                     "entries." % (dwis.shape[-1], bvals.shape[0]))

    if args.mask is not None:
        mask = nib.load(args.mask).get_fdata()
        if mask.shape != (dim_x, dim_y, dim_z):
            parser.error("mask and image shape mismatch.")
    else:
        mask = np.ones((dim_x, dim_y, dim_z))

    print("Preparing reconstruction with Laplace regularization...")
    print("  average b=0 image")
    b0_indices = bvals == 0
    b0 = np.mean(dwis[..., b0_indices], axis=-1)

    print("  diffusion signal attenuation")
    dwis_indices = bvals > 0
    signal = dwis[..., dwis_indices] / b0[..., np.newaxis]
    del dwis, dwis_img, b0

    print("  modified SPH observation and Laplace regularization matrices")
    qnorms = np.sqrt(bvals[dwis_indices] / (4 * np.pi**2 * args.tau))
    theta, phi = sphere.to_spherical(bvecs[dwis_indices])
    H = mspf.matrix(qnorms, theta, phi, radial_order=args.radial_order,
                    angular_rank=args.angular_rank, zeta=args.zeta)
    Lambda = mspf.Lambda(args.radial_order, args.angular_rank, args.zeta)

    print("  equivalent signal (see Caruyer and Deriche, MedIA 2012)")
    signal -= np.exp(-qnorms**2 / (2 * args.zeta))[
        np.newaxis, np.newaxis, np.newaxis, :]

    x0 = -np.dot(np.linalg.inv(Lambda),
                 mspf.v(args.radial_order, args.angular_rank, args.zeta))
    signal -= np.dot(H, x0)[np.newaxis, np.newaxis, np.newaxis, :]

    print("Reconstruction with Laplace regularization...")
    pseudo_inv = np.dot(np.linalg.inv(np.dot(H.T, H) + args.alpha * Lambda),
                        H.T)
    mspf_coefficients = np.dot(signal, pseudo_inv.T)
    mspf_coefficients += x0[np.newaxis, np.newaxis, np.newaxis, :]

    print("Computing marginal ODF...")
    print("  converting to the SPF basis")
    M = mspf.to_spf_matrix(args.radial_order, args.angular_rank, args.zeta)
    spf_coefficients = np.dot(mspf_coefficients, M.T)

    print("  evaluating ODF coefficients")
    dim_sh = sh.dimension(args.angular_rank)
    odf_coefficients = np.zeros((dim_x, dim_y, dim_z, dim_sh))
    spherical_polar_fourier = spf.SphericalPolarFourier(
        args.radial_order, args.angular_rank, args.zeta)
    for i, j, k in zip(*np.nonzero(mask)):
        spherical_polar_fourier.coefficients = \
            spf_coefficients[i, j, k].reshape((args.radial_order, dim_sh))
        odf_coefficients[i, j, k] = spherical_polar_fourier.odf_marginal() \
            .coefficients

    odf_img = nib.Nifti1Image(odf_coefficients, affine)
    io.ensure_parent_dir(args.output)
    nib.save(odf_img, args.output)
    print("Saved ODF coefficients to %s." % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
