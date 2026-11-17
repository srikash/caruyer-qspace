"""Reading and writing diffusion-encoding gradient tables.

Supported formats:

* ``siemens``: the ``.dvs`` text format used by Siemens scanners. Each
  direction is written as a vector whose norm is ``sqrt(b / b_max)``.
* ``mrtrix``: one line per volume, ``gx gy gz b``.
* ``fsl``: a pair of files ``<basename>.bval`` (one line of b-values) and
  ``<basename>.bvec`` (three lines, x, y and z, of unit directions).
"""
import os

import numpy as np


_FORMATS = ("siemens", "mrtrix", "fsl")


def write(bvals, bvecs, filename, format="siemens"):
    """Write a gradient table to disk.

    Parameters
    ----------
    bvals : array-like, shape (nb_points,)
        b-values, in s/mm^2 (or s/m^2, consistently with the rest of the
        package).
    bvecs : array-like, shape (nb_points, 3)
        Gradient directions. Unit vectors are expected, except for b=0 rows.
    filename : str
        Output path. For ``format="fsl"``, this is the *basename*: the files
        ``<filename>.bval`` and ``<filename>.bvec`` are written.
    format : {"siemens", "mrtrix", "fsl"}
        Output format.

    Raises
    ------
    ValueError
        If the shapes do not match or ``format`` is not supported.
    """
    bvals = np.asarray(bvals, dtype=float)
    bvecs = np.asarray(bvecs, dtype=float)
    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        raise ValueError("The b-vectors should contain 3 coordinates.")
    if bvals.shape[0] != bvecs.shape[0]:
        raise ValueError("Number of b-values and b-vectors do not match.")
    if format not in _FORMATS:
        raise ValueError("Unknown format %r; expected one of %s."
                         % (format, ", ".join(_FORMATS)))
    if format == "siemens":
        _write_siemens(bvals, bvecs, filename)
    elif format == "mrtrix":
        _write_mrtrix(bvals, bvecs, filename)
    else:
        write_fsl(bvals, bvecs, filename)


def _write_siemens(bvals, bvecs, filename):
    """Write a Siemens ``.dvs`` file. See :func:`write`."""
    nb_points = bvals.shape[0]
    epsilon = 1.0e-6
    b_norms = np.sqrt(bvals / (epsilon + np.max(bvals)))
    b_table = b_norms[:, np.newaxis] * bvecs
    with open(filename, "w") as dvs_file:
        dvs_file.write("[directions=%d]\n" % nb_points)
        dvs_file.write("CoordinateSystem = xyz\n")
        dvs_file.write("Normalisation = none\n")
        for i in range(nb_points):
            dvs_file.write("Vector[%d] = ( %.6f, %.6f, %.6f )\n" %
                           (i, b_table[i, 0], b_table[i, 1], b_table[i, 2]))


def _write_mrtrix(bvals, bvecs, filename):
    """Write an MRtrix gradient table (``gx gy gz b`` per line)."""
    table = np.column_stack((bvecs, bvals))
    np.savetxt(filename, table, fmt="%.6f")


def write_fsl(bvals, bvecs, basename):
    """Write FSL ``.bval`` and ``.bvec`` files.

    Parameters
    ----------
    bvals : array-like, shape (nb_points,)
    bvecs : array-like, shape (nb_points, 3)
    basename : str
        Path without extension. ``.bval`` and ``.bvec`` are appended.
    """
    bvals = np.asarray(bvals, dtype=float)
    bvecs = np.asarray(bvecs, dtype=float)
    np.savetxt(basename + ".bval", bvals[np.newaxis, :], fmt="%.6f")
    np.savetxt(basename + ".bvec", bvecs.T, fmt="%.6f")


def ensure_parent_dir(path):
    """Create the parent directory of ``path`` if it does not exist yet.

    Parameters
    ----------
    path : str
        A file path that is about to be written.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)


def read_fsl(basename):
    """Read FSL ``.bval`` and ``.bvec`` files given their common basename.

    Parameters
    ----------
    basename : str
        Path without extension. ``.bval`` and ``.bvec`` are appended.

    Returns
    -------
    bvals : ndarray, shape (nb_points,)
    bvecs : ndarray, shape (nb_points, 3)
    """
    return read_bval_bvec(basename + ".bval", basename + ".bvec")


def read_bval_bvec(bval_path, bvec_path):
    """Read a b-value file and a b-vector file.

    The b-vector file has three rows (x, y, z) in the standard FSL layout.
    Tables with one row per volume (three columns) are also accepted.

    Parameters
    ----------
    bval_path : str
        Text file with the b-values.
    bvec_path : str
        Text file with the gradient directions.

    Returns
    -------
    bvals : ndarray, shape (nb_points,)
    bvecs : ndarray, shape (nb_points, 3)

    Raises
    ------
    ValueError
        If the files do not describe the same number of volumes.
    """
    bvals = np.atleast_1d(np.loadtxt(bval_path))
    bvecs = np.loadtxt(bvec_path, ndmin=2)
    if bvecs.ndim != 2:
        raise ValueError("bvec file should contain a 2d table.")
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    if bvecs.shape[1] != 3:
        raise ValueError("bvec file should contain 3 rows/columns (x, y, z).")
    if bvals.shape[0] != bvecs.shape[0]:
        raise ValueError("bval/bvec tables should have the same number of "
                         "entries.")
    return bvals, bvecs
