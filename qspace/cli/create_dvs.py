"""Create a Siemens ``.dvs`` gradient table from a multiple-shell scheme.

The input is a direction set as published by E. Caruyer, one line per
direction: a shell index followed by the unit vector ``x y z``. The b-value of
each shell is given on the command line. Optionally, b=0 acquisitions are
added, either all at the start (default) or interleaved between batches of
directions (``-i``).

Example::

    create_dvs shells.txt 8 1000 2000 3000 -o scheme.dvs
"""
import argparse

import numpy as np

from qspace.sampling import io


def prepend_b0(input_dirs, nb_b0):
    """Insert ``nb_b0`` zero rows at the start of a table.

    Parameters
    ----------
    input_dirs : array-like, shape (nb_dirs, k)
        Table of rows, for example ``(b, x, y, z)``.
    nb_b0 : int
        Number of rows to add.

    Returns
    -------
    ndarray, shape (nb_dirs + nb_b0, k)
    """
    input_dirs = np.asarray(input_dirs, dtype=float)
    nb_dirs = input_dirs.shape[0]
    result = np.zeros((nb_dirs + nb_b0, input_dirs.shape[1]))
    result[nb_b0:] = input_dirs
    return result


def interleave_b0(input_dirs, nb_b0):
    """Spread ``nb_b0`` zero rows evenly through a table.

    The table is split into ``nb_b0`` batches of equal size, and a b=0 row
    starts each batch. With no b=0 rows, the table is returned unchanged.

    Parameters
    ----------
    input_dirs : array-like, shape (nb_dirs, k)
        Table of rows, for example ``(b, x, y, z)``.
    nb_b0 : int
        Number of b=0 rows to add.

    Returns
    -------
    ndarray, shape (nb_dirs + nb_b0, k)
    """
    input_dirs = np.asarray(input_dirs, dtype=float)
    if nb_b0 == 0:
        return input_dirs.copy()
    nb_dirs = input_dirs.shape[0]
    nb_acquisitions = nb_dirs + nb_b0
    batch_size = nb_acquisitions // nb_b0
    result = np.zeros((nb_acquisitions, input_dirs.shape[1]))
    indices_b0 = np.zeros(nb_acquisitions, dtype=bool)
    indices_b0[[i * batch_size for i in range(nb_b0)]] = True
    result[np.logical_not(indices_b0)] = input_dirs
    return result


def build_table(shells, bvals, nb_b0, interleave):
    """Scale and order a shell scheme into a gradient table.

    Parameters
    ----------
    shells : ndarray, shape (nb_dirs, 4)
        Rows of ``(shell_index, x, y, z)`` as read from the input file.
    bvals : ndarray, shape (nb_shells,)
        b-value of each shell.
    nb_b0 : int
        Number of b=0 acquisitions to add.
    interleave : bool
        Interleave the b=0 rows rather than prepending them.

    Returns
    -------
    bvals_out : ndarray, shape (nb_acq,)
    bvecs_out : ndarray, shape (nb_acq, 3)
        Unit vectors, zero for b=0 rows.
    """
    shift = np.min(shells[:, 0])
    shell_indices = np.asarray(shells[:, 0] - shift, dtype=int)
    unit_dirs = shells[:, 1:] / np.linalg.norm(shells[:, 1:], axis=1)[:, None]
    per_row_b = bvals[shell_indices]

    # Columns: b-value, then unit vector. b=0 rows are all zeros.
    rows = np.column_stack((per_row_b, unit_dirs))
    if interleave:
        rows = interleave_b0(rows, nb_b0)
    else:
        rows = prepend_b0(rows, nb_b0)
    return rows[:, 0], rows[:, 1:]


def main(argv=None):
    """Command-line entry point. Returns the process exit status."""
    description = ("Create a dvs file from a multiple-shell direction set "
                   "(as published by E. Caruyer) and the b-value of each "
                   "shell.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input",
                        help="Multishell direction set from E. Caruyer's "
                             "website: shell index and x y z per line.")
    parser.add_argument("-i", dest="interleave_b0", action="store_true",
                        help="Interleave the b=0 acquisitions between "
                             "batches of directions instead of prepending "
                             "them.")
    parser.add_argument("nb_b0", type=int,
                        help="Number of b=0 acquisitions.")
    parser.add_argument("bvals", type=float, nargs="+",
                        help="b-value of each shell, in order of the shell "
                             "index.")
    parser.add_argument("-o", dest="output", default="output.dvs",
                        help="Output dvs filename (%(default)s).")
    args = parser.parse_args(argv)

    shells = np.loadtxt(args.input, ndmin=2)
    shift = np.min(shells[:, 0])
    nb_shells = int(np.max(shells[:, 0]) + 1 - shift)
    if len(args.bvals) != nb_shells:
        parser.error("the input has %d shells but %d b-values were given."
                     % (nb_shells, len(args.bvals)))
    bvals = np.array(args.bvals)
    print("Shell b-values:", bvals)

    table_bvals, table_bvecs = build_table(
        shells, bvals, args.nb_b0, args.interleave_b0)
    io.ensure_parent_dir(args.output)
    io.write(table_bvals, table_bvecs, args.output, format="siemens")
    print("Wrote %d acquisitions to %s." % (table_bvals.shape[0], args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
