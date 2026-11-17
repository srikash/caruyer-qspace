"""Visualize the diffusion-encoding directions of an FSL gradient table.

The b-values are grouped into shells (rounded to the nearest 100 s/mm^2), and
each shell is drawn in its own panel. Use ``-o`` to save the figure to a file
instead of opening a window.

Example::

    visu_btable -b scheme.bval -r scheme.bvec -o scheme.png
"""
import argparse

import numpy as np
from matplotlib import pyplot as plt

from qspace.sampling import io
from qspace.visu import visu_points


def main(argv=None):
    """Command-line entry point. Returns the process exit status."""
    description = ("Visualize diffusion-encoding gradient directions as "
                   "described in an FSL gradient table.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-b", dest="bvals", required=True,
                        help="FSL .bval file.")
    parser.add_argument("-r", dest="bvecs", required=True,
                        help="FSL .bvec file.")
    parser.add_argument("-o", dest="output",
                        help="Save the figure to this file (e.g. .png) "
                             "instead of showing it.")
    args = parser.parse_args(argv)

    bvals, bvecs = io.read_bval_bvec(args.bvals, args.bvecs)
    bvals = np.round(bvals, decimals=-2)
    shell_values = np.unique(bvals[bvals > 0])
    nb_shells = len(shell_values)
    if nb_shells == 0:
        parser.error("the table contains no diffusion-weighted volume.")

    vects = np.zeros((0, 3))
    points_per_shell = []
    rs = []
    for bval in shell_values:
        dirs = bvecs[bvals == bval]
        vects = np.concatenate((vects, dirs))
        points_per_shell.append(dirs.shape[0])
        rs.append(np.sqrt(bval / np.max(bvals)))

    print("Shells (b-value: number of directions):")
    for bval, count in zip(shell_values, points_per_shell):
        print("  %g: %d" % (bval, count))

    fig = plt.figure(figsize=(6 * nb_shells, 6))
    visu_points.draw_points_sphere(vects, nb_shells, points_per_shell, rs, fig)
    if args.output:
        io.ensure_parent_dir(args.output)
        fig.savefig(args.output, dpi=150)
        print("Saved figure to %s." % args.output)
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
