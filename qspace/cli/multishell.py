"""Generate an optimized multiple-shell sampling scheme.

Given the number of directions on each shell, the directions are optimized to
spread evenly over the sphere, with shells coupled to one another (see
:mod:`qspace.sampling.multishell`). The scheme is written to a text file and
two figures are saved for visual inspection.

Example::

    multishell 16 32 48 --seed 0 -o scheme3shells
"""
import argparse

import numpy as np
from matplotlib import pyplot as plt

from qspace.sampling import io
from qspace.sampling import multishell as ms
from qspace.visu import visu_points


def _initial_directions(K, seed):
    """Return K random unit vectors from a generator seeded with ``seed``."""
    rng = np.random.default_rng(seed)
    vects = rng.normal(size=(K, 3))
    return vects / np.linalg.norm(vects, axis=1)[:, np.newaxis]


def main(argv=None):
    """Command-line entry point. Returns the process exit status."""
    parser = argparse.ArgumentParser(
        description="Generate a multiple-shell sampling scheme with optimal "
                    "angular coverage (Caruyer et al., MRM 2013).")
    parser.add_argument("points_per_shell", nargs="+", type=int,
                        help="Number of points on each shell.")
    parser.add_argument("-o", dest="output",
                        help="Output basename. Writes <basename>.txt and two "
                             "figures. Default is derived from the shell "
                             "sizes.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random starting directions, for "
                             "reproducible schemes.")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Maximum optimizer iterations (%(default)s).")
    args = parser.parse_args(argv)

    if any(n < 1 for n in args.points_per_shell):
        parser.error("each shell needs at least one point.")

    nb_shells = len(args.points_per_shell)
    K = int(np.sum(args.points_per_shell))

    # Relative shell radii (for visualization only).
    rs = (np.arange(nb_shells) + 1) / nb_shells

    # Groups of shells to couple: each shell alone, and all shells together.
    shell_groups = [[i] for i in range(nb_shells)]
    shell_groups.append(list(range(nb_shells)))
    alphas = np.ones(len(shell_groups))
    weights = ms.compute_weights(nb_shells, args.points_per_shell,
                                 shell_groups, alphas)

    init = _initial_directions(K, args.seed)
    points = ms.optimize(nb_shells, args.points_per_shell, weights,
                         max_iter=args.max_iter, init_points=init)

    if args.output:
        basename = args.output
    else:
        basename = "%02d-shells-%s" % (
            nb_shells, "-".join(str(n) for n in args.points_per_shell))

    filename = "%s.txt" % basename
    io.ensure_parent_dir(filename)
    ms.write(points, nb_shells, args.points_per_shell, filename)
    print("Wrote scheme to %s." % filename)

    spacing = 0.05

    # Figure 1: each shell in its own 3d panel.
    fig = plt.figure(figsize=(3.0 * nb_shells, 3.0))
    plt.subplots_adjust(left=spacing / nb_shells,
                        right=1 - spacing / nb_shells,
                        bottom=spacing,
                        top=1 - spacing,
                        wspace=2 * spacing / nb_shells)
    visu_points.draw_points_sphere(points, nb_shells, args.points_per_shell,
                                   rs, fig)
    plt.suptitle("Individual shells")
    plt.savefig("%s-shells.png" % basename, dpi=200)
    plt.close(fig)

    # Figure 2: all shells reprojected onto the unit sphere.
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=spacing / nb_shells,
                        right=1 - spacing / nb_shells,
                        bottom=spacing,
                        top=1 - spacing,
                        wspace=2 * spacing / nb_shells)
    visu_points.draw_points_reprojected(points, nb_shells,
                                        args.points_per_shell, rs, ax)
    plt.suptitle("Shells reprojected")
    plt.savefig("%s-shells-reprojected.png" % basename, dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
