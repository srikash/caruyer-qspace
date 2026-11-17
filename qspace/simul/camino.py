"""Monte-Carlo diffusion simulations with Camino.

This module wraps Camino's ``datasynth`` command, which must be installed and on
the ``PATH`` (see http://camino.org.uk). Simulations write a binary ``.traj``
file, which :func:`parse_trajfile` converts to a NumPy array.

Run a small simulation from the command line with::

    python -m qspace.simul.camino -n 100 -m 1000 -o walkers.npy
"""
import argparse
import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np

from qspace.simul.substrates import CylindersLattice


class CaminoSimulation:
    """Settings of a Camino ``datasynth`` run.

    Parameters
    ----------
    substrate : qspace.simul.substrates.Substrate
        Geometry of the simulation.
    trajfile : str
        Path of the binary ``.traj`` output file.
    nb_walkers : int
        Number of random walkers.
    nb_steps : int
        Number of time steps recorded per walker.
    duration : float
        Duration of the diffusion experiment, in s.
    seed : int
        Random seed passed to Camino.
    initial : {"uniform", "spike", "intra", "extra"}
        Initial position distribution.
    permeability : float
        Membrane permeability.
    diffusivity : float
        Free diffusion coefficient, in m^2/s.
    """

    def __init__(self, substrate, trajfile,
                 nb_walkers=1000, nb_steps=1000, duration=0.1, seed=1,
                 initial="uniform", permeability=0.0, diffusivity=2.0e-9):
        self.substrate = substrate
        self.trajfile = trajfile
        self.nb_walkers = nb_walkers
        self.nb_steps = nb_steps
        self.duration = duration
        self.seed = seed
        self.initial = initial
        self.permeability = permeability
        self.diffusivity = diffusivity

    def get_options(self):
        """Return the Camino options of this run, as a dict of strings."""
        return {
            "-walkers": "%d" % self.nb_walkers,
            "-tmax": "%d" % (self.nb_steps - 1),
            "-duration": "%f" % self.duration,
            "-trajfile": self.trajfile,
            "-seed": "%d" % self.seed,
            "-initial": self.initial,
            "-p": "%f" % self.permeability,
            "-diffusivity": "%.2e" % self.diffusivity,
        }

    def run(self):
        """Run ``datasynth`` and write the trajectories to ``trajfile``.

        Raises
        ------
        RuntimeError
            If ``datasynth`` is not found on the ``PATH``.
        subprocess.CalledProcessError
            If ``datasynth`` exits with an error.
        """
        if shutil.which("datasynth") is None:
            raise RuntimeError(
                "Camino's 'datasynth' was not found on the PATH. Install "
                "Camino (http://camino.org.uk) to run simulations.")
        command = ["datasynth", "-voxels", "1"]
        for key, value in self.get_options().items():
            command.extend([key, value])
        for key, value in self.substrate.camino_options.items():
            command.extend([key, value])
        result = subprocess.run(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError("datasynth failed with exit status %d: %s"
                               % (result.returncode, result.stderr.strip()))

    def __str__(self):
        return "_".join(f"{key}={value}"
                        for key, value in self.get_options().items())


def _read_header(fhandle):
    """Read the trajfile header.

    Layout (big-endian, per the Camino documentation): dynamics duration
    (double), number of spins (double), tmax (double).

    Returns
    -------
    duration : float
    nb_spins : int
    nb_steps : int
        ``tmax + 1``.
    """
    sformat = ">ddd"
    size = struct.calcsize(sformat)
    duration, nb_spins, tmax = struct.unpack(sformat, fhandle.read(size))
    return duration, int(nb_spins), int(tmax) + 1


def parse_trajfile(filename):
    """Parse a Camino trajfile into an array of walker positions.

    Parameters
    ----------
    filename : str
        Path to a ``.traj`` file.

    Returns
    -------
    trajectories : ndarray, shape (nb_spins, nb_steps, 3)
        Position ``(x, y, z)`` of each walker at each time step.

    Raises
    ------
    ValueError
        If the file does not contain the expected number of samples.
    """
    with open(filename, "rb") as trajfile:
        _, nb_spins, nb_steps = _read_header(trajfile)
        indices = np.zeros(nb_spins, dtype=int)
        trajectories = np.zeros((nb_spins, nb_steps, 3))
        # Each update: time (double), spin index (int32), x, y, z (doubles).
        sformat = ">diddd"
        for _, i, x, y, z in struct.iter_unpack(sformat, trajfile.read()):
            trajectories[i, indices[i]] = (x, y, z)
            indices[i] += 1
    if not np.all(indices == nb_steps):
        raise ValueError("Trajfile is incomplete: not every walker has "
                         "%d recorded steps." % nb_steps)
    return trajectories


def main(argv=None):
    """Command-line entry point. Returns the process exit status."""
    parser = argparse.ArgumentParser(
        description="Generate trajectories with Camino's Monte-Carlo "
                    "simulator (datasynth) in a cylinder lattice.")
    parser.add_argument("-n", dest="nb_walkers", type=int, default=100,
                        help="Number of walkers (%(default)s).")
    parser.add_argument("-m", dest="nb_steps", type=int, default=1000,
                        help="Number of timesteps (%(default)s).")
    parser.add_argument("-t", dest="duration", type=float, default=0.1,
                        help="Duration of the simulation in s (%(default)s).")
    parser.add_argument("-r", dest="radius", type=float, default=1e-6,
                        help="Cylinder radius in m (%(default)s).")
    parser.add_argument("-d", dest="density", type=float, default=0.50,
                        help="Cylinder density, i.e. intra-axonal volume "
                             "fraction (%(default)s).")
    parser.add_argument("-o", dest="output", default="trajectories.traj",
                        help="Output file: Camino .traj or NumPy .npy "
                             "(%(default)s).")
    args = parser.parse_args(argv)

    if not args.output.endswith((".traj", ".npy")):
        parser.error("output must end with .traj or .npy")

    fd, tmp_path = tempfile.mkstemp(suffix=".traj")
    os.close(fd)
    try:
        substrate = CylindersLattice(args.radius, density=args.density)
        simulation = CaminoSimulation(
            substrate, tmp_path, nb_walkers=args.nb_walkers,
            nb_steps=args.nb_steps, duration=args.duration)
        print("Running Camino simulation...")
        simulation.run()
        if args.output.endswith(".traj"):
            shutil.move(tmp_path, args.output)
            tmp_path = None
        else:
            print("Parsing trajfile and saving to %s." % args.output)
            np.save(args.output, parse_trajfile(tmp_path))
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
