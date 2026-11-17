"""Reading and writing gradient-waveform scheme files.

Two text formats are supported, both starting with a ``VERSION:`` or
``GRADIENT_WAVEFORM`` header line:

* Camino: one line per waveform, ``nb_timesteps dt gx gy gz gx gy gz ...``.
* MCDC: one line with ``te nb_steps nb_waveforms`` followed by all samples.

Gradients are in T/m and times in s.
"""
import numpy as np


def write_mcdc(waveforms, dt, filename):
    """Write a set of waveforms to an MCDC-compatible scheme file.

    All waveforms must have the same number of time steps.

    Parameters
    ----------
    waveforms : sequence of arrays, each of shape (nb_timesteps, 3)
        Gradient trajectories, in T/m.
    dt : float
        Time step, in s.
    filename : str
        Output path.

    Raises
    ------
    ValueError
        If the waveforms do not all have the same length.
    """
    nb_waveforms = len(waveforms)
    nb_steps = len(waveforms[0])
    for waveform in waveforms:
        if len(waveform) != nb_steps:
            raise ValueError("MCDC is expecting waveforms "
                             "with equal number of bins.")
    te = nb_steps * dt
    with open(filename, "w") as schemefile:
        schemefile.write("VERSION: GRADIENT_WAVEFORM\n")
        schemefile.write("%.6f %d %d " % (te, nb_steps, nb_waveforms))
        for waveform in waveforms:
            for k in range(nb_steps):
                schemefile.write("%.6f %.6f %.6f " %
                                 (waveform[k, 0], waveform[k, 1],
                                  waveform[k, 2]))


def write_camino(waveforms, dt, filename):
    """Write a set of waveforms to a Camino-compatible scheme file.

    Waveforms may have different numbers of time steps.

    Parameters
    ----------
    waveforms : sequence of arrays, each of shape (nb_timesteps, 3)
        Gradient trajectories, in T/m.
    dt : float
        Time step, in s.
    filename : str
        Output path.
    """
    with open(filename, "w") as schemefile:
        schemefile.write("VERSION: GRADIENT_WAVEFORM\n")
        for waveform in waveforms:
            nb_timesteps, _ = waveform.shape
            schemefile.write("%d %.9g " % (nb_timesteps, dt))
            for k in range(nb_timesteps):
                schemefile.write("%.6f %.6f %.6f " %
                                 (waveform[k, 0], waveform[k, 1],
                                  waveform[k, 2]))
            schemefile.write("\n")


def read_mcdc(filename):
    """Read an MCDC-compatible scheme file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    waveforms : list of ndarray, each of shape (nb_timesteps, 3)
        Gradient trajectories, in T/m.
    dt : float
        Time step, in s.

    Raises
    ------
    ValueError
        If the header is missing or the number of samples is inconsistent.
    """
    with open(filename, "r") as schemefile:
        header = schemefile.readline()
        if "GRADIENT_WAVEFORM" not in header:
            raise ValueError("Expecting a mcdc-like waveform description.")
        line = schemefile.readline()
    values = np.array(line.split(), dtype=float)
    te = values[0]
    nb_steps = int(values[1])
    nb_waveforms = int(values[2])
    samples = values[3:]
    dt = te / nb_steps
    if samples.shape[0] != nb_steps * 3 * nb_waveforms:
        raise ValueError("Incomplete scheme")
    waveforms = []
    for i in range(nb_waveforms):
        wf = samples[i * nb_steps * 3:nb_steps * 3 * (i + 1)]
        waveforms.append(wf.reshape((nb_steps, 3)))
    return waveforms, dt


def read_camino(filename):
    """Read a Camino-compatible scheme file.

    Blank lines are ignored.

    Parameters
    ----------
    filename : str

    Returns
    -------
    waveforms : list of ndarray, each of shape (nb_timesteps, 3)
        Gradient trajectories, in T/m.
    dt : float
        Time step of the last waveform read, in s. All waveforms are expected
        to share it.

    Raises
    ------
    ValueError
        If the header is missing or a waveform has the wrong number of samples.
    """
    waveforms = []
    dt = 0.0
    with open(filename, "r") as schemefile:
        header = schemefile.readline()
        if "GRADIENT_WAVEFORM" not in header:
            raise ValueError("Expecting a Camino-like waveform description.")
        for line in schemefile:
            if not line.strip():
                continue
            values = np.array(line.split(), dtype=float)
            nb_timesteps = int(values[0])
            dt = values[1]
            waveform = values[2:]
            if waveform.shape[0] != nb_timesteps * 3:
                raise ValueError("Expected number of timesteps mismatch.")
            waveforms.append(waveform.reshape((nb_timesteps, 3)))
    return waveforms, dt
