"""B-tensors of gradient waveforms and waveform transforms.

For a gradient waveform ``G(t)`` (T/m), the diffusion-encoding b-tensor is
``B = int_0^TE q(t) q(t)^T dt``, where ``q(t) = gamma int_0^t G(s) ds``.

The functions follow the q-space trajectory imaging framework of Westin et al.
(2016), NeuroImage 135, 345-362.
"""
import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid


#: Gyromagnetic ratio of the proton, in rad/s/T.
GAMMA = 267.513e6

#: Alias kept for compatibility with earlier code.
gamma = GAMMA


def compute_b(waveform, dt):
    """Compute the b-tensor of a discretized gradient waveform.

    Parameters
    ----------
    waveform : array-like, shape (nb_steps, 3)
        Gradient waveform, in T/m.
    dt : float
        Time step, in s.

    Returns
    -------
    B : ndarray, shape (3, 3)
        b-tensor, in s/m^2.
    """
    waveform = np.asarray(waveform, dtype=float)
    qs = GAMMA * cumulative_trapezoid(waveform, dx=dt, axis=0, initial=0)
    qqt = np.einsum("ij,ik->ijk", qs, qs)
    B = trapezoid(qqt, dx=dt, axis=0)
    return B


def transform_waveform(source_b_tensor, target_b_tensor):
    """Compute the linear map that turns a source b-tensor into a target one.

    Applying the returned matrix ``M`` to a waveform, ``G -> G @ M.T``, gives a
    waveform whose b-tensor is the target, following Westin et al. (2016).

    Parameters
    ----------
    source_b_tensor : array-like, shape (3, 3)
        b-tensor of the waveform to transform, in s/m^2.
    target_b_tensor : array-like, shape (3, 3)
        Desired b-tensor, in s/m^2.

    Returns
    -------
    M : ndarray, shape (3, 3)
        The transform matrix.

    Raises
    ------
    ValueError
        If the source has lower rank than the target, for example a linear
        source cannot become a spherical target. Also raised if the result
        contains NaN values.

    References
    ----------
    Westin, C.-F., et al. (2016). Q-space trajectory imaging for
    multidimensional diffusion MRI of the human brain. NeuroImage 135, 345-362.
    """
    # Eigen decomposition of the source and target b-tensors.
    b1 = np.asarray(source_b_tensor, dtype=float)
    b2 = np.asarray(target_b_tensor, dtype=float)
    w1, v1 = np.linalg.eigh(b1)
    w2, v2 = np.linalg.eigh(b2)

    # The rank of the source must be at least that of the target.
    epsilon = 1.0e-6
    if np.sum(w1 > epsilon) < np.sum(w2 > epsilon):
        raise ValueError("Trying to map a b-matrix to higher dimension.")

    # Scale the eigen-directions; zero out directions with no weight.
    ratio = np.where(np.logical_or(w1 < epsilon, w2 < epsilon), 0,
                     np.sqrt(w2 / w1))
    if np.any(np.isnan(ratio)):
        raise ValueError("Transform has NaN values: w1=%s, w2=%s."
                         % (w1, w2))
    return np.dot(v2, np.dot(np.diag(ratio), v1.T))
