"""Small numerical helpers shared by the basis modules."""
from scipy.special import binom


def binomial(alpha, k):
    """Return the generalized binomial coefficient ``alpha choose k``.

    Unlike the integer binomial, ``alpha`` may be any real number, for example
    ``binomial(0.5, 2) == -1/8``. The value is
    ``alpha (alpha - 1) ... (alpha - k + 1) / k!``.

    Parameters
    ----------
    alpha : float
        Upper argument, any real number.
    k : int
        Lower argument, a non-negative integer.

    Returns
    -------
    float
        The generalized binomial coefficient.

    Notes
    -----
    Earlier versions multiplied by ``(alpha - k)`` instead of
    ``(alpha - i)`` at each step. That changed the values used by
    :func:`qspace.bases.spf.SphericalPolarFourier.odf_marginal`; this version
    is the standard definition.
    """
    return float(binom(alpha, k))
