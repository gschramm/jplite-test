"""example pdfs for testing the Barankin bound"""

import math
from numba import njit


@njit
def exp_pdf(x: float, mu: float = 1.0) -> float:
    """exponential pdf

    Parameters
    ----------
    x : float
        independent variable
    mu : float, optional
        exponent, by default 1.0

    Returns
    -------
    float
    """
    if x >= 0:
        p = mu * math.exp(-mu * x)
    else:
        p = 0
    return p


@njit
def truncated_gauss_pdf(x: float, mu: float = 2.0, sig: float = 3.0) -> float:
    """truncate gaussian pdf (0 for x < 0)

    Parameters
    ----------
    x : float
        independent variable
    mu : float, optional
        mean, default 2.0
    sig : float, optional
        standard deviation, default 3.0

    Returns
    -------
    float
    """
    if x >= 0:
        p = math.exp(-0.5 * ((x - mu) / sig) ** 2)
    else:
        p = 0
    return p


@njit
def truncated_double_gauss_pdf(
    x: float,
    mu1: float = 7.0 / 100,
    sig1: float = 2.0 / 100,
    mu2: float = 150.0 / 100,
    sig2: float = 60.0 / 100,
) -> float:
    """truncate double gaussian pdf (0 for x < 0)

    Parameters
    ----------
    x : float
        independent variable
    mu1/mu2 : float, optional
        mean, default 7.0/70.9
    sig1/sig2 : float, optional
        standard deviation, default 3.0/60.0

    Returns
    -------
    float
    """
    if x >= 0:
        p = (0.1 / math.sqrt(2 * math.pi * sig1**2)) * math.exp(
            -0.5 * ((x - mu1) / sig1) ** 2
        ) + (0.9 / math.sqrt(2 * math.pi * sig2**2)) * math.exp(
            -0.5 * ((x - mu2) / sig2) ** 2
        )
    else:
        p = 0
    return p


@njit
def _a(t: float, tau: float, t_tr: float, sig_tr: float) -> float:

    x1 = (t_tr + sig_tr * sig_tr / tau) / (math.sqrt(2) * sig_tr)
    x2 = (t - t_tr - sig_tr * sig_tr / tau) / (math.sqrt(2) * sig_tr)

    xmax = 26.0

    if (x1 > xmax) and (x2 < -xmax):
        a1 = (
            sig_tr
            * math.exp(-t_tr * t_tr / (2 * sig_tr * sig_tr) - t / tau)
            / (math.sqrt(2 * math.pi) * (t_tr + sig_tr * sig_tr / tau))
        )
        a2 = (
            sig_tr
            * math.exp(-(t_tr - t) * (t_tr - t) / (2 * sig_tr * sig_tr))
            / (math.sqrt(2 * math.pi) * (t_tr + sig_tr * sig_tr / tau - t))
        )
        a = a2 - a1
    else:
        a = (
            0.5
            * math.exp(-((t - t_tr) / tau) + 0.5 * sig_tr * sig_tr / (tau * tau))
            * (math.erfc(-x2) - math.erfc(x1))
        )

    return a


@njit
def bi_exp_model(
    t: float, tau_d: float, tau_r: float, t_tr: float, sig_tr: float
) -> float:

    if t < 0:
        return 0

    else:
        C = 2 / (1 + math.erf(t_tr / (math.sqrt(2) * sig_tr)))
        return (C / (tau_d - tau_r)) * (
            _a(t, tau_d, t_tr, sig_tr) - _a(t, tau_r, t_tr, sig_tr)
        )


@njit
def det1_pdf(t: float, alpha: float = 0.05) -> float:

    return alpha * bi_exp_model(t, 1.6, 0.35, 0.179, 0.081) + (
        1 - alpha
    ) * bi_exp_model(t, 40.0, 0.01, 0.179, 0.081)


@njit
def det2_pdf(t: float, alpha: float = 0.05) -> float:

    return alpha * bi_exp_model(t, 1.0, 0.01, 0.179, 0.081) + (
        1 - alpha
    ) * bi_exp_model(t, 20.0, 5.0, 0.179, 0.081)


@njit
def double_biexp_pdf(
    t: float,
    alpha: float = 0.004,
    tau_d_cer: float = 3e-3,
    tau_r_cer: float = 1e-3,
    tau_d_sci: float = 300.0,
    tau_r_sci: float = 1e-3,
    t_tr: float = 0.179,
    sig_tr: float = 0.081,
) -> float:
    # approximating a BGO scintillator with slightly longer rise times to
    # avoid numerical problems
    # cerenkov 3ps decay time, 1ps rise time
    # scintillation 300ns decay time, 1ps rise time
    # 179ps mean optical transfer time
    # 81ps sigma of transfer time
    # alpha fraction of cerenkov photons

    return alpha * bi_exp_model(t, tau_d_cer, tau_r_cer, t_tr, sig_tr) + (
        1 - alpha
    ) * bi_exp_model(t, tau_d_sci, tau_r_sci, t_tr, sig_tr)


@njit
def test_pdf(t: float, alpha: float = 0.05) -> float:

    if t < 200:
        return alpha * bi_exp_model(t, 1.0, 0.01, 0.179, 0.081) + (
            1 - alpha
        ) * bi_exp_model(t, 20.0, 5.0, 0.179, 0.081)
    else:
        return 0


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    tmax = 200

    tt = np.linspace(0, tmax, tmax * 2000 + 1)

    p2 = np.array([det2_pdf(t) for t in tt])

    fig, ax = plt.subplots()
    ax.plot(tt, p2, ".-")
    ax.grid(ls=":")
    fig.show()
