# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import functools

from scipy.integrate import quad

from collections.abc import Callable
from utils import barankin_bound

# %%
# setup the pdf function
# you can replace this cell with any other pdf function that is Callable[[float], float]

from pdfs import double_biexp_pdf

# key word arguments for the (double_biexp_pdf) pdf function
pdf_kwargs: dict = dict(
    alpha=0.0064,  # fraction of emission where photon is Cerenkov
    tau_d_cer=0.01,  # decay time of Cerenkov in ns
    tau_r_cer=0.005,  # rise time of Cerenkov in ns
    tau_d_sci=300.0,  # decay time of scintillation in ns
    tau_r_sci=0.05,  # rise time of scintillation in ns
    t_tr=0.179,  # mean optical transfer time in ns
    sig_tr=0.081,  # sigma of transfer time in ns
)

pdf: Callable[[float], float] = functools.partial(double_biexp_pdf, **pdf_kwargs)

# -------------
# at the end of this cell, "pdf" must be a Callable mapping a float onto a float pdf(t[ns])
# -------------


# %%
# input parameters

# number of possible deltas
num_possible_deltas: int = 256
# minimum delta to consider, None mean auto determined
delta_min: float | None = None
# maximum delta to consider, None mean auto determined
delta_max: float | None = None
# number of photons / samples
N: int = 500
# maximum J value
Jmax: int = 32
# point beyond which pdf is essentially zero, None means auto determined
x_zero: float | None = None
# fraction of largest singular value for calculation of pseudo inverse
rcond: float = 1e-12
# show interactive plots on how J values are chosen, requires user interaction
interactive: bool = False
# show condition number of U matrix
show_cond_number: bool = False


# %%
# estimate the point beyond which the pdf is essentially zero (smaller than 1e-5)

if x_zero is None:
    xx = np.logspace(-8, 8, 10000)
    test_pdf = np.array([pdf(x) for x in xx])
    x_zero = xx[np.where(test_pdf >= 1e-5)[0].max() + 1]

# %%
# normalize the pdf

norm = quad(pdf, 0, x_zero)[0]
normalized_pdf = lambda x: pdf(x) / norm

stddev_pdf = np.sqrt(
    quad(lambda x: (x**2) * normalized_pdf(x), 0, x_zero)[0]
    - quad(lambda x: x * normalized_pdf(x), 0, x_zero)[0] ** 2
)

# %%
# plot the (normalized) pdf on a linear and log x scale
iplot_min = np.where(test_pdf >= 0.02 * test_pdf.max())[0].min()
iplot_max = np.where(test_pdf >= 0.02 * test_pdf.max())[0].max() + 1

fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
ax[0, 0].plot(xx[iplot_min:iplot_max], test_pdf[iplot_min:iplot_max])
ax[0, 0].set_title(r"input pdf $\tilde{p}(x)$", fontsize="medium")

ax[1, 0].plot(xx[iplot_min:iplot_max], test_pdf[iplot_min:iplot_max] / norm)
ax[1, 0].set_title(
    r"normalized pdf $p(x)$ with $\int_0^\infty p(x) dx = 1$", fontsize="medium"
)

ax[0, 1].semilogx(xx[iplot_min:iplot_max], test_pdf[iplot_min:iplot_max])
ax[0, 1].set_title(r"input pdf $\tilde{p}(x)$ - log x scale", fontsize="medium")

ax[1, 1].semilogx(xx[iplot_min:iplot_max], test_pdf[iplot_min:iplot_max] / norm)
ax[1, 1].set_title(
    r"normalized pdf $p(x)$ with $\int_0^\infty p(x) dx = 1$ - log x scale",
    fontsize="medium",
)

for axx in ax.ravel():
    axx.grid(ls=":")
    axx.set_xlabel("x [ns]")
fig.show()

# %%
# estimate the min / max delta value to be considered

if delta_max is None:
    delta_max = 100.0 * stddev_pdf / N
if delta_min is None:
    delta_min = 0.001 * stddev_pdf / N

print(f"delta_min: {delta_min:.2E}")
print(f"delta_max: {delta_max:.2E}")

# setup the geometric deltas series
all_possible_deltas: list[float] = np.logspace(
    np.log10(delta_min), np.log10(delta_max), num_possible_deltas
)

upper_int_limit = x_zero + delta_max

# %%
# estimate the Barankin bound for the varinace
bb_vars, chosen_deltas, U = barankin_bound(
    normalized_pdf,
    all_possible_deltas,
    N,
    Jmax,
    upper_int_limit,
    rcond=rcond,
    interactive=interactive,
    show_cond_number=show_cond_number,
)

# %%
# visualize the results

Js = np.arange(1, Jmax + 1)

fig3, ax3 = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
ax3[0].plot(Js, 1000 * np.sqrt(bb_vars), ".-")
ax3[0].set_ylabel("BB std.dev. [ps]")
ax3[1].semilogy(Js, chosen_deltas, ".-")
ax3[2].semilogy(Js, sorted(chosen_deltas), ".-")
for axx in ax3:
    axx.grid(ls=":")
    axx.set_xlabel("J")
for axx in ax3[1:]:
    axx.axhline(delta_min, color="k", ls="--")
    axx.axhline(delta_max, color="k", ls="--")
ax3[0].set_title(f"barankin bound for std.dev. and N={N}")
ax3[1].set_title("chosen deltas")
ax3[2].set_title("sorted chosen deltas")
fig3.show()
