# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.integrate import simpson
from utils import barankin_bound_from_pdf_lut


# %%
# input parameters

# file name of the lookup table containing the pdf
# the file should contains two columns, t and pdf(t), separated by whitespace
# the pdf does not need to be normalized, but t should equally spaced and increasing
pdf_lut_fname: str = "example_pdf.txt"
# number of possible deltas
num_possible_deltas: int = 128
# minimum delta to consider, None mean auto determined
delta_min: float | None = None
# maximum delta to consider, None mean auto determined
delta_max: float | None = None
# number of photons / samples
N: int = 100
# maximum J value
Jmax: int = 32
# point beyond which pdf is essentially zero, None means auto determined
t_upper: float | None = None
# fraction of largest singular value for calculate of pseudo inverse
rcond: float = 1e-12
# show interactive plots on how J values are chose, requires user interaction
interactive: bool = False
# show condition number of U matrix
show_cond_number: bool = False


# %%
# read the pdf from a file
pdf_lut = np.loadtxt(pdf_lut_fname)

# the points beyond which the pdf is 0
t_upper = pdf_lut[-1, 0]

# the max value of the pdf
i_max = np.argmax(pdf_lut[:, 1])
t_max = pdf_lut[i_max, 0]
pdf_max = pdf_lut[i_max, 1]

i_small = np.where(pdf_lut[:, 1] > (1e-3) * t_max)[0].max()
t_small = pdf_lut[i_small, 0]

dt = pdf_lut[1, 0] - pdf_lut[0, 0]

# %%
# normalize the pdf lut
norm = simpson(pdf_lut[:, 1], dx=dt)
normalized_pdf_lut = pdf_lut.copy()
normalized_pdf_lut[:, 1] /= norm

# %%
# plot the input pdf
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True, sharex=True)
ax[0].plot(pdf_lut[:, 0], pdf_lut[:, 1], ".-")
ax[0].set_title(r"input pdf $\tilde{p}(x)$")
ax[0].grid(ls=":")
ax[0].set_xlabel("x")
ax[1].plot(normalized_pdf_lut[:, 0], normalized_pdf_lut[:, 1], ".-")
ax[1].set_title(r"normalized pdf $p(x)$ with $\int_0^\infty p(x) dx = 1$")
ax[1].grid(ls=":")
ax[1].set_xlabel("x")
fig.show()

# %%
# estimate the min / max delta value to be considered

stddev_pdf = np.sqrt(
    simpson(normalized_pdf_lut[:, 0] ** 2 * normalized_pdf_lut[:, 1], dx=dt)
    - simpson(normalized_pdf_lut[:, 0] * normalized_pdf_lut[:, 1], dx=dt) ** 2
)

if delta_max is None:
    delta_max = min(100.0 * stddev_pdf / N, t_upper - t_small)
if delta_min is None:
    delta_min = 0.01 * stddev_pdf / N

print(f"delta_min: {delta_min:.2E}")
print(f"delta_max: {delta_max:.2E}")

if delta_min / dt < 0.3:
    warnings.warn(
        "delta_min smaller than 0.3 times dt of pdf LUT. Use finer sampling in pdf LUT."
    )

# setup the geometric deltas series
all_possible_deltas: list[float] = np.logspace(
    np.log10(delta_min), np.log10(delta_max), num_possible_deltas
)

# %%
bbs, chosen_deltas, U = barankin_bound_from_pdf_lut(
    normalized_pdf_lut, all_possible_deltas, N, Jmax, rcond=rcond, verbose=True
)

# %%
# visualize the results

Js = np.arange(1, Jmax + 1)

fig3, ax3 = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
ax3[0].plot(Js, np.sqrt(bbs), ".-")
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
