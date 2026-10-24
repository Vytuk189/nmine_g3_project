import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})

x = np.linspace(-0.5, 0.5, 201)
p_average = 10
p_floor = 5
p_original = p_average + 14*x

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5), sharey=True)

for ax in axes:
    ax.axhspan(0, p_floor, color="#fce9e7")
    ax.axhline(p_floor, color="#c54135", linestyle="--", linewidth=1.5)
    ax.axvline(-0.5, color="#9aa0a6", linewidth=1)
    ax.axvline(0.5, color="#9aa0a6", linewidth=1)
    ax.plot(x, p_original, linestyle="--", color="#a5abb3", linewidth=2)
    ax.scatter([0], [p_average], color="black", s=45, zorder=5)

    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(0, 19)
    ax.set_xticks([-0.5, 0, 0.5], ["Left face", "Centre", "Right face"])
    ax.set_yticks([0, 3, 5, 10, 15, 17])
    ax.set_xlabel("Position within one cell", labelpad=12)
    ax.spines[["top", "right"]].set_visible(False)

# Original reconstruction.
axes[0].plot(x, p_original, color="#3975b9", linewidth=3)
axes[0].scatter([-0.5], [3], color="#c54135", s=70, zorder=6)
axes[0].set_title("Original reconstruction\nFull slope: θ = 1", pad=15, fontweight="bold")
axes[0].set_ylabel("Pressure [kPa]")
axes[0].annotate("3 kPa: below floor", xy=(-0.5, 3), xytext=(-0.32, 1), fontsize=11, color="#a33128", arrowprops={"arrowstyle": "->", "color": "#a33128"})

# Old fallback: remove the entire slope.
axes[1].plot(x, np.full_like(x, p_average), color="#d28322", linewidth=3)
axes[1].set_title("Old fallback\nDiscard slope: θ = 0", pad=15, fontweight="bold")
axes[1].text(0, 12, "Flat profile", ha="center", color="#9b5b12")

# New limiter: retain the largest admissible slope.
theta = 5/7
p_limited = p_average + theta*(p_original - p_average)

axes[2].plot(x, p_limited, color="#18836b", linewidth=3)
axes[2].scatter([-0.5], [p_floor], color="#18836b", s=65, zorder=6)
axes[2].set_title("New limiter\nReduce slope: θ = 5/7", pad=15, fontweight="bold")
axes[2].annotate("5 kPa: meets floor", xy=(-0.5, 5), xytext=(-0.32, 2), fontsize=11, color="#116551", arrowprops={"arrowstyle": "->", "color": "#116551"})

fig.text(0.5, 0.025, "Black dot: unchanged cell-average state.  Dashed red line: 5 kPa floor.\nIllustrative case: only energy varies, so pressure is linear; the code scales all conserved-variable slopes.", ha="center", fontsize=11, color="#41454b")

fig.tight_layout(rect=[0, 0.15, 1, 1])
fig.savefig("muscl_slope_comparison.png", dpi=300, facecolor="white")
plt.show()