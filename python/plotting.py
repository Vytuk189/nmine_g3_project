"""Domain and saved-primitive plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_domain(domain, area_rule):
    x_faces = domain["x_faces"]
    x_centres = domain["x_centres"]
    n_cells = domain["n_cells"]
    nl = domain["n_ghost_left"]  
    x = np.linspace(x_faces[0], x_faces[-1], 1000)
    area = np.array([area_rule(xi) for xi in x])
    fig, ax = plt.subplots(figsize=(12, 3), facecolor="white")
    ax.plot(x, area/2, color="black")
    ax.plot(x, -area/2, color="black")

    for j, xf in enumerate(x_faces):
        area_face = domain["area_faces"][j]
        color = "red" if j <= nl or j >= nl + n_cells else "blue" 
        ax.plot([xf, xf], [-area_face/2, area_face/2], color=color, linestyle="--", linewidth=0.8)

    for j, xc in enumerate(x_centres):
        ax.text(xc, 0, str(j), ha="center", va="center", fontsize=8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("±A(x)/2 [m²]")
    ax.set_title("Finite-volume mesh including ghost cells")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def simple_plot(primitives, cell_centres, what_primitive_to_plot, time, xlim=None):
    # Accept either one identifier or a list/tuple of two.
    if isinstance(what_primitive_to_plot, str):
        names = [what_primitive_to_plot]
    else:
        names = list(what_primitive_to_plot)

    if len(names) not in (1, 2):
        raise ValueError("Pass one or two primitive identifiers.")

    times = primitives["times"]
    k = np.argmin(np.abs(times - time))

    fig, ax1 = plt.subplots(figsize=(8, 4), facecolor="white")

    name = names[0]
    ax1.plot(
        cell_centres, primitives["primitives"][name][k],
        color="black", label=name
    )
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel(name)
    ax1.set_title(f"t = {times[k]:.6g} s")
    ax1.spines["top"].set_visible(False)
    if xlim is not None:
        ax1.set_xlim(xlim)

    if len(names) == 2:
        ax2 = ax1.twinx()
        name = names[1]

        ax2.plot(
            cell_centres, primitives["primitives"][name][k],
            color="red", label=name
        )
        ax2.set_ylabel(name, color="red")
        ax2.tick_params(axis="y", colors="red")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color("red")
        
    else:
        ax1.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()
