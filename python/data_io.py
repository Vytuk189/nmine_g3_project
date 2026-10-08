"""Save physical-cell primitives and load saved results."""

import numpy as np
from pathlib import Path


def save_primitives(history, domain, folder_name):
    """Save primitive history for physical cells into data/folder_name."""
    data_folder = Path.cwd() / "data"
    data_folder.mkdir(exist_ok=True)

    folder = data_folder / folder_name
    counter = 1

    while folder.exists():
        folder = data_folder / f"{folder_name}_{counter}"
        counter += 1

    folder.mkdir()

    physical = domain["physical_ind"]
    x_centres = domain["x_centres"][physical]

    np.savetxt(folder / "times.txt", history["times"], fmt="%.18e")
    np.savetxt(folder / "x_centres.txt", x_centres, fmt="%.18e")

    for name, values in history["primitives"].items():
        values = np.asarray(values)
        np.savetxt(
            folder / f"{name}.txt",
            values[:, physical],
            fmt="%.18e"
        )

    print(f"Saved results to: {folder}")
    return folder


def load_primitives(folder_name):
    """Load from a folder name inside data, or from a folder path."""
    folder = Path(folder_name).expanduser()

    # A bare name refers to a folder inside the current data folder.
    if not folder.is_absolute() and folder.parent == Path("."):
        folder = Path.cwd() / "data" / folder

    x_centres = np.loadtxt(folder / "x_centres.txt", ndmin=1)
    times = np.loadtxt(folder / "times.txt", ndmin=1)

    history = {
        "times": times,
        "primitives": {}
    }

    for file in sorted(folder.glob("*.txt")):
        if file.name in ("times.txt", "x_centres.txt"):
            continue

        values = np.loadtxt(file, ndmin=2)

        if values.shape != (len(times), len(x_centres)):
            raise ValueError(f"Unexpected data shape in {file.name}.")

        legacy_names = {"Tg": "t_g", "Tl": "t_l", "ag": "alpha_g", "al": "alpha_l", "E": "e_total", "K": "k"}
        name = legacy_names.get(file.stem, file.stem)
        history["primitives"][name] = values

    return x_centres, history
