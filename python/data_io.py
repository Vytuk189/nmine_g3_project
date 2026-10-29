"""Save physical-cell primitives and load saved results."""

import numpy as np
from pathlib import Path

def create_results_folder(domain, folder_name):
    """Create a unique results folder and save the domain."""
    data_folder = Path.cwd() / "data"
    data_folder.mkdir(exist_ok=True)

    folder = data_folder / folder_name
    counter = 1

    while folder.exists():
        folder = data_folder / f"{folder_name}_{counter}"
        counter += 1

    folder.mkdir()
    (folder / "snapshots").mkdir()

    physical = domain["physical_ind"]
    np.savetxt(folder / "x_centres.txt", domain["x_centres"][physical], fmt="%.18e")
    np.savez(folder / "domain.npz", **domain)

    print(f"Saving results to: {folder}")
    return folder


def save_snapshot(primitives, time, domain, folder, iteration):
    """Save one complete snapshot, containing physical cells only.

    A temporary file is renamed after writing, so an interrupted write
    does not replace or damage previously completed snapshots.
    """
    folder = Path(folder)
    physical = domain["physical_ind"]

    values = {"time": np.array(time)}

    for name, array in primitives.items():
        values[name] = np.asarray(array)[physical]

    destination = folder / "snapshots" / f"step_{iteration:012d}.npz"
    temporary = destination.with_suffix(".tmp")

    with temporary.open("wb") as file:
        np.savez(file, **values)

    temporary.replace(destination)


def save_primitives(history, domain, folder_name):
    """Save full-grid history using a bare folder_name under ./data.

    history["primitives"][name] has shape (saved_times, n_total), including ghosts;
    domain supplies physical indices. Each TXT row is a time, each column a cell.
    Optional implicit-step diagnostics are saved separately in diagnostics.npz.
    Existing folders get a numeric suffix. Return the actual folder Path.
    """
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

    if "diagnostics" in history:
        np.savez(folder / "diagnostics.npz", **history["diagnostics"])

    print(f"Saved results to: {folder}")
    return folder


def load_primitives(folder_name):
    """Load an existing bare name under ./data, or an explicit folder path.

    Return (physical x_centres, history). Diagnostics, when present, have their
    own time array: one row per accepted step, rather than per saved snapshot.
    Old folders without diagnostics remain readable.
    """
    folder = Path(folder_name).expanduser()

    # A bare name refers to a folder inside the current data folder.
    if not folder.is_absolute() and folder.parent == Path("."):
        folder = Path.cwd() / "data" / folder
        
    # Incrementally saved runs.
    snapshot_folder = folder / "snapshots"

    if snapshot_folder.exists():
        files = sorted(snapshot_folder.glob("step_*.npz"))

        if not files:
            raise ValueError(f"No completed snapshots found in {folder}.")

        x_centres = np.loadtxt(folder / "x_centres.txt", ndmin=1)
        history = {"times": [], "primitives": {}}

        for file in files:
            with np.load(file) as snapshot:
                history["times"].append(float(snapshot["time"]))

                for name in snapshot.files:
                    if name == "time":
                        continue

                    if name not in history["primitives"]:
                        history["primitives"][name] = []

                    history["primitives"][name].append(snapshot[name].copy())

        history["times"] = np.array(history["times"])

        for name in history["primitives"]:
            history["primitives"][name] = np.stack(history["primitives"][name])

        return x_centres, history

    x_centres = np.loadtxt(folder / "x_centres.txt", ndmin=1)
    times = np.loadtxt(folder / "times.txt", ndmin=1)

    history = {
        "times": times,
        "primitives": {}
    }

    for file in sorted(folder.glob("*.txt")):
        if file.name in ("times.txt", "x_centres.txt", "pressure_schedule.txt"):
            continue

        values = np.loadtxt(file, ndmin=2)

        # Keep this check: it detects damaged/inconsistent saved numerical data.
        if values.shape != (len(times), len(x_centres)):
            raise ValueError(f"Unexpected data shape in {file.name}.")

        legacy_names = {"Tg": "t_g", "Tl": "t_l", "ag": "alpha_g", "al": "alpha_l", "E": "e_total", "K": "k"}
        name = legacy_names.get(file.stem, file.stem)
        history["primitives"][name] = values

    if (folder / "diagnostics.npz").exists():
        with np.load(folder / "diagnostics.npz") as saved:
            history["diagnostics"] = {name: saved[name].copy() for name in saved.files}

    return x_centres, history
