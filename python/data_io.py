"""Save physical-cell primitives and optional Rez diagnostics."""

import numpy as np
from pathlib import Path


def create_results_folder(domain, folder_name):
    """Create a unique folder under data and save the domain."""
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


def save_snapshot(primitives, time, domain, folder, iteration, rez=None):
    """Save one snapshot of physical-cell primitives and optional five-value Rez.

    Write a temporary file first, then rename it after writing finishes.
    Previously completed snapshots remain available if the run is interrupted.
    """
    folder = Path(folder)
    physical = domain["physical_ind"]

    values = {"time": np.array(time)}

    if rez is not None:
        values["rez"] = np.asarray(rez)

    for name, array in primitives.items():
        values[name] = np.asarray(array)[physical]

    destination = folder / "snapshots" / f"step_{iteration:012d}.npz"
    temporary = destination.with_suffix(".tmp")

    with temporary.open("wb") as file:
        np.savez(file, **values)

    temporary.replace(destination)


def save_primitives(history, domain, folder_name):
    """Save a complete in-memory history using the original text-file format.

    Input primitive arrays include ghost cells.
    Each output row represents a saved time; each column a physical cell.
    Optional Rez and implicit diagnostics are saved separately as NPZ files.
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
        np.savetxt(folder / f"{name}.txt", values[:, physical], fmt="%.18e")

    if "diagnostics" in history:
        np.savez(folder / "diagnostics.npz", **history["diagnostics"])

    if "rez" in history:
        np.savez(folder / "rez.npz", values=history["rez"])

    print(f"Saved results to: {folder}")
    return folder


def load_primitives(folder_name):
    """Load snapshots or the original text-file format.

    A bare folder name is resolved under ./data.
    Return physical cell centres and history.
    When available, history["rez"] has shape (saved_times, 5).
    """
    folder = Path(folder_name).expanduser()

    if not folder.is_absolute() and folder.parent == Path("."):
        folder = Path.cwd() / "data" / folder

    snapshot_folder = folder / "snapshots"

    # INCREMENTAL SNAPSHOTS
    if snapshot_folder.exists():
        files = sorted(snapshot_folder.glob("step_*.npz"))

        if not files:
            raise ValueError(f"No completed snapshots found in {folder}.")

        x_centres = np.loadtxt(folder / "x_centres.txt", ndmin=1)
        history = {"times": [], "primitives": {}}
        rez_values = []
        has_rez = False

        for file in files:
            with np.load(file) as snapshot:
                history["times"].append(float(snapshot["time"]))

                if "rez" in snapshot.files:
                    rez_values.append(snapshot["rez"].copy())
                    has_rez = True
                else:
                    rez_values.append(np.full(5, np.nan))

                for name in snapshot.files:
                    # Metadata must not be interpreted as a primitive field.
                    if name in ("time", "rez"):
                        continue

                    if name not in history["primitives"]:
                        history["primitives"][name] = []

                    history["primitives"][name].append(snapshot[name].copy())

        history["times"] = np.array(history["times"])

        for name in history["primitives"]:
            history["primitives"][name] = np.stack(history["primitives"][name])

        if has_rez:
            history["rez"] = np.stack(rez_values)

        return x_centres, history

    # ORIGINAL TEXT-FILE FORMAT
    x_centres = np.loadtxt(folder / "x_centres.txt", ndmin=1)
    times = np.loadtxt(folder / "times.txt", ndmin=1)

    history = {"times": times, "primitives": {}}

    legacy_names = {
        "Tg": "t_g", "Tl": "t_l", "ag": "alpha_g", "al": "alpha_l",
        "E": "e_total", "K": "k"
    }

    for file in sorted(folder.glob("*.txt")):
        if file.name in ("times.txt", "x_centres.txt", "pressure_schedule.txt"):
            continue

        values = np.loadtxt(file, ndmin=2)

        if values.shape != (len(times), len(x_centres)):
            raise ValueError(f"Unexpected data shape in {file.name}.")

        name = legacy_names.get(file.stem, file.stem)
        history["primitives"][name] = values

    if (folder / "diagnostics.npz").exists():
        with np.load(folder / "diagnostics.npz") as saved:
            history["diagnostics"] = {name: saved[name].copy() for name in saved.files}

    if (folder / "rez.npz").exists():
        with np.load(folder / "rez.npz") as saved:
            history["rez"] = saved["values"].copy()

    return x_centres, history