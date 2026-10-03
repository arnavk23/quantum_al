"""Loads Materials Project JSON files (see fetch_data.py) into (X, y, meta)
arrays. 21 feature columns: 6 structural/summary descriptors + 15
composition-derived stats. crystal_system is metadata only for regression
tasks (it's the target for the classification task).

The data directory is resolved by :func:`get_data_dir`: the
``QUANTUM_AL_DATA_DIR`` environment variable if set, otherwise ``data/``
at the root of a source checkout, otherwise ``./data``.
"""
import json
import os

import numpy as np

# <repo_root>/src/quantum_al/data_utils.py -> up three levels to <repo_root>
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "data")


def get_data_dir():
    """Directory holding the ``<task>.json`` files."""
    env = os.environ.get("QUANTUM_AL_DATA_DIR")
    if env:
        return env
    if os.path.isdir(DATA_DIR):
        return DATA_DIR
    return os.path.abspath("data")


def _read_task_file(name, data_dir=None):
    path = os.path.join(data_dir or get_data_dir(), f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Fetch the real Materials Project data first with "
            "`MP_API_KEY=... python -m quantum_al.fetch_data` (needs the [data] extra), "
            "or point QUANTUM_AL_DATA_DIR at an existing copy."
        )
    with open(path, "r") as f:
        return json.load(f)

FEATURE_COLUMNS = [
    "nelements", "density", "volume_per_atom", "nsites", "energy_above_hull",
    "space_group_number",
    "X_mean", "X_std", "X_range",
    "atomic_radius_mean", "atomic_radius_std", "atomic_radius_range",
    "atomic_mass_mean", "atomic_mass_std", "atomic_mass_range",
    "row_mean", "row_std", "row_range",
    "group_mean", "group_std", "group_range",
]

REGRESSION_TASKS = [
    "band_gap", "formation_energy", "bulk_modulus",
    "magnetic_moment", "dielectric_constant",
]
CLASSIFICATION_TASK = "crystal_system"


def load_task(name, drop_na=True, data_dir=None):
    """Load one task.

    Parameters
    ----------
    name : str
        One of :data:`REGRESSION_TASKS` or :data:`CLASSIFICATION_TASK`.
    drop_na : bool
        Drop rows with a missing feature or target.
    data_dir : str, optional
        Overrides :func:`get_data_dir`.

    Returns
    -------
    X : ndarray, shape (n, 21)
        Features in :data:`FEATURE_COLUMNS` order.
    y : ndarray, shape (n,)
        Float targets (string labels for ``crystal_system``).
    meta : list of dict
        ``material_id``, ``formula_pretty`` and ``crystal_system`` per row.
    """
    rows = _read_task_file(name, data_dir)

    X_list, y_list, meta = [], [], []
    for row in rows:
        if drop_na:
            if any(row.get(c) is None for c in FEATURE_COLUMNS):
                continue
            if row.get("target") is None:
                continue
        feat = [row.get(c, np.nan) for c in FEATURE_COLUMNS]
        X_list.append(feat)
        y_list.append(row["target"])
        meta.append({
            "material_id": row.get("material_id"),
            "formula_pretty": row.get("formula_pretty"),
            "crystal_system": row.get("crystal_system"),
        })

    X = np.asarray(X_list, dtype=float)
    if name == CLASSIFICATION_TASK:
        y = np.asarray(y_list)
    else:
        y = np.asarray(y_list, dtype=float)
    return X, y, meta


def load_multi_task(names, drop_na=True, data_dir=None):
    """Load several regression tasks inner-joined on ``material_id``.

    Every returned row has a real, independently computed label for all
    ``K = len(names)`` tasks, so one acquisition reveals all of them (as a
    single DFT calculation does). Feature columns are identical across
    task files for a shared ``material_id``; rows are sorted by it.

    Returns
    -------
    X : ndarray, shape (n, 21)
    Y : ndarray, shape (n, K)
    meta : list of dict
    """
    per_task = {}
    for name in names:
        rows = _read_task_file(name, data_dir)
        per_task[name] = {
            r["material_id"]: r for r in rows
            if r.get("target") is not None
            and not (drop_na and any(r.get(c) is None for c in FEATURE_COLUMNS))
        }

    common_ids = set.intersection(*(set(d) for d in per_task.values()))
    common_ids = sorted(common_ids)

    X_list, Y_list, meta = [], [], []
    for mid in common_ids:
        row0 = per_task[names[0]][mid]
        X_list.append([row0.get(c, np.nan) for c in FEATURE_COLUMNS])
        Y_list.append([per_task[n][mid]["target"] for n in names])
        meta.append({
            "material_id": mid,
            "formula_pretty": row0.get("formula_pretty"),
            "crystal_system": row0.get("crystal_system"),
        })

    X = np.asarray(X_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    return X, Y, meta


def standardize(X_train, *others):
    """Fit per-column mean/std on ``X_train`` and apply it to ``X_train``
    and any further arrays (e.g. a test set), avoiding test-set leakage.
    Constant columns are left centered but unscaled."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    out = [(X_train - mu) / sigma]
    for X in others:
        out.append((X - mu) / sigma)
    return tuple(out) if len(out) > 1 else out[0]


if __name__ == "__main__":
    for t in REGRESSION_TASKS:
        X, y, meta = load_task(t)
        print(t, X.shape, y.shape, "target range:", y.min(), y.max())
    X, y, meta = load_task(CLASSIFICATION_TASK)
    print(CLASSIFICATION_TASK, X.shape, y.shape, "classes:", sorted(set(y.tolist())))
