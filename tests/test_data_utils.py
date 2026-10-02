"""Tests for quantum_al.data_utils on small hand-written task files, and
for quantum_al.fetch_data's API-key handling (no network access)."""
import json

import numpy as np
import pytest

from quantum_al import data_utils
from quantum_al.data_utils import FEATURE_COLUMNS, load_multi_task, load_task, standardize


def _row(mid, target, offset=0.0, **overrides):
    row = {c: float(i) + offset for i, c in enumerate(FEATURE_COLUMNS)}
    row.update(material_id=mid, formula_pretty=f"F{mid}", crystal_system="cubic", target=target)
    row.update(overrides)
    return row


@pytest.fixture
def data_dir(tmp_path):
    tasks = {
        "band_gap": [_row("mp-1", 1.0), _row("mp-2", 2.0, 1.0), _row("mp-3", None),
                     _row("mp-4", 4.0, density=None)],
        "formation_energy": [_row("mp-2", -2.0, 1.0), _row("mp-1", -1.0), _row("mp-9", -9.0)],
    }
    for name, rows in tasks.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(rows))
    return tmp_path


def test_load_task_drops_missing(data_dir):
    X, y, meta = load_task("band_gap", data_dir=str(data_dir))
    assert X.shape == (2, len(FEATURE_COLUMNS))
    assert y.tolist() == [1.0, 2.0]
    assert [m["material_id"] for m in meta] == ["mp-1", "mp-2"]


def test_load_multi_task_inner_joins_on_material_id(data_dir):
    X, Y, meta = load_multi_task(["band_gap", "formation_energy"], data_dir=str(data_dir))
    assert [m["material_id"] for m in meta] == ["mp-1", "mp-2"]
    assert Y.tolist() == [[1.0, -1.0], [2.0, -2.0]]
    assert np.allclose(X[1] - X[0], 1.0)


def test_env_var_selects_data_dir(data_dir, monkeypatch):
    monkeypatch.setenv("QUANTUM_AL_DATA_DIR", str(data_dir))
    assert data_utils.get_data_dir() == str(data_dir)
    X, _, _ = load_task("band_gap")
    assert len(X) == 2


def test_missing_file_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data"):
        load_task("band_gap", data_dir=str(tmp_path))


def test_standardize_uses_training_statistics():
    rng = np.random.default_rng(0)
    X_train = rng.normal(3.0, 2.0, size=(50, 3))
    X_train[:, 2] = 7.0  # constant column
    X_test = rng.normal(size=(10, 3))
    Z_train, Z_test = standardize(X_train, X_test)
    assert np.allclose(Z_train[:, :2].mean(axis=0), 0) and np.allclose(Z_train[:, :2].std(axis=0), 1)
    assert np.allclose(Z_train[:, 2], 0)
    mu, sd = X_train[:, 0].mean(), X_train[:, 0].std()
    assert np.allclose(Z_test[:, 0], (X_test[:, 0] - mu) / sd)


def test_fetch_data_imports_without_api_key(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    from quantum_al import fetch_data

    with pytest.raises(RuntimeError, match="MP_API_KEY"):
        fetch_data._headers()
    assert fetch_data.dotted_get({"a": {"b": 3}}, "a.b") == 3
    assert fetch_data.dotted_get({"a": None}, "a.b") is None
