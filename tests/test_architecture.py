"""Architectural guarantees that are enforced statically (cheap to check, easy to regress)."""
from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "fedhealth"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
            mods |= {f"{n.module}.{a.name}" for a in n.names}
    return mods


def test_coordinator_has_no_access_to_hospital_data():
    """The coordinator may talk to hospitals only over HTTP. It must not import hospital code
    (models, predicate runner) nor refer to a hospital database file."""
    for path in (PKG / "coordinator").rglob("*.py"):
        assert not any(m.startswith("fedhealth.hospital") for m in _imports(path)), path.name
        src = path.read_text(encoding="utf-8")
        assert "hospital_db_path" not in src and "hospital_a.db" not in src and ".db_file" not in src, path.name


def test_hospitals_do_not_import_the_coordinator():
    for path in (PKG / "hospital").rglob("*.py"):
        assert not any(m.startswith("fedhealth.coordinator") for m in _imports(path)), path.name


def test_privacy_modules_are_service_independent():
    """Each privacy technique is a self-contained module (explainable in isolation)."""
    for path in (PKG / "privacy").rglob("*.py"):
        mods = _imports(path)
        assert not any(m.startswith(("fedhealth.hospital", "fedhealth.coordinator", "fedhealth.dashboard")) for m in mods), path.name


def test_ground_truth_is_never_read_by_services():
    for pkg in ("hospital", "coordinator"):
        for path in (PKG / pkg).rglob("*.py"):
            assert "ground_truth" not in path.read_text(encoding="utf-8").replace("ground_truth_demo", ""), path.name
