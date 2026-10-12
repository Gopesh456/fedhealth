"""End-to-end over REAL HTTP: four separate uvicorn processes (3 hospitals + coordinator), each reading
its own database file. Slow (process start-up), so marked `slow`."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

from fedhealth.common.config import load_topology
from fedhealth.synth.generate import generate
from fedhealth.synth.ground_truth import GroundTruth

from .helpers import headline_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_local  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def live_system(tmp_path_factory, monkeypatch_module):
    data = tmp_path_factory.mktemp("e2e_data")
    generate(800, seed=11, out_dir=data)
    monkeypatch_module.setenv("FEDHEALTH_DATA_DIR", str(data))
    monkeypatch_module.setenv("FEDHEALTH_PORT_OFFSET", "300")          # next to, not on top of, a dev system
    procs = run_local.start_services(quiet=True)
    try:
        yield load_topology(), GroundTruth(data / "ground_truth")
    finally:
        run_local.stop_services(procs)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_plain_query_over_real_http_matches_ground_truth(live_system):
    topo, gt = live_system
    base = f"http://127.0.0.1:{topo.coordinator_port}"
    assert httpx.get(f"{base}/health").json()["status"] == "ok"
    spec = headline_spec()
    r = httpx.post(f"{base}/insecure/query", json=spec.model_dump(mode="json", exclude_none=True), timeout=120)
    assert r.status_code == 200
    assert r.json()["result"] == gt.evaluate(spec)


def test_psi_linkage_between_real_processes(live_system):
    """Hospital-to-hospital messages over real sockets: the PSI-linked query equals ground truth, and a
    second query reuses the cached linkage."""
    topo, gt = live_system
    base = f"http://127.0.0.1:{topo.coordinator_port}"
    spec = headline_spec()
    body = spec.model_dump(mode="json", exclude_none=True)
    first = httpx.post(f"{base}/insecure/psi-query", json=body, timeout=300).json()
    again = httpx.post(f"{base}/insecure/psi-query", json=body, timeout=300).json()
    assert first["mode"] == "psi-linked-insecure-aggregation"
    assert first["result"] == again["result"] == gt.evaluate(spec)
    assert first["timings"]["linkage_cached"] == 0.0 and again["timings"]["linkage_cached"] == 1.0
    print(f"\n[real processes] cold PSI linkage {first['timings']['linkage_s']:.2f}s, warm query {again['timings']['total_s']:.3f}s")


def test_secret_shared_query_between_real_processes(live_system):
    """Shares and Beaver openings travel hospital-to-hospital over real sockets; result == ground truth for
    every aggregate type and for 1-, 2- and 3-hospital queries."""
    topo, gt = live_system
    base = f"http://127.0.0.1:{topo.coordinator_port}"
    specs = [headline_spec(),
             headline_spec(agg="histogram", group_by="outcome", b={}),
             headline_spec(agg="avg", value_field="lab_delta", a={}, c={}),
             headline_spec(a=None, b=None),                                  # single hospital (C)
             headline_spec(b=None)]                                          # A + C
    for spec in specs:
        r = httpx.post(f"{base}/insecure/mpc-query", json=spec.model_dump(mode="json", exclude_none=True), timeout=300)
        assert r.status_code == 200, r.text
        got, want = r.json()["result"], gt.evaluate(spec)
        assert got == pytest.approx(want) if isinstance(want, float) else got == want
    print(f"\n[real processes] warm 3-hospital count: {r.json()['timings']}")


def test_strict_mode_has_no_insecure_routes(tmp_path, monkeypatch):
    data = tmp_path / "d"
    generate(200, seed=3, out_dir=data)
    monkeypatch.setenv("FEDHEALTH_DATA_DIR", str(data))
    monkeypatch.setenv("FEDHEALTH_PORT_OFFSET", "400")
    procs = run_local.start_services(strict=True, quiet=True)
    try:
        topo = load_topology()
        assert httpx.post(f"http://127.0.0.1:{topo.coordinator_port}/insecure/query", json={}).status_code == 404
        assert httpx.post(f"http://127.0.0.1:{topo.hospitals['A'].port}/insecure/match", json={}).status_code == 404
    finally:
        run_local.stop_services(procs)
