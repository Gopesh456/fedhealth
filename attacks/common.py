"""Shared helpers for the attack demos and the evaluation scripts.

HOW THE ATTACKS RUN
    The attacker's only interface is the system's own query path. For every DISTINCT query an attack needs, the
    full pipeline (PSI linkage -> secret-shared MPC) is run once WITHOUT noise to get the exact aggregate - the
    evaluation-only exact route, which hospitals accept only with the insecure baseline enabled - and it is checked
    against the independent ground truth. The privacy-relevant part, `release()` (three hospitals' noise shares drawn
    with the hospitals' own exact samplers, k-threshold on the noisy count), is then applied with FRESH randomness for
    every simulated query: that is exactly what repeating the query against the live service does, because the
    MPC's exact part is deterministic and only the noise shares are re-drawn.
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedhealth.common.config import repo_root  # noqa: E402
from fedhealth.common.schemas import QuerySpec  # noqa: E402
from fedhealth.coordinator.bootstrap import bootstrap_secrets  # noqa: E402
from fedhealth.coordinator.release import Exact, exact_from_cells  # noqa: E402
from fedhealth.inproc import build_system  # noqa: E402
from fedhealth.synth.generate import generate  # noqa: E402
from fedhealth.synth.ground_truth import GroundTruth  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"
CHARTS_DIR = ROOT / "evaluation" / "charts"


SOURCES = ("synthea", "synthetic")


def world_dir(n: int, seed: int, source: str = "synthea") -> Path:
    return ROOT / "evaluation" / ".world" / (f"n{n}_s{seed}" if source == "synthetic" else f"{source}_n{n}_s{seed}")


def ensure_world(n: int = 5000, seed: int = 42, source: str = "synthea") -> Path:
    """Generate (once) the world used by the evaluation: n Synthea people (from the shipped extract in ./synthea) or
    n people from the parametric generator. Keys come from the secrets directory."""
    d = world_dir(n, seed, source)
    # regenerate if missing OR stale (a cache from before the federated-learning registries existed)
    if not (d / "hospital_c.db").exists() or not (d / "ground_truth" / "registry_C.csv").exists():
        bootstrap_secrets()
        d.mkdir(parents=True, exist_ok=True)
        generate(n, seed, d, source=source)
    return d


class ExactOracle:
    """Exact aggregates through the REAL private pipeline (PSI + MPC), cached per distinct query."""

    def __init__(self, world: Path, psi_group: str = "modp2048"):
        self.system = build_system(world, psi_group=psi_group, psi_groups=(psi_group,), allow_insecure=True)
        self.orchestrator = self.system.orchestrator
        self.gt = GroundTruth(world / "ground_truth")
        self._cache: dict[str, Exact] = {}
        self.pipeline_runs = 0

    def exact(self, spec: QuerySpec) -> Exact:
        key = spec.model_dump_json()
        if key not in self._cache:
            run = self.orchestrator.mpc_run(spec, None)                  # exact route (no noise plan)
            self.pipeline_runs += 1
            ex = exact_from_cells(spec, run.cells, run.categories)
            if spec.agg == "count":                       # the private pipeline must agree with the independent truth
                assert ex.count == self.gt.evaluate(spec), "pipeline disagrees with ground truth"
            self._cache[key] = ex
        return self._cache[key]


def seeded_randbelow(seed: int):
    return random.Random(seed).randrange


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def save_json(name: str, obj) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def load_json(name: str):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def common_args(description: str):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--patients", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", choices=SOURCES, default="synthea",
                    help="synthea (default: the shipped Synthea extract) or the parametric synthetic generator")
    ap.add_argument("--fast", action="store_true",
                    help="use the INSECURE 768-bit PSI group (speed only) and 1,000 patients")
    return ap


def oracle_from_args(args) -> ExactOracle:
    n = 1000 if args.fast else args.patients
    return ExactOracle(ensure_world(n, args.seed, args.source), psi_group="test768" if args.fast else "modp2048")
