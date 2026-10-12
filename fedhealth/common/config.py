"""Loading of the YAML configs and resolution of paths.

Paths can be redirected with environment variables so tests and containers never touch the
repo's own `data/` directory:
    FEDHEALTH_ROOT        project root          (default: two levels above this file)
    FEDHEALTH_CONFIG_DIR  directory of *.yaml   (default: <root>/configs)
    FEDHEALTH_DATA_DIR    where *.db live       (default: <root>/data)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


def repo_root() -> Path:
    env = os.environ.get("FEDHEALTH_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    env = os.environ.get("FEDHEALTH_CONFIG_DIR")
    return Path(env) if env else repo_root() / "configs"


def data_dir() -> Path:
    env = os.environ.get("FEDHEALTH_DATA_DIR")
    return Path(env) if env else repo_root() / "data"


def _load_yaml(name: str) -> dict:
    with open(config_dir() / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- topology
@dataclass(frozen=True)
class HospitalEntry:
    hospital_id: str
    name: str
    holds: str
    host: str
    port: int
    db_file: str

    def url(self, scheme: str) -> str:
        return f"{scheme}://{self.host}:{self.port}"


@dataclass(frozen=True)
class Topology:
    scheme: str
    ring: tuple[str, ...]
    hospitals: dict[str, HospitalEntry]
    coordinator_host: str
    coordinator_port: int
    coordinator_db: str

    def hospital_url(self, hospital_id: str) -> str:
        return self.hospitals[hospital_id].url(self.scheme)

    def hospital_db_path(self, hospital_id: str) -> Path:
        return data_dir() / self.hospitals[hospital_id].db_file

    @property
    def coordinator_db_path(self) -> Path:
        return data_dir() / self.coordinator_db


def load_topology() -> Topology:
    raw = _load_yaml("topology.yaml")
    # FEDHEALTH_PORT_OFFSET shifts every port (lets tests run next to a dev system on the default ports)
    offset = int(os.environ.get("FEDHEALTH_PORT_OFFSET", "0"))
    hospitals = {}
    for hid, h in raw["hospitals"].items():
        # FEDHEALTH_HOST_A=hospital-a lets Docker/Compose use service names instead of 127.0.0.1
        host = os.environ.get(f"FEDHEALTH_HOST_{hid}", h["host"])
        hospitals[hid] = HospitalEntry(hid, h["name"], h["holds"], host, int(h["port"]) + offset, h["db"])
    coord = raw["coordinator"]
    return Topology(
        scheme=os.environ.get("FEDHEALTH_SCHEME", raw.get("scheme", "http")),
        ring=tuple(raw["ring"]),
        hospitals=hospitals,
        coordinator_host=os.environ.get("FEDHEALTH_HOST_COORDINATOR", coord["host"]),
        coordinator_port=int(coord["port"]) + offset,
        coordinator_db=coord["db"],
    )


# --------------------------------------------------------------------------- field whitelist
@dataclass(frozen=True)
class FieldSpec:
    name: str
    owner: str
    scale: int = 1
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class Fields:
    value_fields: dict[str, FieldSpec]
    group_by: dict[str, FieldSpec]


@lru_cache(maxsize=4)
def _load_fields_cached(path_key: str) -> Fields:
    raw = _load_yaml("fields.yaml")
    vf = {n: FieldSpec(n, v["owner"], int(v.get("scale", 1)), v.get("min"), v.get("max"))
          for n, v in raw["value_fields"].items()}
    gb = {n: FieldSpec(n, v["owner"]) for n, v in raw["group_by"].items()}
    return Fields(vf, gb)


def load_fields() -> Fields:
    return _load_fields_cached(str(config_dir()))
