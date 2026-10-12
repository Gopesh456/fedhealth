"""Phase 5a: AES-256-GCM encryption of sensitive columns at rest."""
from __future__ import annotations

import re
import shutil
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from fedhealth.hospital import models as m
from fedhealth.hospital.app import create_hospital_app
from fedhealth.privacy.encryption.aesgcm import (DecryptionError, NullVault, Vault, load_key, load_or_create_key,
                                                 secrets_dir)
from fedhealth.privacy.encryption.column_types import SensitiveStr, attach_vault
from fedhealth.synth.generate import generate

from .helpers import headline_spec


# ============================================================================ the cipher
def test_roundtrip_and_randomised_nonces():
    v = Vault(b"k" * 32)
    for s in ("SYN-123456789", "", "Zoë Müller-Œ 患者", "x" * 5000):
        assert v.decrypt(v.encrypt(s, "A.patients.full_name"), "A.patients.full_name") == s
    cts = {v.encrypt("same plaintext", "ctx") for _ in range(2000)}
    assert len(cts) == 2000, "a (key, nonce) pair must never repeat - identical plaintexts need different ciphertexts"


def test_tampering_wrong_column_and_wrong_key_are_all_detected():
    v = Vault(b"k" * 32)
    token = v.encrypt("SYN-111111111", "A.patients.national_id")
    import base64
    raw = bytearray(base64.b64decode(token[3:]))
    raw[20] ^= 0x01                                                   # flip one bit in the ciphertext
    with pytest.raises(DecryptionError):
        v.decrypt("v1:" + base64.b64encode(bytes(raw)).decode(), "A.patients.national_id")
    with pytest.raises(DecryptionError):
        v.decrypt(token, "A.patients.full_name")                       # same bytes, different column (AAD mismatch)
    with pytest.raises(DecryptionError):
        Vault(b"j" * 32).decrypt(token, "A.patients.national_id")      # different hospital's key
    with pytest.raises(DecryptionError):
        v.decrypt("SYN-111111111", "A.patients.national_id")           # plaintext where ciphertext is required
    with pytest.raises(ValueError):
        Vault(b"short")


def test_ciphertext_does_not_contain_the_plaintext():
    v = Vault(b"k" * 32)
    assert "SYN-111111111" not in v.encrypt("SYN-111111111", "ctx")


# ============================================================================ keys
def test_keys_are_created_once_and_loaded_never_created(tmp_path):
    k1 = load_or_create_key("hospital_z", tmp_path)
    assert len(k1) == 32 and load_or_create_key("hospital_z", tmp_path) == k1 == load_key("hospital_z", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_key("hospital_nope", tmp_path)                           # a service must NOT silently mint a key


def test_key_files_live_outside_the_data_directory(world):
    assert not list(Path(world["dir"]).rglob("*.key"))
    assert list(secrets_dir().glob("hospital_*.key"))


# ============================================================================ the column type
def test_every_sensitive_column_has_its_own_authenticated_context():
    contexts = []
    for base, h in ((m.ABase, "A"), (m.BBase, "B"), (m.CBase, "C")):
        for table in base.metadata.tables.values():
            for col in table.columns:
                if isinstance(col.type, SensitiveStr):
                    assert col.type.context == f"{h}.{table.name}.{col.name}", col
                    contexts.append(col.type.context)
    assert len(contexts) == len(set(contexts)) == 9
    assert {"A.patients.national_id", "B.patients.national_id", "C.patients.national_id"} <= set(contexts)


def test_column_type_fails_closed_without_a_vault(world):
    engine = create_engine(f"sqlite:///{world['dir'] / 'hospital_a.db'}")          # no vault attached
    with Session(engine) as s, pytest.raises(RuntimeError, match="data-encryption key"):
        s.execute(select(m.APatient.national_id).limit(1)).all()
    engine.dispose()


# ============================================================================ the database file on disk
def _file_bytes(world, name):
    return (Path(world["dir"]) / name).read_bytes()


def test_plaintext_identifiers_are_absent_from_the_encrypted_files_but_present_in_the_control(world, plain_world):
    needles_re = re.compile(rb"SYN-\d{9}")
    probes = [b"Counselled on lifestyle changes.", b"left ventricular hypertrophy", b"increased echogenicity",
              b"Dose adjusted at last visit."]
    for name in ("hospital_a.db", "hospital_b.db", "hospital_c.db"):
        enc, plain = _file_bytes(world, name), _file_bytes(plain_world, name)
        assert not needles_re.search(enc), f"{name}: a national ID is readable on disk"
        assert len(needles_re.findall(plain)) > 1000, "control failed: the plaintext DB should expose IDs"
        for probe in probes:
            assert probe not in enc
    names = [line.split(",")[2].encode() for line in (Path(world["gt_dir"]) / "universe.csv").read_text().splitlines()[1:200]]
    a_enc, a_plain = _file_bytes(world, "hospital_a.db"), _file_bytes(plain_world, "hospital_a.db")
    assert not any(n in a_enc for n in names)
    assert sum(n in a_plain for n in names) > 100                       # control: names ARE visible without encryption


def test_only_the_sensitive_columns_are_encrypted(world):
    con = sqlite3.connect(str(world["dir"] / "hospital_a.db"))
    nid, birth_year, sex = con.execute("select national_id, birth_year, sex from patients limit 1").fetchone()
    assert nid.startswith("v1:") and isinstance(birth_year, int) and sex in ("F", "M")
    con.close()


def test_encrypted_and_plaintext_worlds_are_logically_identical(world, plain_world):
    """Same seed -> same data; decrypting the encrypted world gives back exactly the plaintext world."""
    def rows(path, vault, model, cols):
        eng = attach_vault(create_engine(f"sqlite:///{path}"), vault)
        with Session(eng) as s:
            out = sorted(s.execute(select(*[getattr(model, c) for c in cols])).all())
        eng.dispose()
        return out
    v = {h: Vault(load_key(f"hospital_{h}")) for h in "abc"}
    checks = [("hospital_a.db", "a", m.APatient, ["patient_id", "national_id", "full_name", "date_of_birth", "zip_code"]),
              ("hospital_a.db", "a", m.ACondition, ["condition_id", "notes"]),
              ("hospital_b.db", "b", m.BPatient, ["patient_id", "national_id"]),
              ("hospital_b.db", "b", m.BDiagnosticReport, ["report_id", "imaging_finding"]),
              ("hospital_c.db", "c", m.CPatient, ["patient_id", "national_id"]),
              ("hospital_c.db", "c", m.CPrescription, ["rx_id", "notes"])]
    for fname, h, model, cols in checks:
        assert rows(world["dir"] / fname, v[h], model, cols) == rows(plain_world["dir"] / fname, NullVault(), model, cols)


# ============================================================================ tampering and key mix-ups on a copy
@pytest.fixture()
def db_copy(world, tmp_path):
    dst = tmp_path / "hospital_a.db"
    shutil.copy(world["dir"] / "hospital_a.db", dst)
    return dst


def test_swapping_a_ciphertext_into_another_column_is_detected(db_copy):
    con = sqlite3.connect(str(db_copy))
    con.execute("update patients set national_id = full_name where patient_id = (select min(patient_id) from patients)")
    con.commit(); con.close()
    app = create_hospital_app("A", db_path=db_copy, allow_insecure=True)
    with Session(app.state.engine) as s, pytest.raises(DecryptionError):
        s.execute(select(m.APatient.national_id).order_by(m.APatient.patient_id)).all()


def test_modified_ciphertext_is_detected_not_silently_accepted(db_copy):
    con = sqlite3.connect(str(db_copy))
    token = con.execute("select national_id from patients order by patient_id limit 1").fetchone()[0]
    flipped = token[:-3] + ("AAA" if token[-3:] != "AAA" else "BBB")
    con.execute("update patients set national_id = ? where national_id = ?", (flipped, token))
    con.commit(); con.close()
    app = create_hospital_app("A", db_path=db_copy, allow_insecure=True)
    with Session(app.state.engine) as s, pytest.raises(Exception):
        s.execute(select(m.APatient.national_id)).all()


def test_a_hospital_with_the_wrong_key_cannot_read_the_data(world):
    wrong = Vault(load_key("hospital_b"))                               # B's key on A's database
    app = create_hospital_app("A", db_path=world["dir"] / "hospital_a.db", allow_insecure=True, vault=wrong)
    with Session(app.state.engine) as s, pytest.raises(DecryptionError):
        s.execute(select(m.APatient.national_id).limit(1)).all()


def test_service_refuses_to_start_without_its_key(world, tmp_path, monkeypatch):
    monkeypatch.setenv("FEDHEALTH_SECRETS_DIR", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        create_hospital_app("A", db_path=world["dir"] / "hospital_a.db")
    monkeypatch.setenv("ALLOW_PLAINTEXT_DB", "1")                       # the explicit, greppable opt-out (control experiments)
    create_hospital_app("A", db_path=world["dir"] / "hospital_a.db")


def test_queries_work_through_encrypted_columns(hospital_clients, ground_truth):
    """End to end: the baseline filters on plaintext columns and returns IDs it had to DECRYPT."""
    from fedhealth.common.schemas import local_query
    rows = hospital_clients["C"].post("/insecure/match",
                                      json=local_query(headline_spec(), "C").model_dump(mode="json", exclude_none=True)).json()["rows"]
    universe_ids = {p.national_id for p in ground_truth.persons.values()}
    assert rows and all(r["national_id"] in universe_ids for r in rows)
