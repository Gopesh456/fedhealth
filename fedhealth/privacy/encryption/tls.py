"""A tiny private certificate authority for the local / Compose deployment (self-signed, as the brief allows).

WHAT mTLS GIVES US
    Every service connection uses TLS (confidential + tamper-proof on the wire), and the HOSPITALS demand a
    CLIENT certificate signed by our CA ("mutual" TLS): a process without a CA-issued certificate cannot even
    complete the handshake with a hospital. Requesters talk to the coordinator over server-authenticated TLS and
    prove who they are with a JWT instead.

WHAT WE GENERATE (all ECDSA P-256 - small, fast, modern)
    ca.pem                       the CA certificate (the trust anchor every service pins)
    <service>.crt / <service>.key  one per service: coordinator, hospital-a, hospital-b, hospital-c
    Each certificate carries serverAuth + clientAuth usage and SANs for the service name, localhost and 127.0.0.1.

LIMITS (THREAT_MODEL.md): keys are unencrypted PEM files, the CA key sits next to them, there is no revocation
and no rotation. A production deployment would use a managed PKI.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

COORDINATOR = "coordinator"
SERVICES = (COORDINATOR, "hospital-a", "hospital-b", "hospital-c")


def hospital_service_id(hospital_id: str) -> str:
    return f"hospital-{hospital_id.lower()}"


def tls_dir() -> Path:
    import os
    from fedhealth.privacy.encryption.aesgcm import secrets_dir
    env = os.environ.get("FEDHEALTH_TLS_DIR")
    return Path(env) if env else secrets_dir() / "tls"


@dataclass(frozen=True)
class TlsPaths:
    directory: Path
    service_id: str

    @property
    def ca(self) -> Path:
        return self.directory / "ca.pem"

    @property
    def cert(self) -> Path:
        return self.directory / f"{self.service_id}.crt"

    @property
    def key(self) -> Path:
        return self.directory / f"{self.service_id}.key"


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "fedhealth (synthetic demo)"),
                      x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _pem_key(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def generate_ca(directory: Path, days: int = 365 * 5) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    directory.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(_name("fedhealth demo CA")).issuer_name(_name("fedhealth demo CA"))
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)   # strict X.509 mode needs SKI/AKI
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                         key_encipherment=False, data_encipherment=False, key_agreement=False,
                                         encipher_only=False, decipher_only=False), critical=True)
            .sign(key, hashes.SHA256()))
    (directory / "ca.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (directory / "ca.key").write_bytes(_pem_key(key))
    return cert, key


def issue_service_cert(directory: Path, service_id: str, ca_cert: x509.Certificate, ca_key: ec.EllipticCurvePrivateKey,
                       extra_dns: tuple[str, ...] = (), days: int = 365) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.timezone.utc)
    sans = [x509.DNSName(n) for n in dict.fromkeys((service_id, "localhost", *extra_dns))]
    sans.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
    cert = (x509.CertificateBuilder().subject_name(_name(service_id)).issuer_name(ca_cert.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()), critical=False)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=False, crl_sign=False, content_commitment=False,
                                         key_encipherment=False, data_encipherment=False, key_agreement=True,
                                         encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .sign(ca_key, hashes.SHA256()))
    (directory / f"{service_id}.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (directory / f"{service_id}.key").write_bytes(_pem_key(key))


def generate_all(directory: Path | None = None, extra_dns: tuple[str, ...] = ()) -> Path:
    """Create the CA and a certificate for every service. Existing files are overwritten."""
    directory = directory or tls_dir()
    ca_cert, ca_key = generate_ca(directory)
    for svc in SERVICES:
        issue_service_cert(directory, svc, ca_cert, ca_key, extra_dns=extra_dns)
    for f in directory.glob("*.key"):
        try:
            f.chmod(0o600)
        except OSError:
            pass
    return directory


def load_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())
