from __future__ import annotations

import ipaddress
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def discover_tls_hosts(bind_host: str = "") -> list[str]:
    candidates = ["localhost", "127.0.0.1", "::1"]
    hostname = socket.gethostname().strip()
    fqdn = socket.getfqdn().strip()
    candidates.extend([hostname, fqdn])
    if bind_host.strip() not in {"", "0.0.0.0", "::"}:
        candidates.append(bind_host.strip())
    for name in {hostname, fqdn} - {""}:
        try:
            candidates.extend(item[4][0] for item in socket.getaddrinfo(name, None))
        except OSError:
            continue
    return normalize_tls_hosts(candidates)


def normalize_tls_hosts(hosts: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in hosts:
        value = str(raw).strip().strip("[]")
        if not value or value in {"0.0.0.0", "::"}:
            continue
        key = value.casefold()
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _general_names(hosts: Iterable[str]) -> list[x509.GeneralName]:
    names: list[x509.GeneralName] = []
    for host in normalize_tls_hosts(hosts):
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    return names


def _certificate_is_usable(cert_path: Path, key_path: Path,
                           required_hosts: Iterable[str]) -> bool:
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if cert.not_valid_after_utc <= datetime.now(timezone.utc) + timedelta(days=1):
            return False
        if cert.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ) != key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ):
            return False
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        existing = {
            str(item.value).casefold()
            for item in san
            if isinstance(item, (x509.DNSName, x509.IPAddress))
        }
        return all(host.casefold() in existing for host in normalize_tls_hosts(required_hosts))
    except (OSError, ValueError, TypeError, x509.ExtensionNotFound):
        return False


def ensure_self_signed_certificate(cert_path: Path, key_path: Path,
                                   hosts: Iterable[str], valid_days: int = 825) -> tuple[Path, Path]:
    cert_path = cert_path.expanduser().resolve()
    key_path = key_path.expanduser().resolve()
    normalized_hosts = normalize_tls_hosts(hosts)
    if not normalized_hosts:
        raise ValueError("at least one TLS hostname or IP address is required")
    if cert_path.exists() != key_path.exists():
        raise ValueError("auto HTTPS certificate and private key must either both exist or both be absent")
    if cert_path.is_file() and key_path.is_file() and _certificate_is_usable(
        cert_path, key_path, normalized_hosts
    ):
        return cert_path, key_path

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    common_name = next(
        (host for host in normalized_hosts if not _looks_like_ip(host)), normalized_hosts[0]
    )[:64]
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=max(1, valid_days)))
        .add_extension(x509.SubjectAlternativeName(_general_names(normalized_hosts)), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_bytes = cert.public_bytes(serialization.Encoding.PEM)
    temporary_key = key_path.with_suffix(key_path.suffix + ".tmp")
    temporary_cert = cert_path.with_suffix(cert_path.suffix + ".tmp")
    temporary_key.write_bytes(key_bytes)
    temporary_cert.write_bytes(cert_bytes)
    os.chmod(temporary_key, 0o600)
    temporary_key.replace(key_path)
    temporary_cert.replace(cert_path)
    return cert_path, key_path


def _looks_like_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False
