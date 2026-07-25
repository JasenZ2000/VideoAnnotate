from __future__ import annotations

import tempfile
import unittest
from ipaddress import ip_address
from pathlib import Path

from cryptography import x509

from workflow_platform.tls import ensure_self_signed_certificate, normalize_tls_hosts


class PlatformTlsTests(unittest.TestCase):
    def test_self_signed_certificate_is_reused_and_contains_dns_and_ip_sans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cert_path = Path(temporary) / "tls" / "certificate.pem"
            key_path = Path(temporary) / "tls" / "private-key.pem"
            cert, key = ensure_self_signed_certificate(
                cert_path, key_path, ["annotation-host", "192.0.2.10"],
            )
            first_bytes = cert.read_bytes()
            first_key_bytes = key.read_bytes()
            loaded = x509.load_pem_x509_certificate(first_bytes)
            san = loaded.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            self.assertIn("annotation-host", san.get_values_for_type(x509.DNSName))
            self.assertIn(ip_address("192.0.2.10"), san.get_values_for_type(x509.IPAddress))

            ensure_self_signed_certificate(
                cert_path, key_path, ["annotation-host", "192.0.2.10"],
            )
            self.assertEqual(cert.read_bytes(), first_bytes)
            self.assertEqual(key.read_bytes(), first_key_bytes)

            ensure_self_signed_certificate(
                cert_path, key_path, ["annotation-host", "192.0.2.10", "annotation-host.local"],
            )
            self.assertNotEqual(cert.read_bytes(), first_bytes)

    def test_host_normalization_removes_bind_any_and_duplicates(self) -> None:
        self.assertEqual(
            normalize_tls_hosts(["localhost", "LOCALHOST", "0.0.0.0", "[::1]", ""]),
            ["localhost", "::1"],
        )


if __name__ == "__main__":
    unittest.main()
