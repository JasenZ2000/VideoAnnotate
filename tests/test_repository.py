from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


class RepositoryTests(unittest.TestCase):
    def test_readme_local_links_exist(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(readme):
            if "://" in target or target.startswith("#"):
                continue
            path = PROJECT_ROOT / target.split("#", 1)[0]
            with self.subTest(target=target):
                self.assertTrue(path.exists(), f"README link does not exist: {target}")

    def test_deployment_scripts_exist(self) -> None:
        expected = [
            "scripts/windows/run-platform.bat",
            "scripts/windows/run-platform.ps1",
            "scripts/windows/run-local-workbench.bat",
            "scripts/windows/build-local-workbench.ps1",
            "scripts/windows/stop-local-workbench.bat",
            "scripts/linux/run-gpu-service.sh",
            "gpu_services/run_gpu_service.sh",
            "scripts/check-services.py",
        ]
        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue((PROJECT_ROOT / relative).is_file())


if __name__ == "__main__":
    unittest.main()
