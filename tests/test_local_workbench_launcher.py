from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from local_workbench import __main__ as launcher


class LocalWorkbenchLauncherTests(unittest.TestCase):
    def test_packaged_startup_error_waits_before_exit(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(launcher, "_start", side_effect=SystemExit("port already in use")),
            patch.object(launcher.sys, "frozen", True, create=True),
            patch("builtins.input", return_value="") as wait_for_enter,
            patch("sys.stderr", stderr),
        ):
            with self.assertRaisesRegex(SystemExit, "1"):
                launcher.run()

        wait_for_enter.assert_called_once()
        self.assertIn("port already in use", stderr.getvalue())

    def test_source_startup_error_does_not_wait(self) -> None:
        with (
            patch.object(launcher, "_start", side_effect=RuntimeError("boom")),
            patch.object(launcher.sys, "frozen", False, create=True),
            patch("builtins.input") as wait_for_enter,
            patch("sys.stderr", io.StringIO()),
        ):
            with self.assertRaisesRegex(SystemExit, "1"):
                launcher.run()

        wait_for_enter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
