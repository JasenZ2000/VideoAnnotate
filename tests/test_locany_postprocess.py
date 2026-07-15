from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from locany_batch_tool.postprocess import organize_prelabels


class LocateAnythingPostprocessTests(TestCase):
    def _layout(self, root: Path, name: str = "0001") -> tuple[Path, Path, Path]:
        videos = root / "videos"
        prelabels = root / "prelabels"
        workspace = prelabels / name
        videos.mkdir(parents=True)
        workspace.mkdir(parents=True)
        (videos / f"{name}.mp4").write_bytes(b"video-content")
        return videos, prelabels, workspace

    def test_preview_does_not_modify_files(self) -> None:
        with TemporaryDirectory() as temporary:
            videos, prelabels, workspace = self._layout(Path(temporary))
            labels = workspace / "labels"
            labels.mkdir()
            (labels / "0001_1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            result = organize_prelabels(str(videos), str(prelabels), dry_run=True)
            self.assertEqual(result["counts"]["ready"], 1)
            self.assertTrue(labels.is_dir())
            self.assertFalse((workspace / "0001.mp4").exists())

    def test_execute_copies_video_and_renames_labels(self) -> None:
        with TemporaryDirectory() as temporary:
            videos, prelabels, workspace = self._layout(Path(temporary))
            labels = workspace / "labels"
            labels.mkdir()
            (labels / "0001_1.txt").write_text("label", encoding="utf-8")
            result = organize_prelabels(str(videos), str(prelabels), dry_run=False)
            self.assertEqual(result["counts"]["done"], 1)
            self.assertEqual((workspace / "0001.mp4").read_bytes(), b"video-content")
            self.assertEqual((workspace / "0001" / "0001_1.txt").read_text(encoding="utf-8"), "label")
            self.assertFalse(labels.exists())

    def test_rerun_accepts_already_completed_workspace(self) -> None:
        with TemporaryDirectory() as temporary:
            videos, prelabels, workspace = self._layout(Path(temporary))
            (workspace / "0001.mp4").write_bytes(b"video-content")
            renamed = workspace / "0001"
            renamed.mkdir()
            (renamed / "0001_1.txt").write_text("label", encoding="utf-8")
            result = organize_prelabels(str(videos), str(prelabels), dry_run=False)
            self.assertEqual(result["counts"]["done"], 1)

    def test_partial_merge_keeps_unique_files_and_removes_duplicate_labels(self) -> None:
        with TemporaryDirectory() as temporary:
            videos, prelabels, workspace = self._layout(Path(temporary))
            labels = workspace / "labels"
            renamed = workspace / "0001"
            labels.mkdir();renamed.mkdir()
            (labels / "same.txt").write_text("same", encoding="utf-8")
            (renamed / "same.txt").write_text("same", encoding="utf-8")
            (labels / "new.txt").write_text("new", encoding="utf-8")
            result = organize_prelabels(str(videos), str(prelabels), dry_run=False)
            self.assertEqual(result["counts"]["done"], 1)
            self.assertFalse(labels.exists())
            self.assertEqual((renamed / "new.txt").read_text(encoding="utf-8"), "new")

    def test_conflicting_partial_merge_is_not_modified(self) -> None:
        with TemporaryDirectory() as temporary:
            videos, prelabels, workspace = self._layout(Path(temporary))
            labels = workspace / "labels"
            renamed = workspace / "0001"
            labels.mkdir();renamed.mkdir()
            (labels / "frame.txt").write_text("new", encoding="utf-8")
            (renamed / "frame.txt").write_text("old", encoding="utf-8")
            result = organize_prelabels(str(videos), str(prelabels), dry_run=False)
            self.assertEqual(result["counts"]["error"], 1)
            self.assertEqual((labels / "frame.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse((workspace / "0001.mp4").exists())
