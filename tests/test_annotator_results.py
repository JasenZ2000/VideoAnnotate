from __future__ import annotations

import unittest

from utils.annotator.results import append_tracking_results


class AnnotatorResultsTests(unittest.TestCase):
    def test_append_remaps_every_track_id_without_collisions(self) -> None:
        base = {
            "metadata": {"num_tracks": 1},
            "tracks": [{"track_id": 2, "class_id": 0, "frames": []}],
        }
        additional = {
            "tracks": [
                {"track_id": 0, "class_id": 1, "frames": []},
                {"track_id": 2, "class_id": 2, "frames": []},
            ],
        }

        merged = append_tracking_results(base, additional, class_id_override=9)

        self.assertEqual([track["track_id"] for track in merged["tracks"]], [2, 3, 4])
        self.assertEqual([track["class_id"] for track in merged["tracks"]], [0, 9, 9])
        self.assertEqual(merged["metadata"]["num_tracks"], 3)
