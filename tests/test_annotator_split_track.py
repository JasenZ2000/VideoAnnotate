from __future__ import annotations

import unittest

from utils.annotator.state import AnnotationState


class AnnotationTrackSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = AnnotationState()
        self.state.width = 1920
        self.state.height = 1080
        self.track_id = self.state.add_track(class_id=7)
        for frame_idx in (3, 4, 5, 6):
            self.state.set_frame(self.track_id, frame_idx, [10, 20, 30, 40])

    def test_split_moves_only_frames_strictly_after_boundary(self) -> None:
        new_track_id, moved = self.state.split_track_after(self.track_id, 4)

        self.assertEqual(moved, 2)
        self.assertEqual(self.state.get_annotated_frame_indices(self.track_id), [3, 4])
        self.assertEqual(self.state.get_annotated_frame_indices(new_track_id), [5, 6])
        self.assertEqual(self.state.tracks[new_track_id].class_id, 7)

    def test_split_without_later_annotations_does_not_create_track(self) -> None:
        next_track_id = self.state.next_track_id

        with self.assertRaisesRegex(ValueError, "No annotations after"):
            self.state.split_track_after(self.track_id, 6)

        self.assertEqual(self.state.next_track_id, next_track_id)
        self.assertEqual(self.state.get_track_ids(), [self.track_id])


if __name__ == "__main__":
    unittest.main()
