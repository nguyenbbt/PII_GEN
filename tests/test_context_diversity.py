import random
import unittest

from pii_factory.application.context_catalog import compatible_context_frames
from pii_factory.application.seed_generation import ContextFrameSelector


class ContextDiversityTests(unittest.TestCase):
    def test_common_label_has_multiple_compatible_contexts(self) -> None:
        frames = compatible_context_frames(["DATE"])

        self.assertGreaterEqual(len(frames), 6)
        self.assertTrue(all("DATE" in frame.supported_labels for frame in frames))

    def test_selector_honours_a_compatible_preferred_frame(self) -> None:
        selector = ContextFrameSelector()

        selected = selector.select(
            ["DATE"],
            random.Random(42),
            preferred_frame_id="travel_booking",
        )

        self.assertEqual(selected.frame_id, "travel_booking")


if __name__ == "__main__":
    unittest.main()
