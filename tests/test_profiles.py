import unittest

from voice_asr_lab.profiles import detect_profile, profile_for_vram


class ProfileTests(unittest.TestCase):
    def test_profile_thresholds(self):
        cases = [(14, "t4"), (15.9, "t4"), (20, "l4"), (24, "l4"), (35, "a100"), (80, "a100")]
        for vram, expected in cases:
            with self.subTest(vram=vram):
                self.assertEqual(profile_for_vram(vram).name, expected)

    def test_profile_rejects_too_small_gpu(self):
        with self.assertRaisesRegex(RuntimeError, "at least 14 GB"):
            profile_for_vram(12)

    def test_forced_profile(self):
        self.assertEqual(detect_profile(force="l4", vram_gb=80).name, "l4")

    def test_unknown_forced_profile(self):
        with self.assertRaisesRegex(ValueError, "Unknown LAB_PROFILE"):
            detect_profile(force="mystery", vram_gb=80)


if __name__ == "__main__":
    unittest.main()
