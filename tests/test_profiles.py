from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_asr_lab.profiles import (
    COMMON_EFFECTIVE_BATCH_SIZE,
    COMMON_MAX_AUDIO_SECONDS,
    COMMON_TEST_EXAMPLES,
    COMMON_TRAIN_EXAMPLES,
    COMMON_TRAIN_STEPS,
    COMMON_VALIDATION_EXAMPLES,
    PROFILES,
    detect_profile,
    profile_for_vram,
)


class ProfileTests(unittest.TestCase):
    def test_profile_thresholds(self):
        cases = [
            (14, "t4"),
            (15.9, "t4"),
            (20, "l4"),
            (24, "l4"),
            (35, "a100"),
            (80, "a100"),
            (96, "a100"),
        ]
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

    def test_profiles_share_the_comparison_workload(self):
        self.assertEqual(COMMON_TRAIN_EXAMPLES, 400)
        self.assertEqual(COMMON_VALIDATION_EXAMPLES, 50)
        self.assertEqual(COMMON_TEST_EXAMPLES, 100)
        self.assertEqual(
            {profile.max_audio_seconds for profile in PROFILES.values()},
            {COMMON_MAX_AUDIO_SECONDS},
        )
        self.assertEqual(
            {profile.train_steps for profile in PROFILES.values()},
            {COMMON_TRAIN_STEPS},
        )
        self.assertEqual(
            {
                profile.train_batch_size * profile.gradient_accumulation_steps
                for profile in PROFILES.values()
            },
            {COMMON_EFFECTIVE_BATCH_SIZE},
        )

    def test_encoder_tail_remains_profile_specific(self):
        self.assertEqual(PROFILES["t4"].trainable_encoder_layers, 0)
        self.assertEqual(PROFILES["l4"].trainable_encoder_layers, 2)
        self.assertEqual(PROFILES["a100"].trainable_encoder_layers, 2)


if __name__ == "__main__":
    unittest.main()
