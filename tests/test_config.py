from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from config.config import AgentConfig


class SystemAgentUiTests(unittest.TestCase):
    def test_default_window_configuration_is_compact(self) -> None:
        config = AgentConfig()
        self.assertLess(config.quick_height, config.window_height)
        self.assertLessEqual(config.min_width, config.quick_width)
        self.assertLessEqual(config.window_width, config.max_width)
        self.assertEqual((config.quick_width, config.quick_height), (420, 50))

    def test_preferred_quick_size_round_trips_through_toml(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "system-agent" / "config.toml"
            config = AgentConfig()
            config.save_quick_size(480, 60, config_path)
            restored = AgentConfig.load(config_path)

        self.assertEqual(restored.quick_width, 480)
        self.assertEqual(restored.quick_height, 60)

    def test_legacy_oversized_overlay_config_is_migrated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "system-agent" / "config.toml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                """
shortcut = \"{'activate': '<Super>space'}\"
[window]
width = 700
height = 182
quick_width = 737
quick_height = 100
min_width = 500
min_height = 90
""",
                encoding="utf-8",
            )
            restored = AgentConfig.load(config_path)

        self.assertEqual((restored.quick_width, restored.quick_height), (420, 50))
        self.assertEqual((restored.min_width, restored.min_height), (420, 50))
        self.assertEqual((restored.window_width, restored.window_height), (430, 360))
        self.assertEqual(restored.shortcut, "<Super>space")

if __name__ == "__main__":
    unittest.main()
