from __future__ import annotations

import unittest

from tools.contracts import ToolResult
from ui.output_format import format_tool_result


class OutputFormatTests(unittest.TestCase):
    def test_structured_result_is_readable_text_not_json(self) -> None:
        lines = [line for line, _error in format_tool_result(ToolResult(
            "disk_usage",
            True,
            {
                "path": "/home/pr",
                "total_bytes": 1_000_000_000,
                "used_bytes": 500_000_000,
                "free_bytes": 500_000_000,
            },
        ))]
        rendered = "\n".join(lines)
        self.assertIn("Path: /home/pr", rendered)
        self.assertIn("Total: 953.7 MB", rendered)
        self.assertNotIn("{", rendered)
        self.assertNotIn('"path"', rendered)

    def test_subprocess_output_and_errors_use_normal_labels(self) -> None:
        lines = format_tool_result(ToolResult(
            "ping_connectivity",
            True,
            {"host": "1.1.1.1", "reachable": True, "exit_code": 0, "stdout": "1 packets transmitted\n1 received"},
        ))
        rendered = "\n".join(line for line, _error in lines)
        self.assertIn("Reachable: Yes", rendered)
        self.assertIn("Exit code: 0", rendered)
        self.assertIn("Output", rendered)

        error_lines = format_tool_result(ToolResult("tool", False, error_code="timeout", error_message="Timed out"))
        self.assertEqual(error_lines[0], ("Error: Timed out", True))


if __name__ == "__main__":
    unittest.main()
