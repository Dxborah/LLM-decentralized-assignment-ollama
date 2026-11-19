"""Preserve model rankings when complete replies contain multiline strings."""
import json
import os
import unittest
from unittest.mock import mock_open, patch

from game_theory.agent_rank_once_game import parse_ranking_response


class MultilineRankingTests(unittest.TestCase):
    def parse(self, reply, enabled=True):
        with patch.dict(os.environ, {"LLM_ALLOW_MULTILINE_JSON": str(enabled).lower()}), \
             patch("builtins.open", mock_open()), patch("builtins.print"):
            return parse_ranking_response(reply)

    def test_multiline_explanation_preserves_goal_order_and_text(self):
        reply = '```json\n{"reasoning":"First line.\nSecond line.","explanation":"Team choice.","ranking":["B","A"]}\n```'
        self.assertEqual(self.parse(reply),
                         (["B", "A"], "Team choice.", "First line.\nSecond line."))

    def test_strict_mode_still_rejects_unescaped_newlines(self):
        reply = '{"reasoning":"First\nSecond","ranking":["B","A"]}'
        self.assertEqual(self.parse(reply, enabled=False), ([], "", ""))

    def test_valid_json_has_identical_results_in_both_modes(self):
        reply = json.dumps({"reasoning": "First\nSecond", "ranking": ["B", "A"]})
        self.assertEqual(self.parse(reply), self.parse(reply, enabled=False))
        self.assertEqual(self.parse(reply)[0], ["B", "A"])

    def test_truncated_reply_is_not_completed_or_given_a_ranking(self):
        reply = '{"reasoning":"First\nSecond","ranking":["B","A"'
        self.assertEqual(self.parse(reply), ([], "", ""))

    def test_other_json_errors_are_not_repaired(self):
        reply = '{"reasoning":"First\nSecond" "ranking":["B","A"]}'
        self.assertEqual(self.parse(reply), ([], "", ""))


if __name__ == "__main__":
    unittest.main()
