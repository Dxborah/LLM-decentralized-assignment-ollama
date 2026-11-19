import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from baseline_support import ConflictReplay, RankingReplay, inputs_ready, ranking_rows


class BaselineSupportTests(unittest.TestCase):
    def test_original_empty_ranking_is_preserved_and_order_is_not_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rankings.csv"
            path.write_text("Case,Trial,decision,agent,answer,reasoning,reply\n"
                            "case_1,1,ranking,1,B A,r,e\n"
                            "case_1,1,ranking,2,none,,\n")
            cache = ranking_rows(path)[("case_1", 1)]
        replay = RankingReplay()
        replay.current = cache
        self.assertEqual(replay.select(agent_id=1)[0], ["B", "A"])
        self.assertEqual(replay.select(agent_id=2)[0], [])

    def test_missing_input_is_not_treated_as_an_empty_model_ranking(self):
        self.assertFalse(inputs_ready({"Steps": "20"}, {1: {}}, [1, 2]))
        replay = RankingReplay()
        replay.current = {1: {"ranking": [], "reasoning": "", "explanation": ""}}
        with self.assertRaises(RuntimeError):
            replay.select(agent_id=2)

    def test_original_replay_rejects_changed_costs(self):
        record = {"players": "A1 A2", "kind": "goal", "subject": "B",
                  "a": "4 20", "b": "6 28", "clash_rounds": "0",
                  "final_actions": "YIELD CLAIM"}
        game = SimpleNamespace(players=[0, 1], kind="goal", subject="B",
                               a={0: 5, 1: 20}, b={0: 6, 1: 28})
        with self.assertRaises(AssertionError):
            ConflictReplay([record])(game, "llm")

    def test_recorded_zero_waits_remain_zero_not_a_truthy_string(self):
        record = {"players": "A1 A2", "kind": "goal", "subject": "B",
                  "a": "4 20", "b": "6 28", "clash_rounds": "0",
                  "final_actions": "YIELD CLAIM"}
        game = SimpleNamespace(players=[0, 1], kind="goal", subject="B",
                               a={0: 4, 1: 20}, b={0: 6, 1: 28})
        actions, replayed = ConflictReplay([record])(game, "llm")
        self.assertEqual(actions, ("YIELD", "CLAIM"))
        self.assertFalse(replayed["clash_rounds"])


if __name__ == "__main__":
    unittest.main()
