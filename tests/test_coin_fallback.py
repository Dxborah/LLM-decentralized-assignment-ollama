"""Offline checks for fair matching-game fallbacks; no model calls."""
import contextlib
import csv
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from game_theory import eval_game
from game_theory.conflict_games import CLAIM, YIELD, TeamConflictGame, play_conflict
from game_theory.eval_game import task_key, trial_fallback_seed


class CoinFallbackTests(unittest.TestCase):
    def game(self, players=(0, 1)):
        return TeamConflictGame('goal', 'G', players,
                                {i: 1 + 8 * k for k, i in enumerate(players)},
                                {i: 4 - k for k, i in enumerate(players)}, 15, 7.5)

    def play(self, ask, *, game=None, rng=None, fallback='coin'):
        with contextlib.redirect_stdout(io.StringIO()):
            return play_conflict(game or self.game(), 'llm', form='matching',
                                 prompt_variant='C', ask_fn=ask,
                                 fallback=fallback, rng=rng)

    @staticmethod
    def disagree(game, agent, rotation):
        return agent, {'parse_failed': False}

    def test_first_round_agreement_uses_no_coin(self):
        rng = Mock()
        final, record = self.play(lambda g, a, r: (1, {}), rng=rng)
        self.assertEqual(final, (YIELD, CLAIM))
        self.assertEqual(record['agreed'], 1)
        self.assertEqual(record['coin_fallback'], 0)
        rng.choice.assert_not_called()

    def test_second_round_agreement_uses_no_coin(self):
        rng = Mock()
        final, record = self.play(lambda g, a, r: (a if r == 0 else 1, {}), rng=rng)
        self.assertEqual(final, (YIELD, CLAIM))
        self.assertEqual(record['agreed'], 0)
        self.assertEqual(record['round2_agreed'], 1)
        self.assertEqual(record['clash_rounds'], 1)
        rng.choice.assert_not_called()

    def test_coin_can_pick_the_higher_cost_higher_index_agent(self):
        rng = Mock()
        rng.choice.return_value = 1
        final, record = self.play(self.disagree, rng=rng)
        rng.choice.assert_called_once_with((0, 1))
        self.assertEqual(final, (YIELD, CLAIM))
        self.assertEqual(record['fallback_winner'], 'A2')
        self.assertEqual(record['coin_candidates'], 'A1 A2')
        self.assertEqual(record['coin_win_probability'], 0.5)
        self.assertEqual(record['coin_fallback'], 1)
        self.assertEqual(record['nash_fallback'], 0)
        self.assertEqual(record['agreed'], 0)
        self.assertEqual(record['round2_agreed'], 0)
        self.assertEqual(record['clash_rounds'], 2)

    def test_larger_conflict_uses_all_candidates_equally(self):
        rng = Mock()
        rng.choice.return_value = 8
        final, record = self.play(self.disagree, game=self.game((1, 3, 8)), rng=rng)
        rng.choice.assert_called_once_with((1, 3, 8))
        self.assertEqual(final, (YIELD, YIELD, CLAIM))
        self.assertEqual(record['coin_win_probability'], 1 / 3)
        self.assertEqual(record['fallback_winner'], 'A9')

    def test_invalid_replies_also_trigger_coin_and_stay_failures(self):
        rng = Mock()
        rng.choice.return_value = 0
        _, record = self.play(lambda g, a, r: (-1, {'parse_failed': True}), rng=rng)
        self.assertEqual(record['parse_failures'], 4)
        self.assertEqual(record['coin_fallback'], 1)
        self.assertEqual(record['agreed'], 0)

    def test_default_nash_behavior_is_preserved(self):
        final, record = self.play(self.disagree, fallback='nash')
        self.assertEqual(final, (CLAIM, YIELD))
        self.assertEqual(record['nash_fallback'], 1)
        self.assertEqual(record['coin_fallback'], 0)

    def test_seeded_coins_are_balanced_and_reproducible(self):
        def draws():
            rng = random.Random(20261117)
            return [self.play(self.disagree, rng=rng)[1]['fallback_winner'] for _ in range(1000)]
        first = draws()
        self.assertEqual(first, draws())
        self.assertGreater(first.count('A1'), 440)
        self.assertLess(first.count('A1'), 560)

    def test_trial_seed_is_stable_and_independent(self):
        self.assertEqual(trial_fallback_seed(17, 'case_1', 1), trial_fallback_seed(17, 'case_1', 1))
        self.assertNotEqual(trial_fallback_seed(17, 'case_1', 1), trial_fallback_seed(17, 'case_1', 2))
        self.assertNotEqual(trial_fallback_seed(17, 'case_1', 1), trial_fallback_seed(17, 'case_2', 1))

    def test_coin_result_names_are_separate(self):
        self.assertEqual(task_key('llm', 'matching', 'C'), 'agent_rank_once_game_llm_matching_C')
        self.assertEqual(task_key('llm', 'matching', 'C', 'coin'),
                         'agent_rank_once_game_llm_matching_C_coin')

    def test_evaluation_logs_coin_events_and_summary_frequency(self):
        def episode(**kwargs):
            _, record = self.play(self.disagree, rng=random.Random(kwargs['fallback_seed']))
            record.update(step=0, decision='goal')
            kwargs['conflict_log'].append(record)
            return 5, 4, False, 0

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(eval_game, 'OUTPUT_DIR', folder), \
                patch.object(eval_game, 'true_optimum', return_value=5), \
                patch.object(eval_game, 'greedy_steps', return_value=6), \
                contextlib.redirect_stdout(io.StringIO()):
            key = task_key('llm', 'matching', 'C', 'coin')
            eval_game.evaluate_team(key, episode, {'case_1': ('unused.yaml', [1, 2])},
                                    'llm', .75, .375, 'matching', 'C', 'coin', 17)
            with (Path(folder) / f'{key}_conflicts.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['coin_fallback'] == '1' and r['nash_fallback'] == '0' for r in rows))
            self.assertTrue(all(r['agreed'] == '0' and r['coin_win_probability'] == '0.5' for r in rows))
            self.assertNotEqual(rows[0]['fallback_seed'], rows[1]['fallback_seed'])
            summary = (Path(folder) / f'{key}_team_summary.txt').read_text()
            self.assertIn('Coin fallbacks: 2/2 conflicts (100.0%)', summary)
            self.assertIn('Trials requiring a coin fallback: 2/2', summary)
            self.assertIn('Nash fallbacks after the replay: 0', summary)


if __name__ == '__main__':
    unittest.main()
