"""
game_theory package
===================

Every rule-based conflict decision of agents/agent_rank_once_bfs.py replaced
by a collaborative claim/yield game (design doc: "Game-theoretic conflict
resolution: redesign"). Nothing in agents/, core/ or tasks/ is modified; this
package only imports from core/.

    conflict_games.py         the team game in two forms - claimyield
                              (default, original) and matching (10 Nov) -,
                              the deciders (rule, nash, llm), Option A
                              replays and the score s
    llm_player.py             the two conflict prompts (+ matching variants A, B
                              from 12 Nov), the Ollama call, parsing
    agent_rank_once_game.py   agent_rank_once_bfs with every rule replaced
    eval_game.py              evaluation script (copy of tasks/eval_final.py)
    game.py                   generic exact pure-Nash solver
    replay_conflicts.py       re-asks a run's recorded conflicts with variant A/B
    make_eval_maps.py         the 100 seeded random 20 x 20 maps (configs/eval)

See README.md in this folder for how the games work and how to run them.
"""
