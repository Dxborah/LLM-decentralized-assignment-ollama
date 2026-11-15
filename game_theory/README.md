# game_theory/

Every rule-based conflict decision in `agents/agent_rank_once_bfs.py` is
replaced by a **collaborative team game**. The game is played by a **decider**
you choose: the original rule, an exact Nash solver, or the LLM itself (one
independent call per agent, no communication). Results are scored against
Nash as in *Do LLMs Beat Nash?*. Nothing outside this folder is modified; the
code only imports from `core/`.

The game comes in **two forms**, chosen with `--game`:

| Form | Question each agent answers | Game type | Status |
|---|---|---|---|
| `claimyield` | "CLAIM or YIELD?" | Anti-coordination: the good outcomes need *different* answers | **Default**: the original 5 Nov version, unchanged |
| `matching` | "Which agent should take it?" | Matching: the good outcomes need the *same* answer | New, 10 Nov |

The full design is in the doc "Game-theoretic conflict resolution: redesign";
the payoff matrix and prompts are in "Game-theoretic conflict resolution:
advisor update".

| File | What it is |
|---|---|
| `conflict_games.py` | The team game (`TeamConflictGame`), its matching form (`MatchingConflictGame`), the `rule` / `nash` / `llm` deciders, the Option A replay, and the score *s* |
| `llm_player.py` | The two prompts (claim/yield and matching), the Ollama call, and reply parsing |
| `agent_rank_once_game.py` | Copy of `agent_rank_once_bfs.py` with every rule replaced (each change marked `# ORIGINAL:`) |
| `eval_game.py` | Copy of `tasks/eval_final.py` for this agent |
| `replay_conflicts.py` | Re-asks the conflicts of a finished matching run with prompt variant A or B, without rerunning the maps (new, 12 Nov) |
| `make_eval_maps.py` | Makes the 100 seeded random 20 × 20 maps in `configs/eval/` (new, 11 Nov) |
| `game.py` | Generic exact pure-Nash-equilibrium solver |

## Running the original claim/yield version

**Leave out `--game`.** The default is `claimyield`, which runs exactly the
5 Nov code: same prompts, same results and the same file names
(`agent_rank_once_game_<decider>_*`). This was checked by running the old and
new code side by side on all 10 cases; the output files were byte-for-byte
identical.

```bash
python -m game_theory.eval_game --decider llm --yes                     # original claim/yield
python -m game_theory.eval_game --decider llm --game claimyield --yes   # same thing, explicit
```

## Running the matching version

```bash
python -m game_theory.eval_game --decider llm --game matching --yes
```

Its files get a `_matching` suffix (`agent_rank_once_game_<decider>_matching_*`),
so they never mix with the claim/yield files. Only the `llm` decider needs
running in this form: `rule` and `nash` never read a prompt, so they give
the same steps in both forms.

## Prompt variants A and B (matching form, new 12 Nov)

`--prompt-variant` changes only the **wording** of the matching prompt (in
`llm_player.py`); the game, payoffs and parsing are the same. None of the
variants hints at which agent is better.

| Variant | What changes |
|---|---|
| `current` | **Default.** The 10 Nov matching prompt, unchanged, same file names |
| `A` | Team finishing time in steps (lower is better) instead of negative payoffs; one-line JSON answer with at most 2 sentences of reasoning |
| `B` | A, plus the 2-agent outcomes as a grid (rows = your answer, columns = the other agent's) and one sentence saying the other agent sees the same table. 3+ agents keep A's list plus that sentence |
| `C` | **New 15 Nov.** A, but 2-agent conflicts use the 3+ layout: only the agreed outcomes ("you both name Agent w -> T steps") plus one line for the two disagreements. Adds B's "sees the same information" sentence, and the JSON asks for `team_steps` (the team time under each agreed choice) before `answer` |

**Why C (15 Nov):** in the A replay, all 46 two-agent agreements were on the
agent the prompt listed first, so A's 97.9% agreement came from list order,
and its best-agent rate (51%) was a coin flip. A's 3+ layout did not have
this problem (4 of 11 agreements on the first-listed agent, 83% best agent),
and B's sentence helped 3+ conflicts (+0.88 -> +0.99). The replay summary now
reports **first-listed**: of the agreements, how many were on the agent listed
first. For C, it should fall well below 100% on 2-agent conflicts.

**Quick test (about 10–15 min per variant):** re-ask the 59 conflicts of the
20 × 20 matching run, with the exact same games, and compare round 1 with the
current prompt. No maps are simulated, so there is no makespan here.

```bash
python -m game_theory.replay_conflicts --variants A    # now
python -m game_theory.replay_conflicts --variants B    # later; the summary then shows both
python -m game_theory.replay_conflicts --variants C    # new 15 Nov; the summary shows all three
```

It writes `results_eval/replay_A.csv` / `replay_B.csv` / `replay_C.csv` (one row per conflict:
old and new names, agreed, best agent, *s*, each agent's reasoning) and
`results_eval/replay_summary.txt` (agreement, agreed-on-best-agent, mean *s*
unparsed replies and first-listed, for all / 2-agent / 3+-agent conflicts). Use
`--conflicts <file>` to replay another run's conflicts CSV and `--limit 5` for
a short test.

**Full run with a variant** (only for the variant that wins the replay):

```bash
python -m game_theory.eval_game --decider llm --game matching --prompt-variant A --config-dir configs/eval --cases case_1 case_53 case_13 case_36 case_5 case_27 case_3 case_4 case_2 case_9 --yes
```

Its files get an extra `_A` / `_B` / `_C` suffix (`agent_rank_once_game_llm_matching_A_*`).

## The 100 random maps (`configs/eval/`)

These maps are used to compare with Ismayilov et al.
([arXiv 2510.23824](https://arxiv.org/abs/2510.23824)), whose own map files
aren't in the repo. They use the paper's settings, through the repo's own
generator `core/generate_cases.py`:

- a 20 × 20 grid
- 2 to 6 agents, with the same number of goals
- 15 to 30 obstacles, never on an agent or goal cell

Two additions make the set reproducible and fair:

- **Fixed seed (2026).** The paper used no seed, so its exact maps can't be
  recreated. Rerunning `python -m game_theory.make_eval_maps` always gives
  these same 100 maps.
- **Solvability check.** A map is redrawn if any agent can't reach any goal.
  No map needed redrawing with seed 2026.

`configs/eval/maps_summary.csv` lists each map's number of agents and
obstacles, its optimal makespan and the greedy baseline's makespan. Overall,
the 100 maps have a mean optimal makespan of 13.17 and a mean greedy
makespan of 16.11 (greedy fails on 2 maps). The paper's maps had 13.93 and
17.93, so the difficulty is similar but the maps are not the same.

### Running only some maps

```bash
# first 5 maps
python -m game_theory.eval_game --decider llm --game matching --config-dir configs/eval --limit 5 --yes
# specific maps
python -m game_theory.eval_game --decider llm --game matching --config-dir configs/eval --cases case_3 case_17 --yes
```

- **Output folder:** results for `configs/eval` go to `results_eval/`, so they
  never mix with the 8 × 8 results in `results_game/`.
- **Resuming:** rerunning with a bigger `--limit` or more `--cases` only runs
  the maps not done yet. The summary always covers every map in the results
  file.

## Metrics

All metrics are in `<key>_team_summary.txt`; per-trial values are in
`<key>_team_results.csv` and per-conflict values in `<key>_conflicts.csv`.

**Paper metrics (Ismayilov et al.)**: use these to compare with the paper.

| Metric | Definition |
|---|---|
| Makespan | Number of timesteps until every agent has reached a goal (`Steps`). Lower is better. A run that reaches 100 steps is a failure. |
| Optimal makespan | The best makespan possible on that map: brute-force search over every way of giving each agent its own goal, keeping the one whose longest BFS path is shortest (`TrueOptimal`, from `core/find_optim_sol.py`). It ignores agents blocking each other, so it is a lower bound. |
| Mean makespan | Average makespan over the finished trials. |
| Mean optimal makespan | Average optimal makespan over the same trials. |
| Performance gap | Mean makespan − mean optimal makespan, in steps; also shown as a percentage of the mean optimal makespan. The paper's "difference from the optimal assignment". |
| Greedy baseline | Makespan of the repo's greedy agent (`agents/agent_greedy.py`) on the same map: each agent goes to its nearest goal by BFS, lower index wins. No LLM. Reported per map (`Greedy` column) and as a mean, leaving out maps where greedy fails. |

**Our additional metrics:**

| Metric | Definition |
|---|---|
| Optimal trials | Number of trials whose makespan equals the optimal makespan. |
| Average optimality gap | Mean over finished trials of (makespan − optimal) / optimal. This differs slightly from the paper's gap, which divides the difference of the means. |
| Failures | Trials that hit the 100-step limit. They are left out of the means and counted separately. |
| Agents made to wait | Agents that didn't move on a step because of a cell conflict. |
| Conflicts | Games played: goal conflicts, cell clashes and re-plans. |
| Score *s* | Per conflict, on round 1: (a − N)/(O − N) if a ≥ N, else (a − N)/(N − W). Here a is the payoff achieved, O the best cell, W the worst cell and N the worst Nash equilibrium. 0 = Nash level, +1 = optimum, −1 = worst. Conflicts where O = N have no room to improve and are left out. |
| Easy / hard conflicts | Split by the gap between the best and worst pure equilibrium: 2 steps or more is easy, less is hard. |
| Coordination rate | Share of conflicts with exactly one claimer in round 1. For matching, the same as the agreement rate. |
| Agreement rate (matching) | Share of conflicts where every agent named the same agent in round 1. |
| Clash rate / all-yield rate | Share of conflicts where, in round 1, 2+ agents claimed (or, for matching, named themselves), or nobody did. |
| Nash fallbacks | Conflicts still unresolved after the replay, so the Nash solver decided. |
| Unparsed replies | Model replies that couldn't be read (counted as YIELD, or as naming nobody). |

**Paper reference values** (GPT-4.1, the paper's own 100 maps, so not the
same maps as ours):

| Method | Mean makespan |
|---|---|
| Optimal | 13.93 |
| LLM, rank every step, distances shown | 15.12 |
| LLM, rank once, distances shown (our base agent) | 15.50 |
| LLM, rank every step, no distances | 17.67 |
| Greedy | 17.93 |
| Random assignment | 20.54 |
| LLM, rank every step, LLaVA | 27.98 |

The fairest comparison is the performance gap and our greedy baseline
measured on our own maps, set against the paper's gap (15.50 − 13.93 = 1.57
steps for rank once) and its greedy result.

## Decision points

| Decision | Original rule | Now |
|---|---|---|
| 2+ agents rank the same goal first | Lowest index wins | Team game, one per contested goal |
| 2+ BFS moves land on the same cell | Lowest index moves | Team game (take it = move, give way = wait) |
| Ranking can't be parsed | Random goal | Distance order; the agent still plays the games |
| An agent's target is taken | Stands still until step 100 | The goal game is re-played over the free goals (always on) |
| BFS step, stop on any free goal | — | Unchanged |

## The game

`a_i` is agent *i*'s cost if it takes the goal or cell; `b_i` is its cost if
it gives way. Every player gets the same payoff, minus the local makespan.
`K` = 0.75 × grid size and `D` = 0.375 × grid size (6 and 3 on 8 × 8).

**Claim/yield form:**

- **One claimer:** −max(`a` of the claimer, `b` of each yielder)
- **Clash (2+ claim):** −max(`K` + the largest `a` among claimers, the largest `b` among yielders)
- **Nobody claims:** −(the largest `b` + `D`)

**Matching form:**

- **Everyone names the same agent w:** w takes it; the payoff is −max(`a_w`, `b` of every other agent)
- **Any disagreement fails.** Agents that named themselves head there anyway: −max(`K` + their largest `a`, the others' largest `b`). If nobody named themselves: −(the largest `b` + `D`)

With two agents the two forms have the same four outcomes and payoffs
(naming yourself = CLAIM), so their scores are directly comparable.

**Option A (both forms):** if round 1 fails (claim/yield: not exactly one
claimer; matching: the agents disagree), it is replayed once with the option
order reversed, and then the Nash solver decides. Each failed round costs a
waiting step.

**Score** (round 1): *s* = 0 is Nash level, +1 the optimum and −1 the worst
outcome. *N* is the worst equilibrium; mixed equilibria are included for
2-player conflicts.

## Outputs (in `results_game/`, `results_eval/` for the random maps, or `$RESULTS_DIR`)

| File | Contents |
|---|---|
| `<key>_team_results.csv` | One row per trial: Steps, Failed, Collisions, TrueOptimal, and Greedy for files created from 11 Nov |
| `<key>_conflicts.csv` | One row per conflict: who played, their answers, the payoffs, *s*, the equilibrium gap. The matching form adds `round1_names`, `final_names` and `agreed` |
| `<key>_moves.csv` | Every agent's move at every step, with Case and Trial columns |
| `<key>_team_summary.txt` | Task-level and game-level summary; the matching form adds the agreement rate |
| `<key>_reasoning.csv` | **`llm` runs only (new 10 Nov).** One row per model answer: each agent's goal ranking (`decision` = ranking) and every conflict answer (goal / cell / replan, round 1 or the round-2 replay), with the model's `reasoning` text and its full raw `reply` |

`<key>` is `agent_rank_once_game_<decider>` for claim/yield,
`agent_rank_once_game_<decider>_matching` for matching, and
`agent_rank_once_game_<decider>_matching_A` / `_B` for prompt variants. A run writes these 4
files, plus the reasoning file for `llm`. The reasoning log only records
text; it doesn't change any decision or result.

The full list of commands, and what to expect, is in the design doc.
