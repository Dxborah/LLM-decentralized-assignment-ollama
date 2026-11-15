# Running the 100 random maps with a larger model

How to run the best conflict prompt (`--prompt-variant C`) on all 100 random
20 × 20 maps in `configs/eval/`, plus the Nash and rule baselines, and draw
the maps. All commands run from the repo root.

## 1. Set the model

The model is chosen with environment variables. **Set both.** If
`LLM_BACKEND` is missing, the conflict games use Ollama but the goal
rankings silently go to OpenAI's GPT-4.1 (the defaults in `core/request.py`
and `game_theory/llm_player.py` differ).

```bash
ollama pull qwen3.8:27b                  # or the model you choose (must accept images)
export LLM_BACKEND=ollama
export OLLAMA_MODEL=qwen3.8:27b
export OLLAMA_NUM_CTX=8192               # default; the prompts fit well within it
export RESULTS_DIR=results_eval_qwen3.8_27b
```

- The model **must accept images**: the goal ranking sends a picture of the
  grid with the prompt.
- `RESULTS_DIR` keeps this run apart from the 7B results in `results_eval/`.
  Without it, the run would see the 10 maps already done there, skip them,
  and mix the two models in one file. Putting the model name in the folder
  name is also the only record of which model was used.

## 2. Quick check (a few minutes)

```bash
python -m game_theory.eval_game --decider llm --game matching --prompt-variant C --config-dir configs/eval --limit 2 --yes
```

Check that `$RESULTS_DIR/agent_rank_once_game_llm_matching_C_team_results.csv`
has 4 rows and that the console never prints `Sending request to OpenAI API`
(that line means the rankings went to OpenAI: `LLM_BACKEND` is not set). The full run
below continues from here; nothing is repeated.

## 3. Full run

```bash
# the LLM with the best prompt: 100 maps x 2 trials (list order reversed in trial 2)
python -m game_theory.eval_game --decider llm --game matching --prompt-variant C --config-dir configs/eval --yes

# baselines on the same maps (no model calls, a few minutes each)
python -m game_theory.eval_game --decider nash --game matching --config-dir configs/eval --trials 1 --yes
python -m game_theory.eval_game --decider rule --game matching --config-dir configs/eval --trials 1 --yes
```

- **Interrupted?** Run the same command again with the same environment
  variables. It skips every trial already written.
- **Shorter run:** `--trials 1` halves the time. With the 7B model the two
  trials gave the same makespan on 9 of 10 maps.
- **Time:** the 7B model took about 50 min for 10 maps × 2 trials, so about
  8 h for all 100 maps × 2. A 27–31B model is usually several times slower
  on the same GPU, and much slower again if the model "thinks" before
  answering (see the note at the end).

## 4. Where the results are

Everything is in `$RESULTS_DIR/` (e.g. `results_eval_qwen3.8_27b/`):

| File | Contents |
|---|---|
| `agent_rank_once_game_llm_matching_C_team_summary.txt` | **Start here.** Mean makespan, gap to optimal, greedy baseline, and the game metrics (agreement, mean *s*) |
| `agent_rank_once_game_llm_matching_C_team_results.csv` | One row per trial: makespan (`Steps`), `TrueOptimal`, `Greedy`, failures |
| `agent_rank_once_game_llm_matching_C_conflicts.csv` | One row per conflict: players, answers, payoffs, score *s* |
| `agent_rank_once_game_llm_matching_C_reasoning.csv` | Every model answer with its reasoning text and raw reply |
| `agent_rank_once_game_llm_matching_C_moves.csv` | Every agent's move at every step (used for the map pictures) |
| `agent_rank_once_game_nash_matching_*`, `..._rule_matching_*` | The same files for the Nash solver and the original rule |

Metric definitions: `game_theory/README.md`, section "Metrics".

## 5. Map pictures

```bash
python3 figures/plot_eval_maps.py --results-dir $RESULTS_DIR
```

Writes `$RESULTS_DIR/maps/case_<n>.png` (optimal paths, best prompt, Nash,
side by side, each with its makespan) and `overview.png`, `overview_2.png`,
... (optimal vs best prompt, 20 maps per page). Run it after the Nash
baseline so the Nash panel is filled. `--cases case_5 case_27` draws only
some maps; `--trial 2` draws the second trial.

## Reference: the 7B results (qwen2.5vl:7b, 10 maps × 2 trials)

| Method | Mean makespan | Gap to optimal |
|---|---|---|
| Optimal | 14.40 | – |
| Nash | 16.00 | 1.60 steps (11.1%) |
| LLM, best prompt (C) | 16.10 | 1.70 steps (11.8%) |
| LLM, old prompt | 16.75 | 2.35 steps (16.3%) |
| Greedy | 19.20 | 4.80 steps |
| Original rule | 21.00 | 6.60 steps (45.8%) |

On all 100 maps: mean optimal makespan 13.17, mean greedy makespan 16.11
(greedy fails on 2 maps).

## Note: thinking models

Many current models (the Qwen3.5+ family, Gemma 4) can reason at length
before answering. The code sends no thinking setting, so the model's Ollama
default applies. Thinking may help the arithmetic in 3-agent conflicts but
makes every call much slower, and it changes what is being compared with the
7B run. Check the default with `ollama show <model>` and decide before the
full run; record the choice with the results.
