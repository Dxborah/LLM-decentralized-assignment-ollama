# Full 100-world benchmark: LLM matching game and paired rule controls

All **100 worlds (`case_1` through `case_100`) × 5 trials** are complete for
seven model configurations, seven matched rule controls, and one fully non-LLM
greedy control: **7,500 recorded production trials**. Runs finished on 19 November
2026 UTC. No model service is needed to read or verify these data.

Start with [the combined Excel table](reports/combined_comparison.xlsx),
[the same table as CSV](reports/combined_comparison.csv), or
[the full workbook](reports/full_results.xlsx), which includes every world,
every trial, and paired model-versus-rule comparisons.

## Combined results

Values are mean team finishing times in steps; lower is better. Means exclude
task failures, which reached the 100-step cap. Each failed task is still a
completed experiment trial and is counted separately. Optimal is the assignment
makespan reference averaged over the same full 100-world set.

| Model | Ranking input | Optimal | Greedy | Rule | New method | Failures: new / rule |
|---|---|---:|---:|---:|---:|---:|
| Gemma4 E4B | Image + text | 13.17 | 16.11 | 14.37 | 13.74 | 0 / 4 |
| Gemma4 12B | Image + text | 13.17 | 16.11 | 14.06 | 13.78 | 0 / 3 |
| Phi-4 14B | Text only | 13.17 | 16.11 | 15.26 | 14.26 | 4 / 8 |
| Ministral 3 8B | Image + text | 13.17 | 16.11 | 16.21 | 14.77 | 1 / 7 |
| Qwen2.5-VL 7B | Image + text | 13.17 | 16.11 | 18.13 | 15.37 | 0 / 9 |
| Llama3.2 3B | Text only | 13.17 | 16.11 | 18.16 | 17.07 | 4 / 5 |
| Llama3.1 8B | Text only | 13.17 | 16.11 | 18.26 | 17.36 | 0 / 1 |

- **New method:** each model produces initial goal rankings, then resolves
  conflicts using the matching game with prompt C. After two unsuccessful
  agreement rounds, a fair coin/uniform draw selects among the competing agents.
  Its probability is `1 / number of contestants`, independent of cost or ID.
- **Rule:** reuses that model's exact saved initial rankings for each case and
  trial; the lowest competing agent ID wins each conflict. The shared frozen
  movement/replanning code is retained. These are controls for the conflict
  method, not reruns of the entire legacy `agent_rank_once_bfs` agent.
- **Greedy:** shortest-path goal preferences recomputed during movement, with no
  LLM. It had **10 failures / 500 trials**. Five actual deterministic reruns were
  performed; they are not five independent random samples.

Gemma4 12B reached the optimum in 394/500 trials; Gemma4 E4B in 383/500.
All optimal-trial counts, fallback counts, and failure-inclusive averages are in
[summary.csv](reports/summary.csv). Coin rates use conflict count as their
denominator, not trial count. The earlier Gemma4 12B figure of 12.69 covered
only the first 23 worlds (115 trials); its full-set mean is 13.778.

## Data layout

```text
models/<run_id>/results/     Team outcomes, model reasoning/raw replies,
                             conflict decisions/payoffs, and movement traces
models/<run_id>/*.json      Settings, exact model manifests, completion proofs
baselines/<control_id>/     All rule/greedy outcomes, conflict/movement logs,
                             paired input rankings and comparison CSVs
reports/                   Combined tables, 15-configuration summary,
                             1,500 per-world rows, 7,500 trial rows,
                             and 3,500 paired model-versus-rule rows
source_snapshots/          Four exact frozen code/map snapshots as ZIP files,
                             with SHA-256 hashes for every source file
provenance/                Cohort membership, decoding segments, map hashes,
                             original plans and baseline code hashes
tools/                     Integrity verification and CPU baseline replay
MANIFEST.json              SHA-256 and byte size for every published file
SHA256SUMS                 Checksums usable with sha256sum -c
```

`*_team_results.csv` is the canonical one-row-per-case/trial outcome file.
Use its **`TrueOptimal`** column for assignment makespan comparisons. The legacy
`Optimal` column is a different distance metric. `*_reasoning.csv` retains the
model's full returned replies, including initial rankings and conflict answers.
`*_conflicts.csv` identifies agreement rounds, fair-coin fallbacks, their winner,
and per-trial fallback seed. `*_moves.csv` records each agent's movement.

The raw CSV and JSON files are preserved byte-for-byte. Original metadata can
contain historical CIM host names and absolute paths; those paths are provenance,
not dependencies for accessing the data. The replay helper creates a relocated
plan instead of editing or relying on those original paths.

Model weights, environments, server logs, grid image renderings, and rejected or
superseded pilot/production traces are excluded. Selection outcomes are listed
in [model_runs.json](provenance/model_runs.json); incomplete or rejected runs are
not mixed into the 7,500 production records.

## Settings and reproducibility

Base repository commit: `45e9c2d5d997c60dc8ed1f12a032bcc1100a31d2`.
Ollama version: **0.40.16**. Context setting: **8192**.
All worlds are 20 × 20 with 2–6 agents, with a 100-step trial limit.
The fallback base seed is **20261117**, with independent SHA-256-derived
case/trial seeds. Each model trial obtained fresh rankings; only its paired
rule control reuses those rankings. Conflict prompts are text-only for every
model, including configurations that receive an image for initial rankings.

The configurations differ in modality and decoding; these are comparisons of
the recorded model configurations, not an ablation isolating architecture.
Exact settings and source profile IDs are in each model's `ENVIRONMENT.json`,
`RUN_METADATA.json`, and [model_runs.json](provenance/model_runs.json).

- Gemma4 12B: thinking disabled, output cap 2048 tokens.
- Llama3.1 8B: text only, output cap 2048 tokens.
- Llama3.2 3B: first 185 trials (worlds 1–37) retain the original uncapped
  configuration; the remaining 315 trials use a 2048-token cap after a timeout.
  Both frozen code versions and the recovery record are included.
- Ministral 3 8B and Phi-4 14B: output cap 4096 tokens; opt-in parsing accepts
  literal control characters in otherwise complete JSON strings. It does not
  repair truncated objects, synthesize answers, or reorder goal choices.
- Qwen2.5-VL 7B and Gemma4 E4B retain their original decoding profiles.

The frozen prompts retain two existing limitations: the initial ranking text
contains a fixed obstacle example rather than each world's actual obstacle list,
and its worked example does not give the optimal team assignment. Images/BFS use
the actual map. Agents may also stop on a free goal crossed en route to another
target. These behaviors were preserved in paired controls; the source snapshots
contain the exact prompts and movement code used.

### Verify the downloaded data

From the repository root, using Python 3.11 (the verification itself needs only
the standard library):

```bash
python experiments/c_coin_100x5_20261117/tools/verify_results.py
```

This checks file hashes, all 15 sets of 500 unique expected trials, reported
means/failures/optimal counts, shared references, exact saved ranking reuse,
and every fixed priority conflict decision. Alternatively, run `sha256sum -c
SHA256SUMS` from this experiment directory for file integrity alone.

### Replay the CPU baselines without model calls

On Linux, install the repository requirements into an environment, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# First verify original episode outcomes and movement trajectories.
python experiments/c_coin_100x5_20261117/tools/replay_baselines.py \
  --output /tmp/c-coin-control-validation --validate-only

# Rerun all 4,000 rule/greedy trials into a separate, new directory.
python experiments/c_coin_100x5_20261117/tools/replay_baselines.py \
  --output /tmp/c-coin-controls --workers 4
```

The helper extracts the recorded source snapshots, relocates the original
baseline runner, and blocks model calls. Each output directory must be new and
outside this published dataset. Recorded model rankings and conflicts allow
replay without installing Ollama or downloading model weights.

For a fresh model evaluation, extract the source profile named in
`provenance/model_runs.json`, install its requirements, and use that model's
recorded inference environment with fresh local service/results paths. The
evaluation command is:

```bash
python -u -m game_theory.eval_game --decider llm --game matching \
  --prompt-variant C --fallback coin --fallback-seed 20261117 \
  --config-dir configs/eval --trials 5 --yes
```

Fresh model inference can produce different replies. Use the saved replies and
rankings for exact analysis/replay of this dataset, and preserve the two recorded
decoding segments when reconstructing the Llama3.2 run.
