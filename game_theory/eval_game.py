"""
game_theory/eval_game.py
========================

Evaluation script for the game-theoretic agent: runs every case in
configs/difficult with one decider and writes the results to results_game/.
It is a copy of `tasks/eval_final.py`; differences are marked "# CHANGED" or
"# NEW" below:

    1. Only one agent is evaluated: game_theory.agent_rank_once_game.
    2. --decider rule|nash|llm (default llm) chooses who plays the conflict
       games. Each decider gets its own result files:
           agent_rank_once_game_<decider>_team_results.csv
           agent_rank_once_game_<decider>_conflicts.csv
           agent_rank_once_game_<decider>_moves.csv
           agent_rank_once_game_<decider>_team_summary.txt
       These 4 files are ALL a decider writes (5 Nov): the per-trial step
       logs of eval_final.py are merged into the moves CSV, one row per agent
       per step, with Case and Trial columns.
       (Files from earlier runs, e.g. agent_rank_once_game_team_l0.5_*, are
       left untouched.)
    3. Trials alternate the prompt rotation: trial 1 = rotation 0 (CLAIM
       listed first), trial 2 = rotation 1 (YIELD first), and so on.
    4. Results go to results_game/ (or $RESULTS_DIR), never results_team/.
    5. A "TrueOptimal" column (best possible makespan, as in
       core/find_optim_sol.py) is added to the results CSV.
    6. Every conflict of every trial is written to the conflicts CSV, and
       the summary reports the game-level score s next to the makespan gap.
    7. Resuming after Ctrl+C runs only the missing (case, trial) pairs.
    8. K and D scale with the grid size (--clash-factor, --delay-factor).

    9. NEW (10 Nov) --game claimyield|matching chooses the FORM of the game:
           claimyield (DEFAULT) = the original 5 Nov version, unchanged: same
                                  prompts, same file names
                                  (agent_rank_once_game_<decider>_*)
           matching             = "which agent should take it?"; files get a
                                  _matching suffix
                                  (agent_rank_once_game_<decider>_matching_*),
                                  so they never mix with the claim/yield files.
       Running without --game therefore reproduces the 5 Nov runs exactly.
   10. NEW (10 Nov) llm runs also write <key>_reasoning.csv: one row per model
       answer (each agent's goal ranking, and every conflict answer with the
       model's reasoning and full reply). Logging only; results unchanged.

   11. NEW (11 Nov) random 20 x 20 maps and the paper's metrics:
           --config-dir configs/eval   the 100 seeded maps made by
                                       game_theory/make_eval_maps.py
           --cases case_3 case_17      run only these maps
           --limit 5                   run only the first 5 maps (by number)
       Maps are run in numeric order (case_2 before case_10). With any
       folder other than configs/difficult, results go to
       results_<folder name>/ (e.g. results_eval/) unless $RESULTS_DIR is
       set, so they never mix with the 8 x 8 results in results_game/.
       New results files get a "Greedy" column (the repo's greedy baseline
       on the same map), and the summary adds the metrics of Ismayilov et
       al.: mean makespan, mean optimal makespan, performance gap, and the
       greedy baseline. Definitions: game_theory/README.md, "Metrics".

   12. NEW (12 Nov) --prompt-variant current|A|B|C (matching form only; C: 15 Nov):
       the wording of the matching prompt (see llm_player.py). "current" is
       the default and keeps the 10 Nov prompt and file names; A and B add
       a _A / _B suffix (agent_rank_once_game_llm_matching_A_*).
       Quick comparison without rerunning maps: game_theory/replay_conflicts.py.

Removed with the old global game: --payoff, --llm-weight, --replan
(re-planning is now always on and is itself a game).

Usage (from the repo root, with Ollama running and LLM_BACKEND=ollama):
    python -m game_theory.eval_game --decider rule
    python -m game_theory.eval_game --decider nash
    python -m game_theory.eval_game --decider llm
    python -m game_theory.eval_game --decider llm --game matching    # NEW
    python -m game_theory.eval_game --decider llm --game matching \
        --config-dir configs/eval --limit 5                            # NEW 11 Nov
"""

import csv
import os
import argparse
from core.environment import GridWorld
from core.plot import plot_grid_unassigned_labeled
from core.find_optim_sol import compute_distance_matrix, find_best_assignment   # NEW

from game_theory.conflict_games import DECIDERS, FORMS, CLASH_FACTOR, YIELD_DELAY_FACTOR  # NEW
from game_theory import agent_rank_once_game                                     # NEW

# Constants
IMAGE_PATH = "data/grid.png"
# CHANGED: separate output folder (was "results_team")
OUTPUT_DIR = os.getenv("RESULTS_DIR", "results_game")
VISUALIZE = False  # set to True if you want to save visuals
TRIALS_PER_CASE = 2

AGENT_NAME = "agent_rank_once_game"

# NEW: columns of the conflicts CSV (one row per conflict)
CONFLICT_FIELDS = [
    "Case", "Trial", "step", "decision", "kind", "subject", "players", "a", "b",
    "decider", "rotation", "round1_outcome", "round1_actions", "final_actions", "round1_payoff",
    "final_payoff", "O", "N", "W", "s", "equilibrium_gap", "coordinated",
    "clash_rounds", "nash_fallback", "parse_failures", "mixed_checked",
]

# NEW (10 Nov): columns of the conflicts CSV in the matching form - the same
# columns plus the names each agent gave. The claim/yield CSV keeps the
# original columns above, so old and new claim/yield files stay compatible.
MATCHING_CONFLICT_FIELDS = CONFLICT_FIELDS[:12] + ["round1_names", "final_names", "agreed"] + CONFLICT_FIELDS[12:]

# NEW (10 Nov): columns of the reasoning CSV - one row per model answer:
# each agent's Phase 1 ranking (decision = "ranking") and every conflict
# answer (decision = goal / cell / replan; round 2 = the Option A replay).
# "reply" is the model's full raw reply for conflicts and the ranking's
# one-line summary for rankings.
REASONING_FIELDS = ["Case", "Trial", "step", "decision", "kind", "subject", "agent",
                    "round", "rotation", "answer", "reasoning", "reply"]

# NEW: columns of the moves CSV (the agent's step log + Case and Trial)
MOVE_FIELDS = ["Case", "Trial", "step", "agent_id", "position_before",
               "position_after", "chosen_direction", "explanation"]

# NEW: a conflict whose two best pure equilibria differ by less than this many
# steps counts as "hard" (pure symmetry-breaking) in the summary.
HARD_GAP = 2


def task_key(decider, game_form="claimyield", prompt_variant="current"):
    """
    NEW: file-name prefix for a run.
        ('llm', 'claimyield') -> 'agent_rank_once_game_llm'           (original name)
        ('llm', 'matching')   -> 'agent_rank_once_game_llm_matching'  (NEW 10 Nov)
    """
    key = f"{AGENT_NAME}_{decider}"
    key = key if game_form == "claimyield" else f"{key}_{game_form}"
    return key if prompt_variant == "current" else f"{key}_{prompt_variant}"   # NEW 12 Nov


def true_optimum(config_path):
    """NEW: best achievable makespan (same as core/find_optim_sol.py)."""
    env = GridWorld(config_path)
    _, cost = find_best_assignment(compute_distance_matrix(env))
    return cost


def greedy_steps(config_path):
    """
    NEW (11 Nov): makespan of the repo's greedy baseline on this map
    (agents/agent_greedy.py: nearest goal by BFS, lower index wins), the
    paper's "Greedy" row. No LLM; 100 = the greedy run hit the step limit.
    """
    from game_theory.make_eval_maps import greedy_makespan
    steps, _ = greedy_makespan(config_path)
    return steps


def case_number(name):
    """NEW (11 Nov): 'case_10_insane' -> 10, for numeric ordering."""
    parts = name.split("_")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10**9


def evaluate_team(key, run_fn, cases, decider, clash_factor, delay_factor,
                  game_form="claimyield", prompt_variant="current"):
    """
    Run the missing trials of every case and append one CSV row per trial,
    one conflicts-CSV row per conflict and one moves-CSV row per agent move.
    """
    print(f"\n=== Evaluating Structured Cases: {key} ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, f"{key}_team_results.csv")
    conflicts_path = os.path.join(OUTPUT_DIR, f"{key}_conflicts.csv")
    moves_path = os.path.join(OUTPUT_DIR, f"{key}_moves.csv")          # NEW
    # NEW 10 Nov: the reasoning log, only for the llm decider (rule and nash
    # never ask the model during conflicts)
    reasoning_path = os.path.join(OUTPUT_DIR, f"{key}_reasoning.csv") if decider == "llm" else None
    if reasoning_path:
        new_file = not os.path.exists(reasoning_path)
        freason = open(reasoning_path, mode='a', newline='')
        rwriter = csv.DictWriter(freason, fieldnames=REASONING_FIELDS, extrasaction="ignore")
        if new_file:
            rwriter.writeheader()
    summary_path = os.path.join(OUTPUT_DIR, f"{key}_team_summary.txt")

    write_header = not os.path.exists(log_path)
    # NEW (11 Nov): new results files get a "Greedy" column; an existing file
    # keeps the columns it was created with, so old files stay consistent.
    if write_header:
        with_greedy = True
    else:
        with open(log_path, newline="") as fh:
            with_greedy = "Greedy" in (fh.readline().strip().split(","))
    write_conflict_header = not os.path.exists(conflicts_path)
    write_move_header = not os.path.exists(moves_path)

    with open(log_path, mode='a', newline='') as f, \
         open(conflicts_path, mode='a', newline='') as fc, \
         open(moves_path, mode='a', newline='') as fm:
        writer = csv.writer(f)
        fields = CONFLICT_FIELDS if game_form == "claimyield" else MATCHING_CONFLICT_FIELDS
        cwriter = csv.DictWriter(fc, fieldnames=fields, extrasaction="ignore")
        mwriter = csv.DictWriter(fm, fieldnames=MOVE_FIELDS, extrasaction="ignore")
        if write_header:
            # CHANGED: "TrueOptimal" appended at the end; NEW 11 Nov: "Greedy"
            writer.writerow(["Case", "Trial", "Steps", "Optimal", "Failed", "Collisions", "TrueOptimal"]
                            + (["Greedy"] if with_greedy else []))
        if write_conflict_header:
            cwriter.writeheader()
        if write_move_header:
            mwriter.writeheader()

        for case_name, (case_config_path, trials_left) in cases.items():
            best = true_optimum(case_config_path)
            greedy = greedy_steps(case_config_path) if with_greedy else None   # NEW 11 Nov
            for trial in trials_left:
                rotation = (trial - 1) % 2                       # NEW
                print(f"\n--- {case_name}, Trial {trial} (rotation {rotation}) ---")

                if VISUALIZE:
                    env = GridWorld(case_config_path)
                    plot_grid_unassigned_labeled(env, image_path=f"data/{key}_{case_name}_t{trial}.png")

                # CHANGED (5 Nov): no per-trial log files; rows are collected
                # in these lists and written to the moves/conflicts CSVs.
                conflict_log = []                                # NEW
                move_log = []                                    # NEW
                reasoning_log = [] if reasoning_path else None   # NEW 10 Nov

                steps, optimal, failed, collisions = run_fn(
                    config_path=case_config_path,
                    image_path=IMAGE_PATH,
                    max_steps=100,
                    decider=decider,                 # NEW
                    rotation=rotation,               # NEW
                    clash_factor=clash_factor,       # NEW
                    delay_factor=delay_factor,       # NEW
                    conflict_log=conflict_log,       # NEW
                    move_log=move_log,               # NEW
                    game_form=game_form,             # NEW 10 Nov
                    prompt_variant=prompt_variant,   # NEW 12 Nov
                    reasoning_log=reasoning_log,     # NEW 10 Nov
                )

                writer.writerow([case_name, trial, steps, optimal, int(failed), collisions, best]
                                + ([greedy] if with_greedy else []))
                for rec in conflict_log:
                    cwriter.writerow({"Case": case_name, "Trial": trial, **rec})
                for row in move_log:
                    mwriter.writerow({"Case": case_name, "Trial": trial, **row})
                if reasoning_path:                               # NEW 10 Nov
                    for row in reasoning_log:                    # rankings
                        rwriter.writerow({"Case": case_name, "Trial": trial, **row})
                    for rec in conflict_log:                     # conflict answers
                        for call in rec.get("llm_calls", []):
                            rwriter.writerow({"Case": case_name, "Trial": trial,
                                              "step": rec["step"], "decision": rec["decision"],
                                              "kind": rec["kind"], "subject": rec["subject"], **call})
                    freason.flush()
                f.flush()
                fc.flush()
                fm.flush()

    if reasoning_path:
        freason.close()
    write_summary(log_path, conflicts_path, summary_path)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _steps(x):
    """NEW (11 Nov): a number of steps to 2 decimals, or n/a."""
    return "n/a" if x is None else f"{x:.2f}"


def _gap(finished):
    """NEW (11 Nov): mean makespan - mean optimal makespan, over the same trials."""
    if not finished:
        return None
    return _mean([int(r["Steps"]) for r in finished]) - _mean([int(r["TrueOptimal"]) for r in finished])


def _gap_pct(finished):
    """NEW (11 Nov): the performance gap relative to the mean optimal makespan."""
    if not finished:
        return None
    return _gap(finished) / _mean([int(r["TrueOptimal"]) for r in finished])


def _greedy_line(rows):
    """
    NEW (11 Nov): mean greedy makespan on the same maps, one value per map
    (greedy has no LLM, so it does not change between trials). Maps where
    greedy hit the step limit (100) are left out and counted.
    """
    if not rows or "Greedy" not in rows[0]:
        return "  Greedy baseline: n/a (results file has no Greedy column)"
    per_map = {r["Case"]: int(r["Greedy"]) for r in rows if r.get("Greedy") not in (None, "")}
    ok = [v for v in per_map.values() if v < 100]
    return (f"  Greedy baseline, mean makespan on these maps: {_steps(_mean(ok))}"
            f" ({len(per_map) - len(ok)} greedy failures left out)")


def _num(x):
    return "n/a" if x is None else f"{x:+.2f}"


def write_summary(log_path, conflicts_path, summary_path):
    """
    NEW: summary of the WHOLE results and conflicts CSVs (correct after
    resuming).

    Task level  : optimal trials, average makespan gap vs TrueOptimal
                  (failed trials excluded and counted separately), failures,
                  agents made to wait.
    Game level  : number of conflicts; mean s (round 1, conflicts with no
                  headroom excluded) overall, by decision point, and for easy
                  vs hard conflicts; coordination rate (exactly one claimer in
                  round 1); clash and all-yield rates; Nash fallbacks;
                  unparsed replies.
    """
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    conflicts = []
    if os.path.exists(conflicts_path):
        with open(conflicts_path, newline="") as f:
            conflicts = list(csv.DictReader(f))

    finished = [r for r in rows if int(r["Failed"]) == 0]
    gaps = [(int(r["Steps"]) - int(r["TrueOptimal"])) / int(r["TrueOptimal"]) for r in finished]
    n_optimal = sum(1 for r in finished if int(r["Steps"]) == int(r["TrueOptimal"]))

    def scores(rows_):
        return [float(c["s"]) for c in rows_ if c["s"] not in ("", "None")]

    by = lambda key, val: [c for c in conflicts if c[key] == val]
    easy = [c for c in conflicts if float(c["equilibrium_gap"]) >= HARD_GAP]
    hard = [c for c in conflicts if float(c["equilibrium_gap"]) < HARD_GAP]
    n = len(conflicts)

    lines = [
        f"Evaluated {len(rows)} trials across {len({r['Case'] for r in rows})} scenarios",
        "",
        "Task level",
        f"  Optimal (Steps == TrueOptimal): {n_optimal}/{len(rows)}",
        f"  Average optimality gap (finished trials): {_pct(_mean(gaps))}",
        f"  Failures (max steps): {len(rows) - len(finished)}/{len(rows)}",
        f"  Agents made to wait by cell conflicts: {sum(int(r['Collisions']) for r in rows)}",
        "",
        # NEW (11 Nov): the metrics of Ismayilov et al., over finished trials
        "Paper metrics (Ismayilov et al.; finished trials only, see README 'Metrics')",
        f"  Mean makespan: {_steps(_mean([int(r['Steps']) for r in finished]))}",
        f"  Mean optimal makespan: {_steps(_mean([int(r['TrueOptimal']) for r in finished]))}",
        f"  Performance gap (mean makespan - mean optimal): "
        f"{_steps(_gap(finished))} steps ({_pct(_gap_pct(finished))})",
        _greedy_line(rows),
        "",
        "Game level (round 1, s: 0 = Nash, +1 = optimum, -1 = worst)",
        f"  Conflicts: {n} (goal {len(by('decision', 'goal'))}, cell {len(by('decision', 'cell'))}, "
        f"replan {len(by('decision', 'replan'))})",
        f"  Mean s, all: {_num(_mean(scores(conflicts)))}  (scored: {len(scores(conflicts))})",
        f"  Mean s, goal / cell / replan: {_num(_mean(scores(by('decision', 'goal'))))} / "
        f"{_num(_mean(scores(by('decision', 'cell'))))} / {_num(_mean(scores(by('decision', 'replan'))))}",
        f"  Mean s, easy (gap >= {HARD_GAP}) / hard: {_num(_mean(scores(easy)))} ({len(easy)}) / "
        f"{_num(_mean(scores(hard)))} ({len(hard)})",
        f"  Coordination rate (exactly one claim): {_pct(_mean([int(c['coordinated']) for c in conflicts]))}",
        f"  Clash rate (2+ claims in round 1): {_pct(_mean([int(c['round1_outcome'] in ('clash', 'disagree_clash')) for c in conflicts]))}",
        f"  All-yield rate (nobody claims in round 1): {_pct(_mean([int(c['round1_outcome'] in ('none', 'disagree_none')) for c in conflicts]))}",
        f"  Nash fallbacks after the replay: {sum(int(c['nash_fallback']) for c in conflicts)}",
        f"  Unparsed LLM replies (counted as YIELD): {sum(int(c['parse_failures']) for c in conflicts)}"
        if not (conflicts and "agreed" in conflicts[0]) else
        f"  Unparsed LLM replies (counted as naming nobody): {sum(int(c['parse_failures']) for c in conflicts)}",
    ]
    # NEW (10 Nov): matching-form runs also report how often everyone named
    # the same agent ("coordination rate" above is the same number there;
    # in the matching form a "clash" = disagreement where someone named
    # itself, "all-yield" = disagreement where nobody did).
    if conflicts and "agreed" in conflicts[0]:
        lines.insert(lines.index("Game level (round 1, s: 0 = Nash, +1 = optimum, -1 = worst)") + 1,
                     f"  Form: matching - agreement rate (everyone named the same agent): "
                     f"{_pct(_mean([int(c['agreed']) for c in conflicts]))}")
    print("\n".join(lines))
    with open(summary_path, "w") as f:
        f.writelines(line + "\n" for line in lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the game-theoretic agent on coordination scenarios.")
    # CHANGED: no --agents flag (one agent); NEW --decider and cost factors
    parser.add_argument("--decider", choices=DECIDERS, default="llm",
                        help="who plays the conflict games: rule (original), nash (solver) or llm")
    parser.add_argument("--game", choices=FORMS, default="claimyield",
                        help="form of the conflict game: claimyield (default, original 5 Nov) "
                             "or matching (NEW 10 Nov: 'which agent should take it?')")
    parser.add_argument("--prompt-variant", choices=["current", "A", "B", "C"], default="current",
                        help="NEW 12 Nov: wording of the matching prompt (matching form only)")
    parser.add_argument("--clash-factor", type=float, default=CLASH_FACTOR,
                        help="K = factor x grid size (default 0.75 -> 6 on 8x8)")
    parser.add_argument("--delay-factor", type=float, default=YIELD_DELAY_FACTOR,
                        help="D = factor x grid size (default 0.375 -> 3 on 8x8)")
    parser.add_argument("--config-dir", type=str, default="configs/difficult", help="Directory containing configuration YAML files")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations for each scenario")
    parser.add_argument("--trials", type=int, default=TRIALS_PER_CASE, help="Number of trials per case (alternating rotations)")
    parser.add_argument("--yes", action="store_true", help="NEW: skip the 'Press Enter' prompt")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="NEW 11 Nov: run only these maps, e.g. --cases case_3 case_17")
    parser.add_argument("--limit", type=int, default=None,
                        help="NEW 11 Nov: run only the first N maps (numeric order)")
    args = parser.parse_args()

    # NEW (11 Nov): maps outside configs/difficult get their own results
    # folder (e.g. configs/eval -> results_eval/) unless $RESULTS_DIR is set.
    if "RESULTS_DIR" not in os.environ and \
            os.path.normpath(args.config_dir) != os.path.normpath("configs/difficult"):
        OUTPUT_DIR = "results_" + os.path.basename(os.path.normpath(args.config_dir))

    VISUALIZE = args.visualize
    os.makedirs("data", exist_ok=True)  # agents save grid.png here
    TRIALS_PER_CASE = args.trials
    if args.prompt_variant != "current" and args.game != "matching":
        raise ValueError("--prompt-variant A/B/C needs --game matching")
    key = task_key(args.decider, args.game, args.prompt_variant)

    # Load all YAML cases (CHANGED 11 Nov: numeric order, case_2 before case_10)
    names = sorted((os.path.splitext(f)[0] for f in os.listdir(args.config_dir) if f.endswith(".yaml")),
                   key=lambda n: (case_number(n), n))
    cases = {n: os.path.join(args.config_dir, n + ".yaml") for n in names}
    if not cases:
        raise ValueError(f"No YAML files found in directory: {args.config_dir}")
    # NEW (11 Nov): run only some maps
    if args.cases:
        unknown = [c for c in args.cases if c not in cases]
        if unknown:
            raise ValueError(f"Unknown cases {unknown}; names look like {names[:3]}")
        cases = {c: cases[c] for c in names if c in args.cases}
    if args.limit is not None:
        cases = dict(list(cases.items())[:args.limit])

    print(f"Found {len(cases)} cases in {args.config_dir}")
    print("Cases:", list(cases.keys())[:10])
    print(f"Decider: {args.decider}, game: {args.game}  ->  {key}   (results in {OUTPUT_DIR}/)")

    completed = set()
    log_path = os.path.join(OUTPUT_DIR, f"{key}_team_results.csv")
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                completed.add((row["Case"], int(row["Trial"])))
    print(f"Completed trials for {key}: {len(completed)}")

    if not args.yes:
        input("Press Enter to continue...")

    todo = {}
    for case, path in cases.items():
        missing = [t for t in range(1, TRIALS_PER_CASE + 1) if (case, t) not in completed]
        if missing:
            todo[case] = (path, missing)

    print(f"\n{key} has {len(todo)} cases left to run.")
    if not todo:
        print(f"✅ Skipping {key} — all trials completed.")
        write_summary(log_path, os.path.join(OUTPUT_DIR, f"{key}_conflicts.csv"),
                      os.path.join(OUTPUT_DIR, f"{key}_team_summary.txt"))
    else:
        evaluate_team(key, agent_rank_once_game.run, todo, args.decider,
                      args.clash_factor, args.delay_factor, args.game, args.prompt_variant)
