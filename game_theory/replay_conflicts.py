"""
replay_conflicts.py  (NEW, 12 Nov)
==================================

Quick test of the matching-prompt variants WITHOUT rerunning the maps.

It reads every conflict the LLM already played in a matching run (default:
results_eval/agent_rank_once_game_llm_matching_conflicts.csv, the 59
conflicts of the 10 random 20 x 20 maps), rebuilds the exact same game
(same players, same costs a and b, same K and D, same rotation) and asks
each agent again, independently, with the chosen prompt variant. Only
round 1 is replayed: no Option A replay, no Nash fallback, and the map is
not simulated, so the makespan is NOT measured here - only how well the
agents coordinate on the identical conflicts.

The variants (see llm_player.py):
    A   steps instead of negative payoffs, short one-line JSON answer
    B   A + the 2-agent payoffs as a grid + "the other agent sees this same table"

For each variant it writes
    results_eval/replay_<variant>.csv      one row per conflict, new answers
    results_eval/replay_summary.txt        current prompt vs every variant
                                           replayed so far (rewritten each run)

and compares with the CURRENT prompt using the round-1 answers already
logged in the conflicts CSV (no new model calls for "current").

Metrics, all on round 1 (split all / 2 agents / 3+ agents):
    agreement       every agent named the same agent
    best agent      agreed AND the agreed agent gives the best team result
    mean s          the paper's score (0 = Nash level, +1 = optimum)
    unparsed        replies that could not be read

Usage (repo root, venv active, LLM_BACKEND=ollama, OLLAMA_MODEL set):
    python -m game_theory.replay_conflicts --variants A        # ~10-15 min
    python -m game_theory.replay_conflicts --variants B        # later
    python -m game_theory.replay_conflicts --variants A B      # both
    python -m game_theory.replay_conflicts --summary-only      # just redo the summary

Nothing here changes any other file or result.
"""

import argparse
import csv
import os
import time

from game_theory.conflict_games import (TeamConflictGame, MatchingConflictGame,
                                        default_costs, EPS)

DEFAULT_CSV = "results_eval/agent_rank_once_game_llm_matching_conflicts.csv"
REPLAY_FIELDS = ["row", "Case", "Trial", "step", "decision", "kind", "subject", "players",
                 "a", "b", "rotation", "variant", "current_names", "names", "agreed",
                 "best_agent", "payoff", "O", "N", "W", "s", "current_s", "unparsed",
                 "answers_reasoning"]


def parse_subject(kind, text):
    """Goal subjects are letters ("B"); cell subjects are "(13, 17)" -> (13, 17)."""
    if kind == "cell":
        r, c = text.strip("() ").split(",")
        return (int(r), int(c))
    return text


def parse_names(text, players):
    """'A2 A1' -> [1, 0] (0-indexed agents); anything unreadable -> -1."""
    out = []
    for tok in text.split():
        try:
            out.append(int(tok.lstrip("A")) - 1)
        except ValueError:
            out.append(-1)
    return out if len(out) == len(players) else None


def rebuild_game(row, grid_size):
    """The same TeamConflictGame the run played, from one conflicts-CSV row."""
    players = [int(p.lstrip("A")) - 1 for p in row["players"].split()]
    a = dict(zip(players, map(float, row["a"].split())))
    b = dict(zip(players, map(float, row["b"].split())))
    K, D = default_costs(grid_size)
    game = TeamConflictGame(row["kind"], parse_subject(row["kind"], row["subject"]),
                            players, a, b, K, D,
                            context={"unreachable": 4 * grid_size * grid_size})
    return game


def evaluate(game, names):
    """agreed, best_agent, payoff, s, benchmarks for one round-1 name profile."""
    mgame = MatchingConflictGame(game)
    bench = mgame.benchmarks()
    names = tuple(names)
    agreed = mgame.winner(names) is not None
    payoff = mgame.payoff(names)
    best = max(mgame.payoff(tuple(w for _ in game.players)) for w in game.players)
    return {
        "agreed": int(agreed),
        "best_agent": int(agreed and payoff >= best - EPS),
        "payoff": payoff,
        "s": mgame.score(names, bench),
        "O": bench["O"], "N": bench["N"], "W": bench["W"],
    }


def fmt_names(names):
    return " ".join("?" if n < 0 else f"A{n + 1}" for n in names)


def replay(rows, variant, grid_size, out_path):
    from game_theory.llm_player import ask_agent_matching
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPLAY_FIELDS)
        w.writeheader()
        t0 = time.time()
        for k, row in enumerate(rows, 1):
            game = rebuild_game(row, grid_size)
            rotation = int(row["rotation"])
            names, reasons, unparsed = [], [], 0
            for agent in game.players:                 # independent calls
                name, info = ask_agent_matching(game, agent, rotation, variant=variant)
                names.append(name)
                unparsed += int(name is None or name < 0 or info.get("parse_failed", False))
                reasons.append(f"A{agent + 1}: {info.get('reasoning', '')}")
            names = [-1 if n is None else n for n in names]
            ev = evaluate(game, names)
            cur = parse_names(row["round1_names"], game.players)
            w.writerow({
                "row": k, **{c: row[c] for c in ("Case", "Trial", "step", "decision", "kind",
                                                 "subject", "players", "a", "b", "rotation")},
                "variant": variant, "current_names": row["round1_names"],
                "names": fmt_names(names), "unparsed": unparsed,
                "current_s": row["s"], "answers_reasoning": " | ".join(reasons), **ev,
            })
            f.flush()
            print(f"[{variant}] {k}/{len(rows)}  {row['Case']} t{row['Trial']} {row['decision']} "
                  f"{row['subject']}: {fmt_names(names)}  agreed={ev['agreed']}  "
                  f"({time.time() - t0:.0f}s)")


def stats(items):
    """items: list of dicts with n_players, agreed, best_agent, s, unparsed."""
    n = len(items)
    if n == 0:
        return "n=0"
    s_vals = [x["s"] for x in items if x["s"] is not None]
    ag = sum(x["agreed"] for x in items)
    be = sum(x["best_agent"] for x in items)
    un = sum(x["unparsed"] for x in items)
    ms = f"{sum(s_vals) / len(s_vals):+.2f}" if s_vals else "n/a"
    return (f"n={n:3d}  agreement {ag:3d} ({100 * ag / n:5.1f}%)  "
            f"best agent {be:3d} ({100 * be / n:5.1f}%)  mean s {ms}  unparsed {un}")


def summary(rows, grid_size, results_dir, csv_path):
    blocks = {}
    # current prompt: from the round-1 answers already in the conflicts CSV
    cur = []
    for row in rows:
        game = rebuild_game(row, grid_size)
        names = parse_names(row["round1_names"], game.players) or [-1] * len(game.players)
        ev = evaluate(game, names)
        ev.update(n_players=len(game.players), unparsed=int(row["parse_failures"] or 0))
        cur.append(ev)
    blocks["current (logged run)"] = cur
    for v in ("A", "B"):
        p = os.path.join(results_dir, f"replay_{v}.csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            items = []
            for r in csv.DictReader(f):
                items.append({"n_players": len(r["players"].split()),
                              "agreed": int(r["agreed"]), "best_agent": int(r["best_agent"]),
                              "s": None if r["s"] in ("", "None") else float(r["s"]),
                              "unparsed": int(r["unparsed"])})
        partial = "" if len(items) == len(rows) else f" (partial: {len(items)}/{len(rows)})"
        blocks[f"variant {v}{partial}"] = items

    lines = [f"Replay of the recorded conflicts in {csv_path}", "Round 1 only; no makespan.", ""]
    for name, items in blocks.items():
        lines.append(name)
        lines.append("  all        " + stats(items))
        lines.append("  2 agents   " + stats([x for x in items if x["n_players"] == 2]))
        lines.append("  3+ agents  " + stats([x for x in items if x["n_players"] > 2]))
        lines.append("")
    lines.append("Note: the current prompt's numbers come from the original run, whose later")
    lines.append("conflicts depended on its own earlier answers; the variants answer the very")
    lines.append("same conflicts, so the comparison is like-for-like per conflict.")
    text = "\n".join(lines)
    with open(os.path.join(results_dir, "replay_summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--variants", nargs="+", choices=["A", "B"], default=["A"])
    ap.add_argument("--conflicts", default=DEFAULT_CSV, help="matching conflicts CSV to replay")
    ap.add_argument("--grid-size", type=int, default=20, help="grid of those maps (sets K, D)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N conflicts (testing)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    with open(args.conflicts) as f:
        rows = [r for r in csv.DictReader(f) if r["decider"] == "llm"]
    if args.limit:
        rows = rows[:args.limit]
    results_dir = os.path.dirname(args.conflicts) or "."
    if not args.summary_only:
        for v in args.variants:
            replay(rows, v, args.grid_size, os.path.join(results_dir, f"replay_{v}.csv"))
    summary(rows, args.grid_size, results_dir, args.conflicts)


if __name__ == "__main__":
    main()
