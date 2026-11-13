"""
game_theory/make_eval_maps.py
=============================

NEW (11 Nov). Creates the 100 random evaluation maps used to compare with
Ismayilov et al. (arXiv 2510.23824), whose own map files are not in the repo.

Same settings as the paper (Section "Evaluation") and as the repo's own
generator, core/generate_cases.py, which this script reuses unchanged:
    * 20 x 20 grid
    * 2 to 6 agents, the same number of goals
    * 15 to 30 obstacles, never on an agent or goal cell
    * positions drawn uniformly at random

Two additions, so the set is reproducible and every map is solvable:
    1. A fixed random seed (default 2026). The paper and core/generate_cases.py
       use no seed, so the paper's exact maps cannot be recreated; with a
       seed, anyone re-running this script gets exactly these 100 maps.
    2. A solvability check. A map is redrawn if any agent cannot reach any
       goal (an obstacle wall can enclose a cell). The paper does not say
       whether it filtered maps; redrawn maps are listed in the summary.

Output (in configs/eval/, the folder name core/generate_cases.py uses):
    case_1.yaml ... case_100.yaml    the maps, same format as configs/difficult
    maps_summary.csv                 one row per map: agents, obstacles, the
                                     optimal makespan, the greedy makespan and
                                     how many redraws it needed

Usage (from the repo root; no LLM involved, takes a few seconds):
    python -m game_theory.make_eval_maps                 # seed 2026 -> configs/eval
    python -m game_theory.make_eval_maps --seed 7 --out configs/eval_seed7

Running it again with the same seed overwrites the folder with identical maps.
"""

import argparse
import csv
import os
import random

import yaml

from core.environment import GridWorld
from core.generate_cases import generate_case          # the repo's generator, unchanged
from core.utils import shortest_path_length
from core.find_optim_sol import compute_distance_matrix, find_best_assignment

DEFAULT_SEED = 2026
DEFAULT_OUT = os.path.join("configs", "eval")
NUM_MAPS = 100


def solvable(path):
    """True if every agent can reach every goal (BFS distance is finite)."""
    env = GridWorld(path)
    return all(
        shortest_path_length(a, g, env) != float("inf")
        for a in env.agents for g in env.goals
    )


def greedy_makespan(path):
    """
    Makespan of the repo's greedy baseline (agents/agent_greedy.py): every
    agent heads for its nearest goal by BFS distance, conflicts go to the
    lower agent index - the paper's "Greedy" row. No LLM. Returns (steps,
    failed).
    """
    import contextlib
    import io
    from agents.agent_greedy import run as greedy_run
    with contextlib.redirect_stdout(io.StringIO()):
        steps, _, failed, _ = greedy_run(config_path=path, log_path=os.devnull, max_steps=100)
    return steps, failed


def main():
    parser = argparse.ArgumentParser(description="Create the 100 seeded evaluation maps.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for i in range(1, NUM_MAPS + 1):
        path = os.path.join(args.out, f"case_{i}.yaml")
        redraws = 0
        while True:
            case_data = generate_case(case_num=i)   # 20x20, 2-6 agents, 15-30 obstacles
            with open(path, "w") as f:
                yaml.dump(case_data, f, sort_keys=False, default_flow_style=None)
            if solvable(path):
                break
            redraws += 1
        env = GridWorld(path)
        _, optimum = find_best_assignment(compute_distance_matrix(env))
        g_steps, g_failed = greedy_makespan(path)
        rows.append({
            "case": f"case_{i}", "agents": len(env.agents), "obstacles": len(case_data["obstacles"]),
            "optimal_makespan": optimum, "greedy_makespan": g_steps, "greedy_failed": int(g_failed),
            "redraws": redraws,
        })

    with open(os.path.join(args.out, "maps_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    ok = [r for r in rows if not r["greedy_failed"]]
    print(f"Wrote {n} maps to {args.out} (seed {args.seed})")
    print(f"  agents per map: " + ", ".join(f"{k}: {sum(r['agents'] == k for r in rows)}" for k in range(2, 7)))
    print(f"  mean optimal makespan: {sum(r['optimal_makespan'] for r in rows) / n:.2f}")
    print(f"  mean greedy makespan:  {sum(r['greedy_makespan'] for r in ok) / max(len(ok), 1):.2f}"
          f" ({n - len(ok)} greedy failures excluded)")
    print(f"  maps redrawn for solvability: {sum(r['redraws'] > 0 for r in rows)}")


if __name__ == "__main__":
    main()
