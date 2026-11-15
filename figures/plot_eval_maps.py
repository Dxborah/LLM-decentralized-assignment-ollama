"""
figures/plot_eval_maps.py  (NEW, 15 Nov)
========================================

Draws the random 20 x 20 maps (configs/eval/) without running anything.
For each map, three panels side by side:

    Optimal      the best assignment (brute force over every way of giving
                 each agent its own goal), drawn as each agent's BFS path
    Best prompt  the paths the agents actually took in a logged run
                 (default: the LLM with prompt variant C, trial 1)
    Nash         the paths taken with the Nash solver deciding conflicts

The title of each panel gives its makespan. Agents are numbered circles
(1, 2, ...) at their start, goals are lettered squares (A, B, ...), black
cells are obstacles, and each agent's path is drawn in its own colour. A
small cross marks where an agent ended.

Usage (repo root):
    python3 figures/plot_eval_maps.py                          # every map the best-prompt run has
    python3 figures/plot_eval_maps.py --cases case_5 case_27   # only some maps
    python3 figures/plot_eval_maps.py --trial 2
    python3 figures/plot_eval_maps.py --results-dir results_eval_qwen3.8_27b   # another run

Output: <results-dir>/maps/<case>.png, plus <results-dir>/maps/overview.png
(optimal vs best prompt, at most 20 maps per overview page:
overview.png, overview_2.png, ...). Reads only the config YAMLs and the
moves CSVs; it never changes a result file.
"""

import argparse
import ast
import csv
import itertools
import os
from collections import deque, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

CONFIG_DIR = "configs/eval"
RESULTS_DIR = "results_eval"
OUT_DIR = "results_eval/maps"   # set from --results-dir in main()
RUNS = [  # (panel title, moves CSV key) - the optimal panel is computed, not read
    ("Best prompt", "agent_rank_once_game_llm_matching_C"),
    ("Nash", "agent_rank_once_game_nash_matching"),
]
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#17becf"]


def load_map(case):
    with open(os.path.join(CONFIG_DIR, f"{case}.yaml")) as f:
        cfg = yaml.safe_load(f)
    return (cfg["size"], [tuple(a) for a in cfg["agents"]], [tuple(g) for g in cfg["goals"]],
            {tuple(o) for o in cfg["obstacles"]})


def bfs_path(start, goal, size, obstacles):
    prev = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        if p == goal:
            path = []
            while p is not None:
                path.append(p)
                p = prev[p]
            return path[::-1]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (p[0] + dr, p[1] + dc)
            if 0 <= n[0] < size and 0 <= n[1] < size and n not in obstacles and n not in prev:
                prev[n] = p
                q.append(n)
    return None


def optimal_paths(size, agents, goals, obstacles):
    """Brute-force min-max assignment, same rule as core/find_optim_sol.py."""
    paths = [[bfs_path(a, g, size, obstacles) for g in goals] for a in agents]
    best, best_cost = None, float("inf")
    for perm in itertools.permutations(range(len(goals))):
        lens = [len(paths[i][j]) - 1 if paths[i][j] else float("inf") for i, j in enumerate(perm)]
        if max(lens) < best_cost:
            best, best_cost = perm, max(lens)
    return [paths[i][j] for i, j in enumerate(best)], best_cost


def logged_paths(key, case, trial, n_agents):
    """Each agent's positions over time from a moves CSV; None if not logged."""
    path_file = os.path.join(RESULTS_DIR, f"{key}_moves.csv")
    if not os.path.exists(path_file):
        return None, None
    steps = defaultdict(list)
    with open(path_file) as f:
        for r in csv.DictReader(f):
            if r["Case"] == case and r["Trial"] == str(trial):
                steps[int(r["agent_id"])].append((int(r["step"]),
                                                  ast.literal_eval(r["position_before"]),
                                                  ast.literal_eval(r["position_after"])))
    if not steps:
        return None, None
    paths, makespan = [], 0
    for aid in range(1, n_agents + 1):
        rows = sorted(steps.get(aid, []))
        if not rows:
            paths.append([])
            continue
        # positions at the start of each step, then the last position reached
        # (position_after can be the blocked target when an agent had to wait)
        p = [before for _, before, _ in rows] + [rows[-1][2]]
        paths.append(p)
        makespan = max(makespan, rows[-1][0] + 1)
    return paths, makespan


def draw(ax, size, agents, goals, obstacles, paths, title):
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    for x in range(size + 1):
        ax.axhline(x, color="#dddddd", lw=0.5, zorder=0)
        ax.axvline(x, color="#dddddd", lw=0.5, zorder=0)
    for r, c in obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#222222", zorder=1))
    for j, (r, c) in enumerate(goals):
        ax.add_patch(plt.Rectangle((c + 0.08, r + 0.08), 0.84, 0.84, color="#f2c6c6",
                                   ec="#c0392b", lw=1.2, zorder=2))
        ax.text(c + 0.5, r + 0.5, chr(65 + j), ha="center", va="center",
                fontsize=8, fontweight="bold", color="#c0392b", zorder=5)
    for i, p in enumerate(paths or []):
        if not p:
            continue
        col = COLORS[i % len(COLORS)]
        # small offset per agent so overlapping paths stay visible
        off = (i - (len(agents) - 1) / 2) * 0.08
        xs = [c + 0.5 + off for _, c in p]
        ys = [r + 0.5 + off for r, _ in p]
        ax.plot(xs, ys, color=col, lw=2, alpha=0.85, zorder=3)
        ax.plot(xs[-1], ys[-1], marker="x", color=col, ms=7, mew=2, zorder=4)
    for i, (r, c) in enumerate(agents):
        ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.42, color=COLORS[i % len(COLORS)], zorder=4))
        ax.text(c + 0.5, r + 0.5, str(i + 1), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(0, size + 1, 5))
    ax.set_yticks(range(0, size + 1, 5))
    ax.tick_params(labelsize=7)


def main():
    global RESULTS_DIR, CONFIG_DIR, OUT_DIR
    ap = argparse.ArgumentParser(description="Draw the eval maps with optimal and logged paths.")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="default: every map in the best-prompt run's results")
    ap.add_argument("--trial", type=int, default=1, help="which logged trial to draw")
    ap.add_argument("--results-dir", default=RESULTS_DIR,
                    help="folder with the *_moves.csv files (default results_eval)")
    ap.add_argument("--config-dir", default=CONFIG_DIR)
    args = ap.parse_args()
    RESULTS_DIR, CONFIG_DIR = args.results_dir, args.config_dir
    OUT_DIR = os.path.join(RESULTS_DIR, "maps")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.cases:
        moves = os.path.join(RESULTS_DIR, f"{RUNS[0][1]}_team_results.csv")
        with open(moves) as f:
            found = []
            for r in csv.DictReader(f):
                if r["Case"] not in found:
                    found.append(r["Case"])
        args.cases = sorted(found, key=lambda c: int(c.split("_")[-1]))

    overview = []
    for case in args.cases:
        size, agents, goals, obstacles = load_map(case)
        opt, opt_cost = optimal_paths(size, agents, goals, obstacles)
        panels = [(f"Optimal: {opt_cost} steps", opt)]
        for label, key in RUNS:
            paths, mk = logged_paths(key, case, args.trial, len(agents))
            panels.append((f"{label}: {mk} steps" if paths else f"{label}: not run", paths))
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.3))
        for ax, (title, paths) in zip(axes, panels):
            draw(ax, size, agents, goals, obstacles, paths, title)
        fig.suptitle(f"{case}  ({len(agents)} agents, {len(obstacles)} obstacles, trial {args.trial})",
                     fontsize=12)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"{case}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)
        overview.append((case, size, agents, goals, obstacles, panels[:2]))

    # overview: one row per map, optimal vs best prompt, 20 maps per page
    for page, start in enumerate(range(0, len(overview), 20), 1):
        chunk = overview[start:start + 20]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(8, 4 * len(chunk)), squeeze=False)
        for row, (case, size, agents, goals, obstacles, panels) in zip(axes, chunk):
            for ax, (title, paths) in zip(row, panels):
                draw(ax, size, agents, goals, obstacles, paths, f"{case} - {title}")
        fig.tight_layout()
        out = os.path.join(OUT_DIR, "overview.png" if page == 1 else f"overview_{page}.png")
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
