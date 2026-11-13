"""
game_theory/agent_rank_once_game.py
===================================

Game-theoretic version of `agents/agent_rank_once_bfs.py`, where every
conflict decision is settled by the collaborative claim/yield game in
game_theory/conflict_games.py (design doc: "Game-theoretic conflict
resolution: redesign").

This file is a COPY of agent_rank_once_bfs.py. Phase 1 (one LLM ranking per
agent, same prompt, same image call), the BFS movement, goal claiming and the
return value are unchanged. Every rule-based decision is replaced; each spot
is marked with "# ORIGINAL:" showing the code it replaced:

    where                      ORIGINAL rule                   NOW
    -------------------------  ------------------------------  ---------------------------------
    select_target()            random goal if JSON fails       distance-order fallback; the agent
                                                               still plays the games below
    Phase 2                    lowest agent index wins a goal  Claim/Yield team game per
                                                               contested goal
    Collision Resolution       lowest agent index moves        Claim/Yield team game per
                                                               contested cell
    start of each step         stranded agent stands still     the goal game re-played over the
                                                               free goals (always on)

Which FORM of the game is played is the `game_form` argument (NEW 10 Nov):
    claimyield  DEFAULT - the original 5 Nov version, unchanged: each agent
                answers CLAIM or YIELD (anti-coordination)
    matching    each agent names the agent that should take the goal/cell;
                any disagreement is a failure (coordination / matching game)
Both forms resolve the same conflicts with the same payoffs and Option A;
only the question asked to the model differs. Calling run() without
game_form reproduces the 5 Nov behaviour exactly.

Who plays the games is the `decider` argument:
    rule  the original lowest-index rule (repo baseline)
    nash  the best pure Nash equilibrium (game-theoretic benchmark)
    llm   each player's own model call, no communication (the experiment)
Note: the re-planning step and the distance fallback apply to ALL deciders,
so the three are compared on equal terms. The untouched original agent
(agents/agent_rank_once_bfs.py) remains the exact paper baseline.

Miscoordination (Option A): see conflict_games.play_conflict. A round with a
clash OR with nobody claiming is replayed once with the action order flipped,
then the Nash solver decides. Each failed round costs a waiting step
(`hold`: the claimers after a clash, everyone after an all-YIELD round). In a
cell conflict nobody moves on that step, and the agent the replay/Nash picked
moves first on the next step if the same conflict comes back.

Pipeline of one episode
-----------------------
    Phase 1  every agent asks the LLM once for a ranking of the goals
    Phase 2  contested first choices -> one goal game each -> one goal per agent
    Phase 3  every step: re-play the goal game for stranded agents; each agent
             proposes its BFS move (UNCHANGED); cell clashes -> one cell game
             each; move; agents that land on a goal stop

Reasoning log (NEW 10 Nov)
--------------------------
Every model answer is kept for analysis, without changing any decision:
    * each agent's Phase 1 ranking reasoning -> `reasoning_log` (if given)
    * each conflict answer -> the "llm_calls" list of its conflict record
eval_game.py writes both to results_game/<key>_reasoning.csv.

Output files
------------
Run on its own (python -m game_theory.agent_rank_once_game), it writes the
usual step log to `log_path` plus <log_path without .csv>_conflicts.json.
Run from eval_game.py, it writes NO files: the caller passes `move_log` and
`conflict_log` lists, which receive the step-log rows and conflict records,
and eval_game.py writes one moves CSV and one conflicts CSV per decider
(5 Nov: fewer files per test).
"""

import time
import json
import csv
import os
from core.environment import GridWorld
from core.prompt import build_target_ranking_prompt
from core.request import send_image_to_model_openai_logprobs
from core.plot import plot_grid_unassigned_labeled
from core.utils import shortest_path_length, select_direction_opt
import re
import argparse

# NEW: the game-theory layer
from game_theory.conflict_games import (
    TeamConflictGame,
    play_conflict,
    default_costs,
    DECIDERS,
    FORMS,
    CLASH_FACTOR,
    YIELD_DELAY_FACTOR,
    CLAIM,
)

# ORIGINAL: import random
#   `random` was only used for the random fallback goal, which is gone.


def parse_ranking_response(text):
    """
    Extract the JSON ranking from the LLM reply. UNCHANGED from the original.
    Returns ([], "", "") and writes the reply to fails.txt if parsing fails.
    """
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        json_str = match.group(1) if match else text[text.index("{"):]
        parsed = json.loads(json_str)

        reasoning = parsed.get("reasoning", "").strip()
        explanation = parsed.get("explanation", "").strip()
        ranking = [g.strip().upper() for g in parsed.get("ranking", []) if isinstance(g, str)]
        return ranking, explanation, reasoning
    except Exception as e:
        print(f"Failed to parse ranking JSON: {e}")
        with open("fails.txt", "a") as f:
            f.write(f"[Ranking Parsing Error] {e}\n{text}\n\n")
        return [], "", ""


def select_target(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,
    grid_size,
    obstacles,
    memory,
    visits,
    agent_targets,
    target_memory,
    image_path,
    step,
    distances
):
    """
    Ask the LLM for this agent's ranking of the goals.

    Same prompt and same model call as the original. The ONLY change: an
    empty ranking (reply could not be parsed, or every letter was invalid)
    stays empty here; preferences() below then falls back to distance order.
    """
    prompt = build_target_ranking_prompt(
        agent_id=agent_id,
        agent_pos=agent_pos,
        goal_positions=goal_positions,
        other_agents=other_agents,
        grid_size=grid_size,
        obstacles=obstacles,
        memory=memory,
        visits=visits,
        agent_targets=agent_targets,
        target_memory=target_memory,
        distances=distances
    )
    time.sleep(0.5)
    # The model name below is what the original passes; with LLM_BACKEND=ollama
    # core/request.py routes this image call to your OLLAMA_MODEL instead.
    response, _ = send_image_to_model_openai_logprobs(image_path, prompt, model="gpt-4.1", temperature=0.0000001)

    raw_ranking, explanation, reasoning = parse_ranking_response(response)

    # Filter invalid goals (UNCHANGED)
    valid_goals = set(chr(65 + i) for i in range(len(goal_positions)) if goal_positions[i] is not None)
    seen = set()
    filtered_ranking = []

    for g in raw_ranking:
        if g in valid_goals and g not in seen:
            filtered_ranking.append(g)
            seen.add(g)

    # ORIGINAL:
    #   # Fallback: choose random goal if nothing valid
    #   if not filtered_ranking and valid_goals:
    #       fallback = random.choice(list(valid_goals))
    #       filtered_ranking = [fallback]
    #
    # Why it was removed: a one-goal random ranking gave the agent no second
    # choice; if it lost that goal it had no target and never moved.
    if not filtered_ranking and valid_goals:
        print(f"⚠️ Agent {agent_id} had no valid ranking. Falling back to distance order.")

    print(f"Agent {agent_id} ranking: {filtered_ranking}")
    if reasoning:
        print(f"Reasoning: {reasoning}")
    if explanation:
        print(f"Summary: {explanation}")

    if filtered_ranking:
        top = filtered_ranking[0]
        target_memory.append((step, top, explanation))
        if len(target_memory) > 5:
            target_memory[:] = target_memory[-5:]

    return filtered_ranking, explanation, reasoning


# ORIGINAL: def resolve_conflicts(agent_rankings, active_agents): ...
#   Gave a contested goal to the lowest agent index; the losers moved to
#   their next-ranked goal, repeated until no conflicts were left.
#   Replaced by assign_goals() below, which plays one game per conflict.


# ===========================================================================
# NEW helpers
# ===========================================================================
def _letter(i):
    return chr(65 + i)


def _goal_pos(env, letter):
    return env.goals[ord(letter) - 65]


def _dist(env, start, letter, unreachable):
    """BFS distance from `start` to goal `letter` (unreachable -> big number)."""
    pos = _goal_pos(env, letter)
    if start is None or pos is None:
        return unreachable
    d = shortest_path_length(start, pos, env)
    return unreachable if d == float("inf") else d


def preferences(env, agent_pos, ranking, allowed, unreachable):
    """
    One agent's full preference list over the `allowed` goal letters.

    LLM ranking first (in its order, allowed goals only), then any allowed
    goal the LLM did not rank, by BFS distance. With an empty ranking this is
    pure distance order: the parse-failure fallback (replaces the random goal).
    """
    ranked = [g for g in ranking if g in allowed]
    rest = sorted((g for g in allowed if g not in ranked),
                  key=lambda g: (_dist(env, agent_pos, g, unreachable), g))
    return ranked + rest


def assign_goals(env, agent_positions, candidates, prefs, decider, rotation,
                 K, D, unreachable, hold, conflict_log, step, decision,
                 game_form="claimyield", prompt_variant="current"):
    """
    Give each candidate agent one goal, settling every conflict with a game.

    REPLACES resolve_conflicts(): same loop structure (everyone starts at its
    first choice; while 2+ agents share a choice, settle it; whoever gives
    way moves to its next choice), but the "who gives way" step is the
    Claim/Yield team game played by `decider` instead of "lowest index wins".

    For a contested goal g and each player i:
        a_i = BFS distance to g
        b_i = BFS distance to i's next choice after g (unreachable if none)

    Outcome applied:
        one claimer -> it keeps g, the others move to their next choice
        nobody      -> everyone moves to its next choice (g stays free)
        clash       -> handled inside play_conflict (Option A); round-1
                       claimers wait one step per clash round (`hold`)

    Args:
        candidates: agent indices to assign
        prefs:      dict agent -> preference list (goal letters)
        decision:   "goal" (Phase 2) or "replan" (stranded agents), for logs
        game_form:  "claimyield" (default, original) or "matching" (NEW 10 Nov)
    Returns:
        dict agent -> goal letter, or None if its preferences ran out
    """
    ptr = {i: 0 for i in candidates}

    def choice(i):
        return prefs[i][ptr[i]] if ptr[i] < len(prefs[i]) else None

    while True:
        wanted = {}
        for i in candidates:
            g = choice(i)
            if g is not None:
                wanted.setdefault(g, []).append(i)
        contested = sorted(g for g, who in wanted.items() if len(who) >= 2)
        if not contested:
            break

        g = contested[0]                       # settle one conflict, then re-check
        players = sorted(wanted[g])
        a = {i: _dist(env, agent_positions[i], g, unreachable) for i in players}
        b = {}
        for i in players:
            nxt = prefs[i][ptr[i] + 1] if ptr[i] + 1 < len(prefs[i]) else None
            b[i] = _dist(env, agent_positions[i], nxt, unreachable) if nxt else unreachable
        game = TeamConflictGame("goal", g, players, a, b, K, D,
                                context={"unreachable": unreachable})

        final, record = play_conflict(game, decider, rotation, form=game_form,
                                      prompt_variant=prompt_variant)      # NEW 12 Nov
        record.update({"step": step, "decision": decision})
        conflict_log.append(record)
        print(f"Goal game for {g}: players {[i + 1 for i in players]}, "
              f"round 1 {record['round1_actions']}, final {record['final_actions']}, s={record['s']}")

        if record["clash_rounds"]:
            # Each failed round costs one step: the claimers wait after a
            # clash, everyone waits after an all-YIELD round.
            if record.get("form") == "matching":
                # NEW 10 Nov: in the matching form, the agents that named
                # themselves are the ones that headed for the goal.
                named = record["round1_names"].split()
                claimed = [i for i, nm in zip(players, named) if nm == f"A{i + 1}"]
            else:
                round1 = record["round1_actions"].split()
                claimed = [i for i, act in zip(players, round1) if act == CLAIM]
            for i in (claimed or players):
                hold[i] += record["clash_rounds"]

        winners = game.claimers(final)
        for i in players:
            if not (len(winners) == 1 and winners[0] == i):
                ptr[i] += 1                    # yield: move to the next choice

    return {i: choice(i) for i in candidates}


# ===========================================================================
# Episode
# ===========================================================================
def run(
    image_path="data/grid.png",
    log_path="data/agent_rank_game_logs.csv",
    max_steps=100,
    config_path=None,
    obstacles={(2, 2), (3, 3), (4, 1)},
    grid_size=6,
    num_agents=3,
    agent_starts: list[tuple[int, int]] = None,
    goal_positions: list[tuple[int, int]] = None,
    decider="llm",                          # NEW: "rule", "nash" or "llm"
    rotation=0,                             # NEW: prompt order for the llm decider
    prompt_variant="current",               # NEW 12 Nov: matching prompt wording
                                            #   ("current" = 10 Nov prompt, or "A", "B")
    game_form="claimyield",                 # NEW 10 Nov: "claimyield" (default =
                                            #   original 5 Nov behaviour) or "matching"
    clash_factor=CLASH_FACTOR,              # NEW: K = clash_factor * grid size
    delay_factor=YIELD_DELAY_FACTOR,        # NEW: D = delay_factor * grid size
    conflict_log=None,                      # NEW: list that receives conflict records
    move_log=None,                          # NEW: list that receives step-log rows;
                                            #      if given, no per-trial files are written
    reasoning_log=None,                     # NEW 10 Nov: list that receives each agent's
                                            #   Phase 1 ranking reasoning (logging only)
    **kwargs,                               # NEW: ignore extra args from eval scripts
):
    """
    Run one episode. Same arguments and return value as the original
    agent_rank_once_bfs.run(), plus the NEW arguments above.

    Returns:
        (steps, total_opt, failed, collisions), exactly like the original.
        `collisions` counts agents that did not move because of a cell
        conflict, the same quantity the original counted.
    """
    if decider not in DECIDERS:
        raise ValueError(f"decider must be one of {DECIDERS}")
    if game_form not in FORMS:
        raise ValueError(f"game_form must be one of {FORMS}")

    if config_path:
        env = GridWorld(config_path)
        grid_size = env.size
        num_agents = len(env.agents)
    else:
        env = GridWorld(grid_size, obstacles=obstacles)
        if agent_starts and goal_positions:
            env.initialize_agents_goals_custom(agents=agent_starts, goals=goal_positions)
        else:
            env.initialize_agents_goals(num_agents=num_agents)

    # NOTE (kept from the original on purpose): with a config file,
    # `obstacles` below is still the default {(2,2),(3,3),(4,1)}, not
    # env.obstacles, so the ranking prompt lists the wrong obstacles. The BFS
    # uses the correct env.obstacles. Kept identical so that differences from
    # agent_rank_once_bfs come from the games, not from a different prompt.

    K, D = default_costs(env.size, clash_factor, delay_factor)   # NEW
    unreachable = 4 * env.size * env.size                        # NEW
    conflicts = [] if conflict_log is None else conflict_log     # NEW
    n_before = len(conflicts)
    hold = [0] * num_agents                                      # NEW: waiting steps
    cell_priority = {}                                           # NEW: cell -> agent

    agent_positions = env.agents[:]
    agent_ids = list(range(1, num_agents + 1))
    active = [True] * num_agents
    visits = [{} for _ in range(num_agents)]
    memories = [[] for _ in range(num_agents)]
    target_memories = [[] for _ in range(num_agents)]
    target_goals = [None for _ in range(num_agents)]
    step = 0
    collisions = 0
    log_rows = []
    agent_rankings = [[] for _ in range(num_agents)]

    # The repo's "Optimal" column (sum of each agent's nearest-goal distance).
    # Kept for compatibility; eval_game.py also records the TRUE optimum.
    total_opt = sum(
        min([shortest_path_length(start, g, env) for g in env.goals if g is not None], default=0)
        for start in env.agents if start is not None
    )

    print(f"Initial agent positions: {agent_positions}")
    print(f"Goal positions: {env.goals}")
    print(f"Obstacles: {obstacles}")
    print(f"Decider: {decider}, game: {game_form}, rotation: {rotation}, K={K:g}, D={D:g}")

    plot_grid_unassigned_labeled(env, image_path=image_path)

    # ----------- Phase 1: Ranking (UNCHANGED) ----------
    for i in range(num_agents):
        if not active[i] or agent_positions[i] is None:
            continue

        agent_id = agent_ids[i]
        agent_pos = agent_positions[i]
        visits[i][agent_pos] = visits[i].get(agent_pos, 0) + 1

        other_infos = [
            (agent_ids[j], agent_positions[j])
            for j in range(num_agents)
            if j != i and agent_positions[j] is not None
        ]

        distances = {
            agent_ids[j]: [
                shortest_path_length(agent_positions[j], goal, env) if agent_positions[j] and goal else float("inf")
                for goal in env.goals
            ]
            for j in range(num_agents)
        }

        ranking, explanation, reasoning = select_target(
            agent_id=agent_id,
            agent_pos=agent_pos,
            goal_positions=env.goals,
            other_agents=other_infos,
            grid_size=grid_size,
            obstacles=obstacles,
            memory=memories[i],
            visits=visits[i],
            agent_targets=target_goals,
            target_memory=target_memories[i],
            image_path=image_path,
            step=step,
            distances=distances
        )
        agent_rankings[i] = ranking

        # NEW 10 Nov: keep the ranking's reasoning for the reasoning log.
        # Logging only - the ranking itself is used exactly as before.
        if reasoning_log is not None:
            reasoning_log.append({
                "step": step, "decision": "ranking", "kind": "ranking", "subject": "",
                "agent": agent_id, "round": "", "rotation": "",
                "answer": " ".join(ranking) if ranking else "none",
                "reasoning": reasoning, "reply": explanation,
            })

    # ----------- Phase 2: one goal game per conflict (CHANGED) ----------
    # ORIGINAL:
    #   proposed_goals = [rank[0] if rank else None for rank in agent_rankings]
    #   ... build conflict_pairs (only goals wanted by EXACTLY 2 agents) ...
    #   print("Conflict pairs detected:", conflict_pairs)
    #   proposed_goals = resolve_conflicts(agent_rankings, active)
    all_goals = [_letter(g) for g, pos in enumerate(env.goals) if pos is not None]
    players = [i for i in range(num_agents) if active[i] and agent_positions[i] is not None]
    prefs = {i: preferences(env, agent_positions[i], agent_rankings[i], all_goals, unreachable)
             for i in players}
    print("Preferences (LLM ranking, then distance):", {i + 1: prefs[i] for i in players})

    assigned = assign_goals(env, agent_positions, players, prefs, decider, rotation,
                            K, D, unreachable, hold, conflicts, step, "goal", game_form,
                            prompt_variant)
    proposed_goals = [assigned.get(i) for i in range(num_agents)]
    print("Proposed goals after the goal games:", proposed_goals)

    while any(active) and step < max_steps:
        print(f"\n--- Step {step} ---")
        plot_grid_unassigned_labeled(env, image_path=image_path)
        proposals = agent_positions[:]

        # ----------- Re-plan stranded agents (NEW, always on) ----------
        # ORIGINAL: nothing. If another agent stepped onto this agent's target
        # (the environment stops an agent on ANY free goal it walks over), the
        # target became None and the agent stood still until max_steps.
        # NOW: the stranded agents play the goal game again over the goals
        # nobody else is heading to, reusing their Phase 1 LLM rankings.
        stranded = [
            i for i in range(num_agents)
            if active[i] and agent_positions[i] is not None
            and (proposed_goals[i] is None or _goal_pos(env, proposed_goals[i]) is None)
        ]
        if stranded:
            taken = {proposed_goals[j] for j in range(num_agents)
                     if active[j] and j not in stranded and proposed_goals[j] is not None}
            free = [g for g in (_letter(k) for k, pos in enumerate(env.goals) if pos is not None)
                    if g not in taken]
            if free:
                sprefs = {i: preferences(env, agent_positions[i], agent_rankings[i], free, unreachable)
                          for i in stranded}
                new_goals = assign_goals(env, agent_positions, stranded, sprefs, decider, rotation,
                                         K, D, unreachable, hold, conflicts, step, "replan", game_form,
                                         prompt_variant)
                for i in stranded:
                    proposed_goals[i] = new_goals[i]
                print(f"Re-plan at step {step}: agents {[i + 1 for i in stranded]} -> "
                      f"{[new_goals[i] for i in stranded]}")

        # ----------- Phase 3: Direction Selection & Movement (UNCHANGED) ----------
        # Each agent still takes the BFS step that shortens its path the most.
        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue

            # NEW: an agent paying for a clash waits this step
            if hold[i] > 0:
                hold[i] -= 1
                proposals[i] = agent_positions[i]
                continue

            agent_id = agent_ids[i]
            agent_pos = agent_positions[i]
            new_target = proposed_goals[i]

            best = select_direction_opt(
                agent_pos,
                new_target,
                env.goals,
                env
            )
            explanation = "Chose shortest path step deterministically."

            if best:
                proposals[i] = env.move_agent(agent_pos, best)
                log_rows.append({
                    "step": step,
                    "agent_id": agent_id,
                    "position_before": agent_pos,
                    "position_after": proposals[i],
                    "chosen_direction": best,
                    "explanation": explanation,
                })
                memories[i].append((agent_pos[0], agent_pos[1], best, proposals[i][0], proposals[i][1]))
                if len(memories[i]) > 5:
                    memories[i] = memories[i][-5:]
            else:
                proposals[i] = agent_pos

        target_goals = proposed_goals[:]

        # ----------- Collision Resolution: one cell game per clash (CHANGED) ----------
        # ORIGINAL:
        #   new_positions = proposals[:]
        #   for i in range(num_agents):
        #       for j in range(i + 1, num_agents):
        #           if new_positions[i] == new_positions[j] and ...:
        #               collisions += 1
        #               new_positions[j] = agent_positions[j]   # higher index waits
        #
        # Only same-cell clashes can happen: GridWorld.is_valid() blocks moves
        # into a cell that is occupied at the start of the step.
        new_positions = proposals[:]
        movers = {}
        for i in range(num_agents):
            if agent_positions[i] is None or proposals[i] is None or proposals[i] == agent_positions[i]:
                continue
            movers.setdefault(proposals[i], []).append(i)

        for cell, group in sorted(movers.items()):
            if len(group) < 2:
                continue
            group = sorted(group)

            # A replay/Nash decision from last step's clash on this cell
            if cell in cell_priority and cell_priority[cell] in group:
                winner = cell_priority.pop(cell)
                for i in group:
                    if i != winner:
                        new_positions[i] = agent_positions[i]
                        collisions += 1
                print(f"Cell {cell}: Agent {winner + 1} goes first (decided after last step's clash)")
                continue

            a, b = {}, {}
            for i in group:
                letter = target_goals[i]
                a[i] = _dist(env, cell, letter, unreachable) if letter else unreachable
                b[i] = (_dist(env, agent_positions[i], letter, unreachable) if letter else unreachable) + 1
            game = TeamConflictGame("cell", cell, group, a, b, K, D,
                                    context={"unreachable": unreachable})
            final, record = play_conflict(game, decider, rotation, form=game_form,
                                          prompt_variant=prompt_variant)  # NEW 12 Nov
            record.update({"step": step, "decision": "cell"})
            conflicts.append(record)
            print(f"Cell game at {cell}: players {[i + 1 for i in group]}, "
                  f"round 1 {record['round1_actions']}, final {record['final_actions']}, s={record['s']}")

            winners = game.claimers(final)
            if record["clash_rounds"]:
                # The failed round costs this step: nobody moves; the agent the
                # replay/Nash picked goes first if the conflict repeats next step.
                for i in group:
                    new_positions[i] = agent_positions[i]
                    collisions += 1
                if len(winners) == 1:
                    cell_priority[cell] = winners[0]
            else:
                for i in group:
                    if not (len(winners) == 1 and winners[0] == i):
                        new_positions[i] = agent_positions[i]
                        collisions += 1

        # Safety net: after the games no two agents should share a cell. If
        # this ever prints, there is a bug in the game layer; the original
        # rule is applied so the run can continue.
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if new_positions[i] == new_positions[j] and agent_positions[i] is not None and agent_positions[j] is not None:
                    print(f"⚠️ Unresolved clash between agents {i + 1} and {j + 1}; using the original rule")
                    collisions += 1
                    new_positions[j] = agent_positions[j]

        agent_positions = new_positions
        env.agents = agent_positions[:]

        # ----------- Goal Claiming (UNCHANGED) ----------
        # Note: an agent stops on ANY free goal it steps on, even one that is
        # not its target. This is an environment rule, kept as is.
        to_remove = []
        claimed_goals = []
        for i in range(num_agents):
            if active[i] and agent_positions[i] in env.goals:
                print(f"Agent {agent_ids[i]} reached goal at {agent_positions[i]}")
                active[i] = False
                to_remove.append(i)
                claimed_goals.append(agent_positions[i])

        for i in to_remove:
            agent_positions[i] = None
            env.agents[i] = None
            target_goals[i] = None

        for goal in claimed_goals:
            if goal in env.goals:
                env.goals[env.goals.index(goal)] = None

        print(f"Remaining agents: {[agent_ids[i] for i in range(num_agents) if active[i]]}")
        print(f"Remaining goals: {env.goals}")
        step += 1

    failed = step >= max_steps
    print(f"\nRun completed in {step} steps. Collisions: {collisions}. Failed: {failed}")
    # CHANGED (5 Nov): when called from eval_game.py the rows go to the
    # caller's lists and no per-trial files are written.
    if move_log is not None:
        move_log.extend(log_rows)
        return step, total_opt, failed, collisions

    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    # NEW: save every conflict of this run next to the step log
    with open(os.path.splitext(log_path)[0] + "_conflicts.json", "w") as f:
        json.dump(conflicts[n_before:], f, indent=2, default=str)

    return step, total_opt, failed, collisions


if __name__ == "__main__":
    # Run a single case, e.g.
    #   python -m game_theory.agent_rank_once_game --decider llm --config configs/difficult/case_1_local_greed.yaml
    parser = argparse.ArgumentParser(description="Run the game-theoretic rank-once agent on one case.")
    parser.add_argument("--config", type=str, default="configs/difficult/case_10_insane.yaml", help="Path to config YAML file")
    parser.add_argument("--decider", choices=DECIDERS, default="llm")
    parser.add_argument("--rotation", type=int, choices=[0, 1], default=0)
    parser.add_argument("--game", choices=FORMS, default="claimyield",
                        help="claimyield (default, original) or matching (NEW 10 Nov)")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    steps, optimal, failed, collisions = run(
        config_path=args.config, decider=args.decider, rotation=args.rotation,
        game_form=args.game
    )
    print(f"\n✅ Done!\nOptimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
