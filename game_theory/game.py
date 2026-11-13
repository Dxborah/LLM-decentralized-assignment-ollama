"""
game_theory/game.py
===================

Exact solver for small finite games, used by conflict_games.py.

What changed (5 Nov redesign): this file used to hold two games of its own,
a global goal-assignment game (solve_goal_assignment_game) and a collision
yield game (resolve_collision_game), played by a Nash solver with an LLM
tie-break (--payoff, --llm-weight). Both are REMOVED: every conflict is now
the collaborative claim/yield game in conflict_games.py, played by the
decider you choose (rule, nash or llm). What stays here is the generic
machinery that game reuses:

    pure_nash_equilibria()     every pure Nash equilibrium, by brute force
    max_regret()               how far a profile is from an equilibrium
    min_max_regret_profile()   fallback when a game has no pure equilibrium
    EPS, MAX_PROFILES          numeric tolerance and a size guard

Terminology
-----------
    player    : an agent involved in the game
    strategy  : one choice a player can make (here CLAIM or YIELD)
    profile   : one strategy per player, e.g. ("CLAIM", "YIELD")
    pure Nash equilibrium (NE):
                a profile where NO single player can raise its own payoff by
                changing only its own strategy
"""

import itertools

# Two payoffs closer than EPS are treated as equal, so floating-point noise
# (e.g. a mixed-equilibrium value) never makes a tie look like a preference.
EPS = 1e-9

# Safety limit: the solver enumerates every profile. A claim/yield game with
# 4 agents has 2^4 = 16 profiles, but a bigger config could explode, so we
# refuse instead of hanging.
MAX_PROFILES = 2_000_000


# ===========================================================================
# Generic exact solver for small finite games
# ===========================================================================
def pure_nash_equilibria(strategy_sets, payoff_fn):
    """
    Find every pure Nash equilibrium of a finite normal-form game by brute
    force.

    Args:
        strategy_sets: list with one entry per player; each entry is the list
                       of strategies available to that player.
        payoff_fn:     function(profile) -> list of payoffs, one per player,
                       in the same order as strategy_sets.

    Returns:
        equilibria: list of (profile, payoffs) for every pure NE
        table:      dict profile -> payoffs for EVERY profile (reused by
                    min_max_regret_profile when no NE exists)

    How it works:
        1. Evaluate the payoff of every profile once and store it.
        2. A profile is an NE if, for every player i and every alternative
           strategy s of player i, switching i to s (everyone else fixed)
           does NOT strictly increase player i's payoff.
    """
    n_profiles = 1
    for s in strategy_sets:
        n_profiles *= len(s)
    if n_profiles > MAX_PROFILES:
        raise RuntimeError(f"Game too large to enumerate ({n_profiles} profiles)")

    # Step 1: payoff table
    table = {p: payoff_fn(p) for p in itertools.product(*strategy_sets)}

    # Step 2: test every profile for profitable unilateral deviations
    equilibria = []
    for profile, payoffs in table.items():
        is_equilibrium = True
        for i, strategies in enumerate(strategy_sets):
            for alt in strategies:
                if alt == profile[i]:
                    continue
                deviation = profile[:i] + (alt,) + profile[i + 1:]
                if table[deviation][i] > payoffs[i] + EPS:
                    is_equilibrium = False   # player i would switch
                    break
            if not is_equilibrium:
                break
        if is_equilibrium:
            equilibria.append((profile, payoffs))
    return equilibria, table


def max_regret(profile, strategy_sets, table):
    """
    Regret of a profile = the most that ANY single player could gain by
    switching its own strategy. A pure NE has regret 0.
    """
    worst = 0.0
    for i, strategies in enumerate(strategy_sets):
        current = table[profile][i]
        for alt in strategies:
            deviation = profile[:i] + (alt,) + profile[i + 1:]
            worst = max(worst, table[deviation][i] - current)
    return worst


def min_max_regret_profile(strategy_sets, table, key=None):
    """
    Fallback when a game has NO pure NE: return the profile that is 'closest'
    to being an equilibrium (smallest max regret). `key` breaks ties between
    profiles with the same regret. In the common-payoff claim/yield game a pure
    NE always exists, so this is a safety net only.
    """
    return min(
        table.keys(),
        key=lambda p: (max_regret(p, strategy_sets, table), key(p) if key else 0),
    )
