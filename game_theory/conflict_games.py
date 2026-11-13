"""
game_theory/conflict_games.py
=============================

The collaborative conflict game used at every decision point of the
game-theoretic agent, plus the three "deciders" that play it and the score
used to compare them (Eq. 1 of "Do LLMs Beat Nash?").

Where it is used (see agent_rank_once_game.py):
    1. Goal conflict  : 2+ agents rank the same goal first
    2. Cell clash     : 2+ BFS moves land on the same empty cell
    3. Stranded agent : the game of point 1, re-played over the free goals

The game (one game per conflict)
--------------------------------
Players : the agents involved in the conflict.
Actions : CLAIM (take the contested goal / make my BFS move)
          YIELD (take my next choice / wait one step).
Costs   : a_i = agent i's cost if it claims and wins
          b_i = agent i's cost if it yields
          K   = clash cost      = CLASH_FACTOR * grid size      (6 on 8x8)
          D   = all-yield delay = YIELD_DELAY_FACTOR * grid size (3 on 8x8)
          Both scale with the grid by the same rule, so K:D stays 2:1.

Common (team) payoff, identical for every player - minus the local makespan:
    exactly one claimer w : -max(a_w, b_j for every yielder j)
    2+ claimers (clash)   : -max(K + largest a among claimers,
                                 largest b among yielders)
    nobody claims         : -(largest b + D)
With two players the clash payoff is simply -(K + max a), as in the doc.

Deciders
--------
    rule : ORIGINAL rule from the repo - the lowest agent index claims, the
           others yield. Never clashes.
    nash : exact solver - the pure Nash equilibrium with the best common
           payoff (ties: the lowest index claims). Never clashes.
    llm  : each player asks its own copy of the model, independently, with no
           communication (llm_player.py).

Two forms of the same game (--game in eval_game.py)
--------------------------------------------------
    claimyield (DEFAULT):
        each agent answers CLAIM or YIELD. The good outcomes need DIFFERENT
        answers (one CLAIM, the rest YIELD): an anti-coordination game.
    matching (NEW, 10 Nov):
        each agent answers "which agent should take it?" by naming one of the
        players. The good outcomes need the SAME answer from everyone: a
        matching (coordination) game, the archetype where open-weight models
        did well in "Do LLMs Beat Nash?". See MatchingConflictGame below.
    With two players the two forms have exactly the same four outcomes and
    payoffs (naming yourself = CLAIM, naming the other = YIELD), so their
    scores are directly comparable; only the framing differs. Running with
    the default form gives exactly the behaviour and files of the 5 Nov code.

Miscoordination handling (Option A, chosen in the design doc)
-------------------------------------------------------------
Round 1 is played by the decider. If it does not produce exactly one
claimer - a clash (2+ CLAIM) or nobody claiming (all YIELD) - round 2 is
played with the action order in the prompt flipped (YIELD listed first). If
round 2 also fails, the nash decider settles it. Each failed round costs one
waiting step (the caller applies it: the claimers wait after a clash,
everyone waits after an all-YIELD round).

Why all-YIELD is included (beyond the clash case in the doc): in a cell
conflict nobody moves when everyone yields, and at temperature 0 the model
gives the same answer next step, so the agents would wait forever. Treating
it like a clash guarantees progress. Only the llm decider can miscoordinate;
rule and nash always name exactly one claimer, except that nash may choose
all-YIELD in a goal conflict when that is truly best for the team.

Reasoning log (NEW 10 Nov)
--------------------------
Every record returned by play_conflict / play_matching_conflict carries an
"llm_calls" list: one entry per model answer (agent, round 1 or 2, rotation,
answer, the model's "reasoning" text, the raw reply). It is empty for the
rule and nash deciders. eval_game.py writes it to <key>_reasoning.csv; it is
not a column of the conflicts CSV, and it never changes a decision.

Score
-----
For every conflict the score s is computed on the ROUND-1 outcome, so a clash
always counts against the decider:
    a = common payoff achieved, O = best cell, W = worst cell,
    N = worst Nash equilibrium (pure; plus the fully mixed one for 2 players)
    s = (a - N) / (O - N)   if a >= N        (0 = Nash level, +1 = optimum)
    s = (a - N) / (N - W)   if a <  N        (-1 = worst outcome)
    s = None when O == N (every equilibrium is already optimal: no headroom,
    excluded from averages as in the paper).
Mixed equilibria are only computed exactly for 2 players; for 3+ players N is
the worst PURE equilibrium (recorded as "mixed_checked": False).
"""

import itertools

from game_theory.game import pure_nash_equilibria, EPS

CLAIM, YIELD = "CLAIM", "YIELD"
ACTIONS = (CLAIM, YIELD)
DECIDERS = ("rule", "nash", "llm")
FORMS = ("claimyield", "matching")   # NEW 10 Nov; "claimyield" = original

# K and D as a fraction of the grid size (decided 5 Nov: scale with the grid,
# same rule for both). 8x8 -> K = 6, D = 3; 20x20 -> K = 15, D = 7.5.
CLASH_FACTOR = 0.75
YIELD_DELAY_FACTOR = 0.375


def default_costs(grid_size, clash_factor=CLASH_FACTOR, delay_factor=YIELD_DELAY_FACTOR):
    """Return (K, D) for a grid of the given size."""
    return clash_factor * grid_size, delay_factor * grid_size


# ===========================================================================
# The game
# ===========================================================================
class TeamConflictGame:
    """
    One collaborative claim/yield conflict.

    Args:
        kind:    "goal" or "cell" (only used in prompts and logs)
        subject: the contested goal letter, or the contested cell (row, col)
        players: list of agent indices (0-based), in ascending order
        a, b:    dicts agent index -> cost if it claims and wins / if it yields
        K, D:    clash cost and all-yield delay
        context: dict with anything the prompt should show (positions, the
                 yield alternative, ...)
    """

    def __init__(self, kind, subject, players, a, b, K, D, context=None):
        self.kind = kind
        self.subject = subject
        self.players = list(players)
        self.a = dict(a)
        self.b = dict(b)
        self.K = K
        self.D = D
        self.context = context or {}
        # Every profile = one action per player, in self.players order
        self.profiles = list(itertools.product(ACTIONS, repeat=len(self.players)))

    # --- payoff --------------------------------------------------------
    def payoff(self, profile):
        """Common payoff of a profile (tuple of CLAIM/YIELD, one per player)."""
        claimers = [i for i, s in zip(self.players, profile) if s == CLAIM]
        yielders = [i for i, s in zip(self.players, profile) if s == YIELD]
        if len(claimers) == 1:
            w = claimers[0]
            return -max([self.a[w]] + [self.b[j] for j in yielders])
        if len(claimers) >= 2:
            clash = self.K + max(self.a[i] for i in claimers)
            return -max([clash] + [self.b[j] for j in yielders])
        return -(max(self.b.values()) + self.D)

    def claimers(self, profile):
        return [i for i, s in zip(self.players, profile) if s == CLAIM]

    def is_clash(self, profile):
        return len(self.claimers(profile)) >= 2

    # --- equilibria ----------------------------------------------------
    def pure_equilibria(self):
        """All pure Nash equilibria as (profile, payoff)."""
        n = len(self.players)
        eqs, _ = pure_nash_equilibria(
            [list(ACTIONS)] * n, lambda p: [self.payoff(p)] * n
        )
        return [(p, u[0]) for p, u in eqs]

    def mixed_equilibrium_2p(self):
        """
        Fully mixed equilibrium of a 2-player game, or None.

        Each player claims with the probability that makes the OTHER player
        indifferent between CLAIM and YIELD (standard 2x2 formula).
        Returns (p_claim_player0, p_claim_player1, expected payoff).
        """
        if len(self.players) != 2:
            return None
        U = {p: self.payoff(p) for p in self.profiles}
        cc, cy, yc, yy = U[(CLAIM, CLAIM)], U[(CLAIM, YIELD)], U[(YIELD, CLAIM)], U[(YIELD, YIELD)]
        denom = cc - cy - yc + yy
        if abs(denom) < EPS:
            return None
        p = (yy - yc) / denom          # player 0 claims with p
        q = (yy - cy) / denom          # player 1 claims with q
        if not (EPS < p < 1 - EPS and EPS < q < 1 - EPS):
            return None
        value = p * q * cc + p * (1 - q) * cy + (1 - p) * q * yc + (1 - p) * (1 - q) * yy
        return p, q, value

    def benchmarks(self):
        """
        O, W, N and the equilibrium gap for the score.

        gap = best pure equilibrium payoff - worst pure equilibrium payoff:
        a large gap means one equilibrium is clearly better (easy conflict);
        0 or 1 step means pure symmetry-breaking (hard conflict).
        """
        values = [self.payoff(p) for p in self.profiles]
        pure = self.pure_equilibria()
        eq_values = [v for _, v in pure]
        mixed = self.mixed_equilibrium_2p()
        if mixed is not None:
            eq_values.append(mixed[2])
        pure_values = [v for _, v in pure] or eq_values
        return {
            "O": max(values),
            "W": min(values),
            "N": min(eq_values) if eq_values else min(values),
            "gap": (max(pure_values) - min(pure_values)) if pure_values else 0.0,
            "num_pure_equilibria": len(pure),
            "mixed": None if mixed is None else {"p_claim": [mixed[0], mixed[1]], "value": mixed[2]},
            "mixed_checked": len(self.players) == 2,
        }

    def score(self, profile, bench=None):
        """Eq. 1 of the paper on the common payoff of `profile`."""
        bench = bench or self.benchmarks()
        a, O, N, W = self.payoff(profile), bench["O"], bench["N"], bench["W"]
        if O - N < EPS:
            return None                       # no headroom: excluded
        if a >= N - EPS:
            return (a - N) / (O - N)
        return (a - N) / (N - W) if N - W > EPS else -1.0

    # --- the two non-LLM deciders --------------------------------------
    def rule_profile(self):
        """ORIGINAL rule: lowest index claims, everyone else yields."""
        low = min(self.players)
        return tuple(CLAIM if i == low else YIELD for i in self.players)

    def nash_profile(self):
        """Best pure equilibrium for the team; ties -> lowest index claims."""
        def key(item):
            profile, value = item
            cl = self.claimers(profile)
            return (-value, len(cl) != 1, cl[0] if cl else len(self.players) + 99)
        pure = self.pure_equilibria()
        if pure:
            return min(pure, key=key)[0]
        # Cannot happen for a common-payoff game (the best profile is always
        # an equilibrium), but never crash a long run.
        return max(self.profiles, key=self.payoff)


# ===========================================================================
# Playing a conflict
# ===========================================================================
def play_conflict(game, decider, rotation=0, ask_fn=None, form="claimyield",
                  prompt_variant="current"):
    """
    Resolve one conflict with the chosen decider, applying Option A.

    Args:
        game:     TeamConflictGame
        decider:  "rule", "nash" or "llm"
        rotation: 0 or 1 - order of the actions/players in the LLM prompt
                  (the trial's rotation; ignored by rule and nash)
        ask_fn:   function(game, agent, rotation) -> (action, info); defaults
                  to llm_player.ask_agent. Only used by the llm decider.
        form:     "claimyield" (default, the original 5 Nov behaviour) or
                  "matching" (NEW: handled by play_matching_conflict below;
                  the claimyield code path is untouched)

    Returns:
        final_profile: the profile that is applied in the simulation
        record:        dict for the conflicts CSV (round-1 actions, score,
                       benchmarks, clash rounds, fallbacks, parse failures)
    """
    if decider not in DECIDERS:
        raise ValueError(f"decider must be one of {DECIDERS}")
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}")
    if form == "matching":                       # NEW 10 Nov
        return play_matching_conflict(game, decider, rotation, ask_fn, prompt_variant)
    if prompt_variant != "current":              # NEW 12 Nov
        raise ValueError("prompt variants A/B exist only for the matching form")
    # ---- ORIGINAL claim/yield form (unchanged from 5 Nov) ----
    bench = game.benchmarks()

    calls = []      # NEW 10 Nov: every model answer, for the reasoning log

    def one_round(rot):
        if decider == "rule":
            return game.rule_profile(), 0
        if decider == "nash":
            return game.nash_profile(), 0
        if ask_fn is None:
            from game_theory.llm_player import ask_agent
            fn = ask_agent
        else:
            fn = ask_fn
        actions, failures = [], 0
        for agent in game.players:          # independent calls, no sharing
            action, info = fn(game, agent, rot)
            failures += int(info.get("parse_failed", False))
            actions.append(action)
            calls.append({"agent": agent + 1, "round": 1 if rot == rotation else 2,
                          "rotation": rot, "answer": action,
                          "reasoning": info.get("reasoning", ""), "reply": info.get("reply", "")})
        return tuple(actions), failures

    first, parse_failures = one_round(rotation)
    def failed(profile):
        # Miscoordination: a clash, or nobody claiming (see module docstring)
        return decider == "llm" and len(game.claimers(profile)) != 1

    final, clash_rounds, nash_fallback = first, 0, False
    if failed(first):
        clash_rounds = 1
        # Option A: one replay with the action order flipped ...
        second, pf2 = one_round(1 - rotation)
        parse_failures += pf2
        final = second
        if failed(second):
            # ... then the Nash solver settles it.
            clash_rounds = 2
            final = game.nash_profile()
            nash_fallback = True

    record = {
        "kind": game.kind,
        "subject": str(game.subject),
        "players": " ".join(f"A{i + 1}" for i in game.players),
        "a": " ".join(f"{game.a[i]:g}" for i in game.players),
        "b": " ".join(f"{game.b[i]:g}" for i in game.players),
        "decider": decider,
        "rotation": rotation,
        "round1_outcome": ("one" if len(game.claimers(first)) == 1
                           else "clash" if game.is_clash(first) else "none"),
        "round1_actions": " ".join(first),
        "final_actions": " ".join(final),
        "round1_payoff": game.payoff(first),
        "final_payoff": game.payoff(final),
        "O": bench["O"],
        "N": bench["N"],
        "W": bench["W"],
        "s": game.score(first, bench),
        "equilibrium_gap": bench["gap"],
        "coordinated": int(len(game.claimers(first)) == 1),
        "clash_rounds": clash_rounds,
        "nash_fallback": int(nash_fallback),
        "parse_failures": parse_failures,
        "mixed_checked": int(bench["mixed_checked"]),
        "llm_calls": calls,     # NEW 10 Nov: reasoning log (not a CSV column)
    }
    return final, record


# ===========================================================================
# NEW (10 Nov): the matching form of the same conflict
# ===========================================================================
class MatchingConflictGame:
    """
    The matching form of a TeamConflictGame: every player names the agent
    that should take the contested goal or cell.

    Strategies: each player names one of the players (agent indices).

    Payoff (common, built from the claim/yield payoffs of `base`):
        everyone names the same agent w (AGREEMENT)
            -> w takes it, the others yield: base payoff with only w claiming
               = -max(a_w, b_j for every other j)
        any disagreement = FAILURE (decided 10 Nov). Its cost depends on who
        named themselves, because those agents head for the goal anyway:
            1+ agents named themselves -> -max(K + largest a among them,
                                               largest b among the others)
            nobody named themselves    -> -(largest b + D)
    Disagreeing always costs K or D on top of an agreed outcome, so every
    agreement is a pure Nash equilibrium and the game is a matching game.

    With two players this is exactly the claim/yield game relabelled
    (naming yourself = CLAIM), so O, W, N and s are identical:
        (A1, A1) = only A1 claims   (A2, A2) = only A2 claims
        (A1, A2) = both claim/clash (A2, A1) = nobody claims
    For 3+ players disagreement is stricter than in claim/yield (there, one
    self-claimer with mixed other answers still counted as a success).
    """

    def __init__(self, base):
        self.base = base
        self.players = base.players
        self.kind, self.subject = base.kind, base.subject
        self.K, self.D = base.K, base.D
        # Every profile = one named agent per player
        self.profiles = list(itertools.product(self.players, repeat=len(self.players)))

    def winner(self, names):
        """
        The agreed agent, or None if the players disagree. An unparsed reply
        is stored as -1 (named nobody), which can never be an agreement.
        """
        if names and names[0] in self.players and all(n == names[0] for n in names):
            return names[0]
        return None

    def payoff(self, names):
        w = self.winner(names)
        if w is not None:
            return self.base.payoff(self.to_actions(names))
        selfish = [i for i, n in zip(self.players, names) if n == i]
        others = [j for j in self.players if j not in selfish]
        if selfish:
            clash = self.K + max(self.base.a[i] for i in selfish)
            return -max([clash] + [self.base.b[j] for j in others])
        return -(max(self.base.b.values()) + self.D)

    def to_actions(self, names):
        """
        Claim/yield profile the simulation applies: only the agreed agent
        claims. Called only on an agreed outcome (failures are replayed).
        """
        w = self.winner(names)
        return tuple(CLAIM if i == w else YIELD for i in self.players)

    def pure_equilibria(self):
        n = len(self.players)
        eqs, _ = pure_nash_equilibria(
            [list(self.players)] * n, lambda p: [self.payoff(p)] * n
        )
        return [(p, u[0]) for p, u in eqs]

    def benchmarks(self):
        """Same quantities as TeamConflictGame.benchmarks, in name space."""
        values = [self.payoff(p) for p in self.profiles]
        pure = self.pure_equilibria()
        pure_values = [v for _, v in pure]
        eq_values = list(pure_values)
        # Two players: the game is the claim/yield game relabelled, so its
        # fully mixed equilibrium is the same one (naming yourself = CLAIM).
        mixed = self.base.mixed_equilibrium_2p()
        if mixed is not None:
            eq_values.append(mixed[2])
        return {
            "O": max(values),
            "W": min(values),
            "N": min(eq_values) if eq_values else min(values),
            "gap": (max(pure_values) - min(pure_values)) if pure_values else 0.0,
            "num_pure_equilibria": len(pure),
            "mixed": None if mixed is None else {"p_name_self": [mixed[0], mixed[1]], "value": mixed[2]},
            "mixed_checked": len(self.players) == 2,
        }

    def score(self, names, bench=None):
        """Eq. 1 of the paper on the common payoff of `names`."""
        bench = bench or self.benchmarks()
        a, O, N, W = self.payoff(names), bench["O"], bench["N"], bench["W"]
        if O - N < EPS:
            return None
        if a >= N - EPS:
            return (a - N) / (O - N)
        return (a - N) / (N - W) if N - W > EPS else -1.0

    def rule_names(self):
        """ORIGINAL rule in this form: everyone names the lowest index."""
        low = min(self.players)
        return tuple(low for _ in self.players)

    def nash_names(self):
        """Best agreement for the team; ties -> the lowest index."""
        agreements = [tuple(w for _ in self.players) for w in self.players]
        return min(agreements, key=lambda p: (-self.payoff(p), p[0]))


def play_matching_conflict(game, decider, rotation=0, ask_fn=None, prompt_variant="current"):
    """
    NEW (10 Nov): resolve one conflict in the matching form.

    Same deciders and the same Option A as the claim/yield form:
        round 1 by the decider; if the players DISAGREE, round 2 with the
        order of the candidate agents in the prompt reversed; if they
        disagree again, the nash decider settles it. Each failed round costs
        one waiting step (applied by the caller, as in claim/yield).

    Returns (final, record) like play_conflict. `final` is a CLAIM/YIELD
    profile of the base game (only the agreed agent claims), so the agent
    code applies it exactly as before. The record has the same columns as
    the claim/yield record plus "form", "round1_names", "final_names" and
    "agreed"; for comparability "coordinated" = agreed.
    """
    mgame = MatchingConflictGame(game)
    bench = mgame.benchmarks()

    calls = []      # NEW 10 Nov: every model answer, for the reasoning log

    def one_round(rot):
        if decider == "rule":
            return mgame.rule_names(), 0
        if decider == "nash":
            return mgame.nash_names(), 0
        if ask_fn is None:
            from functools import partial
            from game_theory.llm_player import ask_agent_matching
            # NEW 12 Nov: which wording of the matching prompt (current, A, B)
            fn = partial(ask_agent_matching, variant=prompt_variant)
        else:
            fn = ask_fn
        names, failures = [], 0
        for agent in game.players:          # independent calls, no sharing
            name, info = fn(game, agent, rot)
            failures += int(info.get("parse_failed", False))
            names.append(name)
            calls.append({"agent": agent + 1, "round": 1 if rot == rotation else 2,
                          "rotation": rot,
                          "answer": "none" if name is None or name < 0 else f"Agent {name + 1}",
                          "reasoning": info.get("reasoning", ""), "reply": info.get("reply", "")})
        return tuple(names), failures

    def failed(names):
        return decider == "llm" and mgame.winner(names) is None

    first, parse_failures = one_round(rotation)
    final_names, rounds, nash_fallback = first, 0, False
    if failed(first):
        rounds = 1
        second, pf2 = one_round(1 - rotation)
        parse_failures += pf2
        final_names = second
        if failed(second):
            rounds = 2
            final_names = mgame.nash_names()
            nash_fallback = True
    final = mgame.to_actions(final_names)

    def fmt(names):
        return " ".join("none" if n is None or n < 0 else f"A{n + 1}" for n in names)

    agreed = mgame.winner(first) is not None
    selfish = [i for i, n in zip(game.players, first) if n == i]
    record = {
        "form": "matching",
        "prompt_variant": prompt_variant,       # NEW 12 Nov
        "kind": game.kind,
        "subject": str(game.subject),
        "players": " ".join(f"A{i + 1}" for i in game.players),
        "a": " ".join(f"{game.a[i]:g}" for i in game.players),
        "b": " ".join(f"{game.b[i]:g}" for i in game.players),
        "decider": decider,
        "rotation": rotation,
        "round1_outcome": ("agree" if agreed else
                           "disagree_clash" if selfish else "disagree_none"),
        "round1_names": fmt(first),
        "final_names": fmt(final_names),
        "round1_actions": " ".join(mgame.to_actions(first)) if agreed else "-",
        "final_actions": " ".join(final),
        "round1_payoff": mgame.payoff(first),
        "final_payoff": mgame.payoff(final_names),
        "O": bench["O"],
        "N": bench["N"],
        "W": bench["W"],
        "s": mgame.score(first, bench),
        "equilibrium_gap": bench["gap"],
        "agreed": int(agreed),
        "coordinated": int(agreed),
        "clash_rounds": rounds,
        "nash_fallback": int(nash_fallback),
        "parse_failures": parse_failures,
        "mixed_checked": int(bench["mixed_checked"]),
        "llm_calls": calls,     # NEW 10 Nov: reasoning log (not a CSV column)
    }
    return final, record
