"""
game_theory/llm_player.py
=========================

The LLM side of the conflict game: builds the prompt one agent sees, sends it
to the model, and parses the reply into CLAIM or YIELD.

Two prompt forms (--game in eval_game.py):
    claimyield (DEFAULT, original 5 Nov version, unchanged): build_prompt,
        parse_action, ask_agent - "CLAIM or YIELD?"
    matching (NEW 10 Nov): build_matching_prompt, parse_name,
        ask_agent_matching - "Which agent should take it?" Every agent gets
        nearly the same text (only "You are Agent N" differs), so identical
        reasoners can land on the same name.

Reasoning log (NEW 10 Nov): ask_agent and ask_agent_matching also return the
model's "reasoning" text and its raw reply in `info`; conflict_games.py
attaches them to each conflict record as "llm_calls", and eval_game.py
writes them to results_game/<key>_reasoning.csv. Logging only.

Prompt variants of the matching form (NEW 12 Nov, --prompt-variant):
    current  DEFAULT - the 10 Nov matching prompt, unchanged
    A        same game, clearer presentation: the team finishing time in
             steps (lower is better) instead of negative payoffs, and a short
             one-line JSON answer
    B        A + the 2-agent payoffs as a grid (rows = your answer, columns =
             the other agent's) + one sentence that the table is common
             knowledge. 3+ agents keep A's list (a grid would be 27+ cells)
             plus the same sentence.
    C        NEW 15 Nov. A's presentation for every conflict size, but the
             2-agent case uses the 3+ layout: only the agreed outcomes
             ("you both name Agent w -> T steps") plus one line for the
             disagreements, so no single answer pair is listed first. Adds
             B's common-knowledge sentence, and the JSON asks for the team
             time under each agreed choice ("team_steps") before the answer.
             Motivation: in the A replay, every 2-agent agreement (46/46)
             was on whichever agent the prompt listed first.
    None of them hints at which agent is better; the game is unchanged.

How a conflict is asked (design doc, "Deciders and the conflict prompt"):
    * Each player gets its own prompt with itself as "you".
    * Calls are independent: no shared memory, nobody sees another answer.
    * The payoff is described as SHARED by the team (decided 5 Nov).
    * rotation 0 lists CLAIM first and the other agents in index order;
      rotation 1 lists YIELD first and the other agents in reverse order.
      Trials alternate rotations, and Option A's replay flips the rotation.
    * Unparsable reply -> counted as YIELD (decided 5 Nov) and appended to
      fails.txt, like the repo does for rankings.

Which model is called
---------------------
LLM_BACKEND=ollama (default): the Ollama model in OLLAMA_MODEL (default
qwen2.5vl:7b), called directly with the `ollama` package at temperature 0,
the same setting the repo uses for its image prompts (decided 5 Nov: keep
the repo's near-zero temperature).

Why not core/request.py: that file defines send_text_to_model_openai twice,
and the second definition (the one Python actually uses) always calls
OpenAI, even with LLM_BACKEND=ollama. Calling Ollama here avoids editing
core/. With LLM_BACKEND=openai the repo's own function is used.
"""

import itertools
import json
import os
import re

from game_theory.conflict_games import CLAIM, YIELD, ACTIONS


def _fmt(x):
    """Payoff/cost as text: whole numbers without decimals, others to 0.1."""
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def build_prompt(game, agent, rotation=0):
    """
    The text agent `agent` (0-based index) sees for this conflict.

    Example (2 agents, goal conflict, rotation 0) - see the design doc:
        You are Agent 1 ... You and Agent 2 both want goal G ...
        CLAIM: keep heading to goal G (3 steps for you)
        YIELD: go to your next choice instead (5 steps for you)
        ...payoff table, shared by the team...
    """
    me = agent + 1
    others = [i for i in game.players if i != agent]
    if rotation == 1:
        others = list(reversed(others))
    labels = [f"Agent {i + 1}" for i in others]
    names = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
    together = "both" if len(labels) == 1 else "all"
    action_order = ACTIONS if rotation == 0 else tuple(reversed(ACTIONS))

    unreachable = game.context.get("unreachable", 1e9)
    if game.kind == "goal":
        what = f"goal {game.subject}"

        def claim_txt(i):
            return f"keep heading to goal {game.subject} ({_fmt(game.a[i])} steps)"

        def yield_txt(i):
            whose = "your" if i == agent else "its"
            if game.b[i] >= unreachable:
                return "give up the goal (no other choice left)"
            return f"go to {whose} next choice instead ({_fmt(game.b[i])} steps)"

        intro = f"You and {names} {together} want {what} as your target."
    else:
        r, c = game.subject
        what = f"cell ({r}, {c})"

        def claim_txt(i):
            return f"step into {what} now ({_fmt(game.a[i])} steps left after)"

        def yield_txt(i):
            return f"wait one step ({_fmt(game.b[i])} steps left counting the wait)"

        intro = f"You and {names} {together} want to step into {what} this turn."

    desc = {CLAIM: claim_txt, YIELD: yield_txt}
    lines = [
        f"You are Agent {me} in a team of robots on a grid. {intro}",
        "You cannot communicate. The other agents are identical copies of you and are",
        "answering this same question at the same time.",
        "",
        "Your options:",
    ]
    for act in action_order:
        lines.append(f"  {act}: {desc[act](agent)}")
    for i in others:
        opts = "; ".join(f"{act}: {desc[act](i)}" for act in action_order)
        lines.append(f"Agent {i + 1} has the same kind of choice ({opts}).")
    lines += [
        "",
        f"A clash (two or more CLAIM) costs about {_fmt(game.K)} extra steps; if everyone",
        f"YIELDs, the {('goal' if game.kind == 'goal' else 'cell')} is wasted and costs about {_fmt(game.D)} extra steps.",
        "",
        "You and the other agents share one team payoff (minus the team's finishing time):",
    ]
    # Every combination, listed from this agent's point of view
    order = [agent] + others
    for combo in itertools.product(action_order, repeat=len(order)):
        chosen = dict(zip(order, combo))
        profile = tuple(chosen[i] for i in game.players)
        parts = [f"you {chosen[agent]}"] + [f"Agent {i + 1} {chosen[i]}" for i in others]
        lines.append(f"  {', '.join(parts)} -> {_fmt(game.payoff(profile))}")
    first, second = action_order
    lines += [
        "",
        "Maximize the shared payoff. Reply in JSON:",
        f'{{"reasoning": "...", "action": "{first}" or "{second}"}}',
    ]
    return "\n".join(lines)


def parse_action(text):
    """
    Read CLAIM or YIELD from the model's reply.

    Tries the JSON "action" field first, then the last CLAIM/YIELD word in
    the text. Returns (action or None).
    """
    if not text:
        return None
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            value = str(json.loads(match.group(0)).get("action", "")).upper()
            if CLAIM in value and YIELD not in value:
                return CLAIM
            if YIELD in value and CLAIM not in value:
                return YIELD
    except Exception:
        pass
    words = re.findall(r"\b(CLAIM|YIELD)\b", text.upper())
    return words[-1] if words else None


def extract_reasoning(text):
    """
    NEW (10 Nov): the "reasoning" field of a JSON reply, for the reasoning
    log (results_game/<key>_reasoning.csv). Logging only - it never changes
    a decision. Returns "" when the reply has no readable reasoning; the full
    raw reply is logged next to it anyway.
    """
    if not text:
        return ""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return str(json.loads(match.group(0)).get("reasoning", "")).strip()
    except Exception:
        pass
    found = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*[,}]', text, re.DOTALL)
    return found.group(1).strip() if found else ""


def call_model(prompt):
    """Send a text prompt to the configured backend and return the reply."""
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "ollama":
        import ollama  # imported here so tests can run without the package
        options = {"temperature": 0, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))}
        if os.getenv("LLM_MAX_OUTPUT_TOKENS"):
            limit = int(os.environ["LLM_MAX_OUTPUT_TOKENS"])
            if limit <= 0:
                raise ValueError("LLM_MAX_OUTPUT_TOKENS must be positive")
            options["num_predict"] = limit
        extra = {}
        if os.getenv("LLM_THINK") is not None:
            value = os.environ["LLM_THINK"].lower()
            if value not in ("true", "false"):
                raise ValueError("LLM_THINK must be true or false")
            extra["think"] = value == "true"
        response = ollama.chat(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"),
            messages=[{"role": "user", "content": prompt}],
            options=options,
            **extra,
        )
        return response["message"]["content"].strip()
    from core.request import send_text_to_model_openai
    return send_text_to_model_openai(prompt, model="gpt-4.1", temperature=0.0000001)


def ask_agent(game, agent, rotation=0):
    """
    One independent model call for one player of one conflict.

    Returns (action, info). An unparsable reply counts as YIELD, is logged to
    fails.txt, and is flagged with info["parse_failed"] = True.
    """
    prompt = build_prompt(game, agent, rotation)
    reply = call_model(prompt)
    action = parse_action(reply)
    info = {"parse_failed": action is None, "reply": reply,
            "reasoning": extract_reasoning(reply)}          # NEW 10 Nov: logged
    if action is None:
        with open("fails.txt", "a") as f:
            f.write(f"[Conflict Parsing Error] Agent {agent + 1}, {game.kind} {game.subject}\n{reply}\n\n")
        action = YIELD
    print(f"Agent {agent + 1} ({game.kind} {game.subject}, rotation {rotation}): {action}")
    return action, info


# ===========================================================================
# NEW (10 Nov): matching form - "which agent should take it?"
# ===========================================================================
PROMPT_VARIANTS = ("current", "A", "B", "C")     # NEW 12 Nov (C: 15 Nov); "current" = 10 Nov prompt


def build_matching_prompt(game, agent, rotation=0, variant="current"):
    """
    The matching-form text agent `agent` sees for this conflict.

    rotation 0 lists the candidate agents in index order, rotation 1 in
    reverse order (Option A's replay flips it). With 2 players all four
    combinations are listed; with 3+ players the prompt lists the payoff of
    each agreement and states the disagreement rule (listing all n^n
    combinations would be 27 or 256 lines).

    Example (2 agents, goal conflict, rotation 0) - see the advisor doc:
        You are Agent 1 ... You and Agent 2 both want goal C.
        If Agent 1 takes goal C: Agent 1 needs 3 steps, Agent 2 ... (9 steps).
        ...
        you name Agent 1, Agent 2 names Agent 1 -> -9   (agreed: Agent 1 takes C)
        ...
        Which agent should take goal C?
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"variant must be one of {PROMPT_VARIANTS}")
    if variant != "current":                      # NEW 12 Nov
        return build_matching_prompt_variant(game, agent, rotation, variant)
    # ---- current (10 Nov) prompt, unchanged ----
    from game_theory.conflict_games import MatchingConflictGame
    mgame = MatchingConflictGame(game)
    me = agent + 1
    cands = list(game.players) if rotation == 0 else list(reversed(game.players))
    others = [i for i in cands if i != agent]
    labels = [f"Agent {i + 1}" for i in others]
    two = len(labels) == 1
    group = f"You and {labels[0]} both" if two else "You, " + ", ".join(labels[:-1]) + " and " + labels[-1] + " all"
    unreachable = game.context.get("unreachable", 1e9)

    if game.kind == "goal":
        thing = f"goal {game.subject}"
        verb, verbs = "take", "takes"
        intro = f"{group} want {thing}."
        rest = ("Only one of you can take it; the other goes to its next choice." if two
                else "Only one of you can take it; the others go to their next choice.")

        def if_takes(w):
            parts = [f"Agent {w + 1} needs {_fmt(game.a[w])} steps"]
            for j in game.players if rotation == 0 else reversed(game.players):
                if j == w:
                    continue
                if game.b[j] >= unreachable:
                    parts.append(f"Agent {j + 1} has no other goal left")
                else:
                    parts.append(f"Agent {j + 1} goes to its next choice ({_fmt(game.b[j])} steps)")
            return ", ".join(parts)
    else:
        r, c = game.subject
        thing = f"cell ({r}, {c})"
        verb, verbs = "step into", "steps into"
        intro = f"{group} want to step into {thing} this turn."
        rest = ("Only one of you can move there now; the other waits one step." if two
                else "Only one of you can move there now; the others wait one step.")

        def if_takes(w):
            parts = [f"Agent {w + 1} has {_fmt(game.a[w])} steps left after"]
            for j in game.players if rotation == 0 else reversed(game.players):
                if j != w:
                    parts.append(f"Agent {j + 1} waits ({_fmt(game.b[j])} steps left counting the wait)")
            return ", ".join(parts)

    lines = [
        f"You are Agent {me} in a team of robots on a grid. {intro}",
        rest,
        "You cannot communicate. The other agents are identical copies of you and are",
        "answering this same question at the same time. It only works if you "
        + ("both" if two else "all"),
        "name the same agent.",
        "",
    ]
    for w in cands:
        lines.append(f"If Agent {w + 1} {verbs} {thing}: {if_takes(w)}.")
    lines += ["", "You share one team payoff (minus the team's finishing time):"]

    if len(game.players) == 2:
        other = others[0]
        combos = [(w, w) for w in cands] + [(x, y) for x in cands for y in cands if x != y]
        for mine, theirs in combos:
            chosen = {agent: mine, other: theirs}
            profile = tuple(chosen[i] for i in game.players)
            if mine == theirs:
                note = f"agreed: Agent {mine + 1} {verbs} {thing}"
            elif mine == agent and theirs == other:
                note = "disagree: both head there, clash"
            elif mine == other and theirs == agent:
                note = "disagree: nobody " + ("takes it" if game.kind == "goal" else "moves")
            else:
                note = "disagree"
            lines.append(f"  you name Agent {mine + 1}, Agent {other + 1} names Agent {theirs + 1} "
                         f"-> {_fmt(mgame.payoff(profile))}   ({note})")
    else:
        for w in cands:
            profile = tuple(w for _ in game.players)
            lines.append(f"  everyone names Agent {w + 1} -> {_fmt(mgame.payoff(profile))}")
        worst = min(mgame.payoff(p) for p in mgame.profiles)
        lines += [
            "  any disagreement fails: agents that named themselves head there anyway",
            f"  and clash (about {_fmt(game.K)} extra steps), or, if nobody named themselves,",
            f"  it is wasted (about {_fmt(game.D)} extra steps). Worst case -> {_fmt(worst)}",
        ]

    opts = " or ".join(f'"Agent {w + 1}"' for w in cands)
    lines += [
        "",
        f"Which agent should {verb} {thing}? Reply in JSON:",
        f'{{"reasoning": "...", "answer": {opts}}}',
    ]
    return "\n".join(lines)


def build_matching_prompt_variant(game, agent, rotation, variant):
    """
    NEW (12 Nov): prompt variants A and B of the matching form; C added 15 Nov.

    Same facts, same payoffs, same rotation as the current prompt; only the
    presentation changes (see the module docstring). Payoffs are shown as
    the team finishing time in steps = minus the payoff, so lower is better.

    Variant A, 2 agents (worked example of the advisor doc):
        ...
        The whole team shares one result: the team's finishing time, in steps
        (the number of steps until every agent is done; lower is better):
          you name Agent 1, Agent 2 names Agent 1 -> 9 steps    (agreed: ...)
        ...
        Reply with JSON only, on a single line, with at most 2 sentences of reasoning:
    Variant B, 2 agents: the four outcomes as a grid, plus
        "Agent 2 sees exactly this same table, knows that you see it too, and is
         also trying to name the same agent as you."
    Variant C (NEW 15 Nov), 2 agents:
          you both name Agent 1 -> 28 steps
          you both name Agent 2 -> 20 steps
          any disagreement fails: if you both named yourselves, you both head there
          and clash -> 35 steps; if you both named the other, nobody takes it -> 35.5 steps
        + B's sentence, and the JSON reply starts with
        "team_steps": {"Agent 1": ..., "Agent 2": ...}. 3+ agents: A's list +
        B's sentence + the same JSON. Only "answer" is parsed.
    """
    from game_theory.conflict_games import MatchingConflictGame
    mgame = MatchingConflictGame(game)
    me = agent + 1
    cands = list(game.players) if rotation == 0 else list(reversed(game.players))
    others = [i for i in cands if i != agent]
    labels = [f"Agent {i + 1}" for i in others]
    two = len(labels) == 1
    group = f"You and {labels[0]} both" if two else "You, " + ", ".join(labels[:-1]) + " and " + labels[-1] + " all"
    unreachable = game.context.get("unreachable", 1e9)
    steps = lambda profile: _fmt(-mgame.payoff(profile))      # finishing time in steps

    if game.kind == "goal":
        thing, verb, verbs = f"goal {game.subject}", "take", "takes"
        intro = f"{group} want {thing}."
        rest = ("Only one of you can take it; the other goes to its next choice." if two
                else "Only one of you can take it; the others go to their next choice.")

        def if_takes(w):
            parts = [f"Agent {w + 1} needs {_fmt(game.a[w])} steps"]
            for j in cands:
                if j == w:
                    continue
                parts.append(f"Agent {j + 1} has no other goal left" if game.b[j] >= unreachable
                             else f"Agent {j + 1} goes to its next choice ({_fmt(game.b[j])} steps)")
            return ", ".join(parts)
    else:
        r, c = game.subject
        thing, verb, verbs = f"cell ({r}, {c})", "step into", "steps into"
        intro = f"{group} want to step into {thing} this turn."
        rest = ("Only one of you can move there now; the other waits one step." if two
                else "Only one of you can move there now; the others wait one step.")

        def if_takes(w):
            parts = [f"Agent {w + 1} has {_fmt(game.a[w])} steps left after"]
            for j in cands:
                if j != w:
                    parts.append(f"Agent {j + 1} waits ({_fmt(game.b[j])} steps left counting the wait)")
            return ", ".join(parts)

    lines = [
        f"You are Agent {me} in a team of robots on a grid. {intro}",
        rest,
        "You cannot communicate. " + (f"{labels[0]} is an identical copy of you and is" if two
                                      else "The other agents are identical copies of you and are"),
        "answering this same question at the same time. It only works if you "
        + ("both" if two else "all") + " name the same agent.",
        "",
    ]
    for w in cands:
        lines.append(f"If Agent {w + 1} {verbs} {thing}: {if_takes(w)}.")
    lines.append("")

    if two and variant == "B":
        other = others[0]
        lines += [
            "Team finishing time in steps (lower is better) for every combination of answers.",
            f"Rows = the agent you name, columns = the agent Agent {other + 1} names:",
            "",
            "|  | " + " | ".join(f"Agent {other + 1} names Agent {y + 1}" for y in cands) + " |",
            "|---|" + "---|" * len(cands),
        ]
        for x in cands:
            cells = []
            for y in cands:
                prof = tuple({agent: x, other: y}[i] for i in game.players)
                if x == y:
                    note = "agreed"
                elif x == agent:
                    note = "disagree: clash"
                else:
                    note = "disagree: nobody"
                cells.append(f"{steps(prof)} ({note})")
            lines.append(f"| You name Agent {x + 1} | " + " | ".join(cells) + " |")
    elif two and variant == "A":
        other = others[0]
        lines.append("The whole team shares one result: the team's finishing time, in steps")
        lines.append("(the number of steps until every agent is done; lower is better):")
        combos = [(w, w) for w in cands] + [(x, y) for x in cands for y in cands if x != y]
        for mine, theirs in combos:
            prof = tuple({agent: mine, other: theirs}[i] for i in game.players)
            if mine == theirs:
                note = f"agreed: Agent {mine + 1} {verbs} {thing}"
            elif mine == agent:
                note = "disagree: both head there, clash"
            else:
                note = "disagree: nobody " + ("takes it" if game.kind == "goal" else "moves")
            lines.append(f"  you name Agent {mine + 1}, Agent {other + 1} names Agent {theirs + 1} "
                         f"-> {steps(prof)} steps   ({note})")
    elif two:                                     # NEW 15 Nov: variant C, 2 agents
        other = others[0]
        lines.append("The whole team shares one result: the team's finishing time, in steps")
        lines.append("(the number of steps until every agent is done; lower is better):")
        for w in cands:
            lines.append(f"  you both name Agent {w + 1} -> {steps(tuple(w for _ in game.players))} steps")
        clash = steps(tuple({agent: agent, other: other}[i] for i in game.players))
        nobody = steps(tuple({agent: other, other: agent}[i] for i in game.players))
        lines += [
            "  any disagreement fails: if you both named yourselves, you both "
            + ("head there" if game.kind == "goal" else "try to step in"),
            f"  and clash -> {clash} steps; if you both named the other, nobody "
            + ("takes it" if game.kind == "goal" else "moves") + f" -> {nobody} steps",
        ]
    else:
        lines.append("The whole team shares one result: the team's finishing time, in steps")
        lines.append("(the number of steps until every agent is done; lower is better):")
        for w in cands:
            lines.append(f"  everyone names Agent {w + 1} -> {steps(tuple(w for _ in game.players))} steps")
        worst = max(-mgame.payoff(p) for p in mgame.profiles)
        lines += [
            "  any disagreement fails: agents that named themselves head there anyway",
            f"  and clash (about {_fmt(game.K)} extra steps), or, if nobody named themselves,",
            f"  it is wasted (about {_fmt(game.D)} extra steps). Worst case -> {_fmt(worst)} steps",
        ]

    if variant in ("B", "C"):                     # C: NEW 15 Nov
        who = labels[0] if two else "The other agents"
        verb_see = "sees" if two else "see"
        lines += ["", f"{who} {verb_see} exactly this same information, know{'s' if two else ''} that you see it too, "
                      f"and {'is' if two else 'are'} also trying to name the same agent as you."]

    opts = " or ".join(f'"Agent {w + 1}"' for w in cands)
    if variant == "C":                            # NEW 15 Nov
        times = ", ".join(f'"Agent {w + 1}": ...' for w in cands)
        lines += [
            "",
            f"Which agent should {verb} {thing}? Reply with JSON only, on a single line. First give the",
            f"team finishing time if you {'both' if two else 'all'} name each agent, "
            "then at most 2 sentences of reasoning:",
            f'{{"team_steps": {{{times}}}, "reasoning": "...", "answer": {opts}}}',
        ]
        return "\n".join(lines)
    lines += [
        "",
        f"Which agent should {verb} {thing}? Reply with JSON only, on a single line, with",
        "at most 2 sentences of reasoning:",
        f'{{"reasoning": "...", "answer": {opts}}}',
    ]
    return "\n".join(lines)


def parse_name(text, players):
    """
    Read the named agent from the model's reply (matching form).

    Uses the JSON "answer" field (e.g. "Agent 2" or 2). Returns the 0-based
    agent index, or None if the reply has no usable answer or names an agent
    that is not in this conflict. The reasoning text is never searched,
    because it mentions several agents.
    """
    if not text:
        return None
    value = None
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            value = str(json.loads(match.group(0)).get("answer", ""))
    except Exception:
        value = None
    if not value:
        # Broken JSON: look for the "answer" field directly
        found = re.search(r'"answer"\s*:\s*"?\s*(?:Agent\s*)?(\d+)', text, re.IGNORECASE)
        value = found.group(1) if found else None
    if not value:
        return None
    digits = re.findall(r"\d+", value)
    if len(digits) != 1:
        return None
    idx = int(digits[0]) - 1
    return idx if idx in players else None


def ask_agent_matching(game, agent, rotation=0, variant="current"):
    """
    One independent model call for one player, matching form.

    Returns (named agent index, info). An unparsable or invalid answer is
    stored as -1 ("named nobody", so it can never be an agreement), logged to
    fails.txt and flagged with info["parse_failed"] = True - the matching
    counterpart of "unparsed = YIELD" in the claim/yield form.
    """
    prompt = build_matching_prompt(game, agent, rotation, variant)   # variant: NEW 12 Nov
    reply = call_model(prompt)
    name = parse_name(reply, game.players)
    info = {"parse_failed": name is None, "reply": reply,
            "reasoning": extract_reasoning(reply)}          # NEW 10 Nov: logged
    if name is None:
        with open("fails.txt", "a") as f:
            f.write(f"[Matching Parsing Error] Agent {agent + 1}, {game.kind} {game.subject}\n{reply}\n\n")
        name = -1
    print(f"Agent {agent + 1} ({game.kind} {game.subject}, rotation {rotation}): "
          f"{'no answer' if name < 0 else f'Agent {name + 1}'}")
    return name, info
