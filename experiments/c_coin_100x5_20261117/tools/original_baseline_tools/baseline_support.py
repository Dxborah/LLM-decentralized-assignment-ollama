"""Read completed source trials and replay their recorded decisions exactly."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SOURCE_KEY = "agent_rank_once_game_llm_matching_C_coin"
csv.field_size_limit(16 * 1024 * 1024)


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def trial_rows(path):
    result = {}
    for row in read_rows(path):
        required = ("Case", "Trial", "Steps", "Failed", "TrueOptimal", "Collisions")
        if any(row.get(k) in (None, "") for k in required):
            continue
        key = (row["Case"], int(row["Trial"]))
        if key in result:
            raise ValueError(f"Duplicate completed trial: {key}")
        result[key] = row
    return result


def ranking_rows(path):
    result = defaultdict(dict)
    for row in read_rows(path):
        if row.get("decision") != "ranking":
            continue
        if any(row.get(k) in (None, "") for k in ("Case", "Trial", "agent", "answer")):
            continue
        key = (row["Case"], int(row["Trial"]))
        agent = int(row["agent"])
        ranking = [] if row["answer"] == "none" else row["answer"].split()
        if agent in result[key] and result[key][agent]["ranking"] != ranking:
            raise ValueError(f"Different rankings recorded for {key}, agent {agent}")
        result[key][agent] = {"ranking": ranking,
                              "reasoning": row.get("reasoning") or "",
                              "explanation": row.get("reply") or ""}
    return dict(result)


def inputs_ready(trial, rankings, expected_agents):
    return trial is not None and set(rankings) == set(expected_agents)


def ranking_digest(rankings):
    values = {str(agent): item["ranking"] for agent, item in sorted(rankings.items())}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class RankingReplay:
    def __init__(self):
        self.current = None
        self.used = []

    def select(self, **kwargs):
        agent = kwargs["agent_id"]
        if self.current is None or agent not in self.current:
            raise RuntimeError(f"Missing recorded ranking for agent {agent}; refusing to substitute")
        row = self.current[agent]
        self.used.append(agent)
        return list(row["ranking"]), row["explanation"], row["reasoning"]


class ConflictReplay:
    """Reconstruct an original episode, checking the conflict costs en route."""
    def __init__(self, records):
        self.records = records
        self.index = 0

    def __call__(self, game, decider, rotation=0, **kwargs):
        if self.index >= len(self.records):
            raise AssertionError("Original replay produced an extra conflict")
        record = dict(self.records[self.index])
        self.index += 1
        # The source CSV schema omits this in-memory discriminator.
        # It determines which agents wait after disagreement in matching mode.
        record["form"] = "matching"
        assert record["players"] == " ".join(f"A{i + 1}" for i in game.players)
        assert record["kind"] == game.kind
        assert record["subject"] == str(game.subject)
        assert [float(v) for v in record["a"].split()] == [float(game.a[i]) for i in game.players]
        assert [float(v) for v in record["b"].split()] == [float(game.b[i]) for i in game.players]
        record["clash_rounds"] = int(record["clash_rounds"])
        actions = tuple(record["final_actions"].split())
        assert len(actions) == len(game.players)
        assert all(action in ("CLAIM", "YIELD") for action in actions)
        return actions, record


def normalized_moves(rows, fields):
    return [{field: "" if row.get(field) is None else str(row[field])
             for field in fields} for row in rows]
