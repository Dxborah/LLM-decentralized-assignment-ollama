"""CPU-only matched rule controls; follow new completed source trials."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time

from baseline_support import (SOURCE_KEY, ConflictReplay, RankingReplay,
                              inputs_ready, normalized_moves, ranking_digest,
                              ranking_rows, read_rows, trial_rows)

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "BASELINE_PLAN.json").read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


class Runtime:
    def __init__(self, job):
        self.job = job
        self.directory = Path(job["directory"])
        self.source = Path(job["source_directory"])
        self.source_job = Path(job["source_job_directory"])
        self.results = self.directory / "results"
        self.key = job["result_key"]
        self.work = Path(PLAN["local_work_root"]) / job["id"]
        self.work.mkdir(parents=True, exist_ok=True)
        os.chdir(self.work)
        os.environ.update(RESULTS_DIR=str(self.results), LLM_BACKEND="ollama",
                          OLLAMA_HOST="http://127.0.0.1:9", OLLAMA_BASE_URL="http://127.0.0.1:9/v1",
                          LLM_INPUT_MODE="text", MPLBACKEND="Agg", PYTHONNOUSERSITE="1")
        sys.path.insert(0, str(self.source))
        from game_theory import agent_rank_once_game, eval_game, llm_player
        from core import request
        from agents import agent_greedy
        import yaml
        self.agent, self.ev, self.greedy = agent_rank_once_game, eval_game, agent_greedy
        self.model_calls = 0
        def forbidden(*args, **kwargs):
            self.model_calls += 1
            raise RuntimeError("A CPU baseline attempted a model call")
        self.agent.send_image_to_model_openai_logprobs = forbidden
        request._client = forbidden
        request.ollama.chat = forbidden
        llm_player.call_model = forbidden
        self.agent.plot_grid_unassigned_labeled = lambda *args, **kwargs: None
        self.ev.plot_grid_unassigned_labeled = lambda *args, **kwargs: None
        self.ev.OUTPUT_DIR = str(self.results)
        self.ev.VISUALIZE = False
        self.ev.TRIALS_PER_CASE = 5
        self.expected_agents = {}
        for case in PLAN["cases"]:
            config = yaml.safe_load((self.source / f"configs/eval/{case}.yaml").read_text())
            self.expected_agents[case] = [i + 1 for i, p in enumerate(config["agents"]) if p is not None]
        benchmark_source = Path(PLAN["benchmark_job_directory"])
        benchmark_rows = trial_rows(benchmark_source / f"results/{SOURCE_KEY}_team_results.csv")
        self.benchmarks = {}
        for (case, _), row in benchmark_rows.items():
            value = (int(row["TrueOptimal"]), int(row["Greedy"]))
            if case in self.benchmarks:
                assert self.benchmarks[case] == value
            self.benchmarks[case] = value
        assert set(self.benchmarks) == set(PLAN["cases"])
        # These reference values were computed on the identical hashed maps.
        # Standalone greedy controls below actually rerun the algorithm five times.
        self.ev.true_optimum = lambda path: self.benchmarks[Path(path).stem][0]
        self.ev.greedy_steps = lambda path: self.benchmarks[Path(path).stem][1]
        self.ranking_replay = RankingReplay()
        self.agent.select_target = self.ranking_replay.select
        self.original_play = self.agent.play_conflict
        self.current_inputs = {}
        self.source_trials = {}
        self.started = now()
        self.original_summary = self.ev.write_summary
        def summary_and_status(*args, **kwargs):
            self.original_summary(*args, **kwargs)
            self.status("running")
        self.ev.write_summary = summary_and_status

    def code_verified(self):
        hashes = json.loads(Path(self.job["source_hashes"]).read_text())
        assert all(hashlib.sha256((self.source / name).read_bytes()).hexdigest() == digest
                   for name, digest in hashes.items()), "frozen source changed"
        assert all(hashlib.sha256((self.source / name).read_bytes()).hexdigest() == digest
                   for name, digest in PLAN["map_hashes"].items()), "worlds differ"

    def output_trials(self):
        return trial_rows(self.results / f"{self.key}_team_results.csv")

    def status(self, phase, **extra):
        rows = list(self.output_trials().values())
        counts = defaultdict(int)
        for row in rows:
            counts[row["Case"]] += 1
        save(self.directory / "STATUS.json", {
            "id": self.job["id"], "host": socket.gethostname(), "model": self.job["model"],
            "kind": self.job["kind"], "worker_pid": os.getpid(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "phase": phase, "started_utc": self.started, "updated_utc": now(),
            "completed_trials": len(rows), "expected_trials": 500,
            "worlds_complete": sum(count == 5 for count in counts.values()),
            "task_failures": sum(int(row["Failed"]) for row in rows),
            "model_calls": self.model_calls, "source_job": self.job.get("source_job"), **extra})

    def load_source(self):
        self.source_trials = trial_rows(self.source_job / f"results/{SOURCE_KEY}_team_results.csv")
        self.current_inputs = ranking_rows(self.source_job / f"results/{SOURCE_KEY}_reasoning.csv")
        return {key for key, row in self.source_trials.items()
                if key[0] in self.expected_agents and key[1] in range(1, 6)
                and inputs_ready(row, self.current_inputs.get(key, {}), self.expected_agents[key[0]])}

    def greedy_episode(self, config_path, max_steps=100, move_log=None, **kwargs):
        log = self.work / "greedy_moves.csv"
        log.unlink(missing_ok=True)
        result = self.greedy.run(config_path=config_path, log_path=str(log), max_steps=max_steps)
        case = Path(config_path).stem
        assert result[0] == self.benchmarks[case][1], "Greedy differs from the campaign reference"
        if move_log is not None:
            for row in read_rows(log):
                move_log.append({"step": row["step"], "agent_id": row["agent_id"],
                                 "position_before": row["position_before"],
                                 "position_after": row["position_after"],
                                 "chosen_direction": row["chosen_direction"],
                                 "explanation": "Greedy target " + row.get("target_goal", "")})
        return result

    def episode(self, **kwargs):
        if self.job["kind"] == "greedy":
            return self.greedy_episode(**kwargs)
        case = Path(kwargs["config_path"]).stem
        seed = kwargs["fallback_seed"]
        trial = next(t for t in range(1, 6)
                     if self.ev.trial_fallback_seed(PLAN["base_seed"], case, t) == seed)
        rankings = self.current_inputs[(case, trial)]
        self.ranking_replay.current = rankings
        self.ranking_replay.used = []
        assert kwargs["decider"] == "rule"
        result = self.agent.run(**kwargs)
        assert sorted(self.ranking_replay.used) == self.expected_agents[case]
        record = {"Case": case, "Trial": trial, "source_job": self.job["source_job"],
                  "rankings": {str(a): row["ranking"] for a, row in sorted(rankings.items())},
                  "rankings_sha256": ranking_digest(rankings), "recorded_utc": now()}
        with (self.directory / "paired_rankings.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return result

    def validate_controls(self):
        self.code_verified()
        if self.job["kind"] == "greedy":
            cases = ["case_1", "case_2"]
            cases += [c for c, value in self.benchmarks.items() if value[1] >= 100][:1]
            checked = []
            with redirect_stdout(io.StringIO()):
                for case in cases:
                    result = self.greedy_episode(str(self.source / f"configs/eval/{case}.yaml"))
                    checked.append({"Case": case, "Steps": result[0], "Failed": result[2]})
            proof = {"verified_utc": now(), "reference_steps_match": True,
                     "controls": checked, "model_calls": self.model_calls}
        else:
            ready = self.load_source()
            if not ready:
                raise RuntimeError("No complete source trial is available to validate")
            conflicts = defaultdict(list)
            for row in read_rows(self.source_job / f"results/{SOURCE_KEY}_conflicts.csv"):
                if row.get("Trial") and row.get("final_actions"):
                    conflicts[(row["Case"], int(row["Trial"]))].append(row)
            samples = [k for k in [("case_1", 1), ("case_2", 1)] if k in ready]
            for predicate in [lambda k: int(self.source_trials[k]["Failed"]),
                              lambda k: any(int(r.get("coin_fallback") or 0) for r in conflicts[k]),
                              lambda k: any(r["kind"] == "cell" for r in conflicts[k])]:
                chosen = next((k for k in sorted(ready) if predicate(k) and k not in samples), None)
                if chosen:
                    samples.append(chosen)
            moves = defaultdict(list)
            for row in read_rows(self.source_job / f"results/{SOURCE_KEY}_moves.csv"):
                if row.get("Trial") and (row["Case"], int(row["Trial"])) in samples:
                    moves[(row["Case"], int(row["Trial"]))].append(row)
            checked = []
            for case, trial in samples:
                original = self.source_trials[(case, trial)]
                replay = ConflictReplay(conflicts[(case, trial)])
                self.agent.play_conflict = replay
                self.ranking_replay.current = self.current_inputs[(case, trial)]
                self.ranking_replay.used = []
                actual_moves = []
                try:
                    with redirect_stdout(io.StringIO()):
                        result = self.agent.run(
                            config_path=str(self.source / f"configs/eval/{case}.yaml"),
                            image_path=str(self.work / "unused.png"), max_steps=100,
                            decider="llm", game_form="matching", prompt_variant="C",
                            fallback="coin", rotation=(trial - 1) % 2,
                            fallback_seed=self.ev.trial_fallback_seed(PLAN["base_seed"], case, trial),
                            conflict_log=[], move_log=actual_moves, reasoning_log=[])
                finally:
                    self.agent.play_conflict = self.original_play
                expected = (int(original["Steps"]), int(original["Optimal"]),
                            bool(int(original["Failed"])), int(original["Collisions"]))
                assert tuple(result) == expected, (case, trial, result, expected)
                assert replay.index == len(replay.records), "Original replay missed a conflict"
                fields = [k for k in self.ev.MOVE_FIELDS if k not in ("Case", "Trial")]
                assert normalized_moves(actual_moves, fields) == normalized_moves(moves[(case, trial)], fields), \
                    f"Original trajectory changed: {case}/{trial}"
                assert sorted(self.ranking_replay.used) == self.expected_agents[case]
                checked.append({"Case": case, "Trial": trial, "Steps": result[0],
                                "Failed": result[2], "conflicts_replayed": replay.index,
                                "moves_match": True,
                                "rankings_sha256": ranking_digest(self.ranking_replay.current)})
            proof = {"verified_utc": now(), "source_outcomes_match": True,
                     "source_trajectories_match": True, "controls": checked,
                     "model_calls": self.model_calls}
        assert self.model_calls == 0
        save(self.directory / "CONTROL_REPLAY_VERIFIED.json", proof)
        return proof

    def write_comparison(self):
        if self.job["kind"] != "paired_rule":
            return
        baseline = self.output_trials()
        path = self.directory / "paired_comparison.csv"
        with path.open("w", newline="") as stream:
            fields = ["Case", "Trial", "LLMSteps", "RuleSteps", "RuleMinusLLMSteps",
                      "LLMFailed", "RuleFailed", "TrueOptimal", "RankingsSHA256"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key, row in sorted(baseline.items()):
                original = self.source_trials[key]
                writer.writerow({"Case": key[0], "Trial": key[1], "LLMSteps": original["Steps"],
                                 "RuleSteps": row["Steps"],
                                 "RuleMinusLLMSteps": int(row["Steps"]) - int(original["Steps"]),
                                 "LLMFailed": original["Failed"], "RuleFailed": row["Failed"],
                                 "TrueOptimal": row["TrueOptimal"],
                                 "RankingsSHA256": ranking_digest(self.current_inputs[key])})

    def finish(self):
        rows = self.output_trials()
        expected = {(case, t) for case in PLAN["cases"] for t in range(1, 6)}
        assert set(rows) == expected and len(rows) == 500
        self.code_verified()
        conflicts = read_rows(self.results / f"{self.key}_conflicts.csv")
        assert all(r["decider"] == "rule" and int(r["agreed"]) == 1 for r in conflicts)
        assert all(not int(r.get("coin_fallback") or 0) and not int(r["nash_fallback"])
                   for r in conflicts)
        proof = {"verified_utc": now(), "unique_trials": 500,
                 "expected_case_trial_pairs_match": True, "source_hashes_match": True,
                 "map_hashes_match": True, "model_calls": self.model_calls,
                 "coin_fallbacks": 0, "nash_fallbacks": 0,
                 "kind": self.job["kind"], "actual_runs_not_copied": True}
        if self.job["kind"] == "paired_rule":
            records = [json.loads(line) for line in (self.directory / "paired_rankings.jsonl").read_text().splitlines()]
            used = {(r["Case"], r["Trial"]): r for r in records}
            assert set(used) == expected
            assert all(used[k]["rankings_sha256"] == ranking_digest(self.current_inputs[k]) for k in expected)
            source_proof = json.loads((self.source_job / "VERIFIED_RESULTS.json").read_text())
            assert source_proof["source_hashes_match"] and source_proof["expected_case_trial_pairs_match"]
            proof.update(exact_source_rankings_match=True, source_job=self.job["source_job"])
            (self.directory / "source_team_results_snapshot.csv").write_bytes(
                (self.source_job / f"results/{SOURCE_KEY}_team_results.csv").read_bytes())
        else:
            assert all(int(r["Steps"]) == self.benchmarks[k[0]][1] for k, r in rows.items())
            proof.update(deterministic_repeats_match=True, campaign_greedy_reference_match=True)
        assert self.model_calls == 0
        save(self.directory / "VERIFIED_RESULTS.json", proof)
        self.status("complete", returncode=0, finished_utc=now())
        save(self.directory / "RUN_COMPLETE.json", json.loads((self.directory / "STATUS.json").read_text()))

    def run(self, validate_only=False):
        self.status("validating")
        proof = self.validate_controls()
        print("CONTROL_REPLAY_VERIFIED", self.job["id"], json.dumps(proof), flush=True)
        if validate_only:
            self.status("validated")
            return
        save(self.directory / "RUN_METADATA.json", {
            "job": self.job, "started_utc": self.started, "source_directory": str(self.source),
            "maps": PLAN["map_hashes"], "max_steps": 100, "trials_per_case": 5,
            "initial_rankings": "exact saved model rankings" if self.job["kind"] == "paired_rule" else "BFS distance",
            "conflict_rule": "lowest competing agent ID wins", "model_calls_allowed": False,
            "rendering": "disabled only; movement code retained", "seed": PLAN["base_seed"],
            "method": PLAN["method"], "sampling_note": PLAN["sampling_note"]})
        expected = {(c, t) for c in PLAN["cases"] for t in range(1, 6)}
        while True:
            completed = set(self.output_trials())
            available = expected if self.job["kind"] == "greedy" else self.load_source()
            missing = available - completed
            if missing:
                cases = {case: (str(self.source / f"configs/eval/{case}.yaml"),
                                sorted(t for c, t in missing if c == case))
                         for case in PLAN["cases"] if any(c == case for c, _ in missing)}
                self.status("running", available_source_trials=len(available))
                self.ev.evaluate_team(self.key, self.episode, cases, "rule",
                                      self.ev.CLASH_FACTOR, self.ev.YIELD_DELAY_FACTOR,
                                      "matching", "C", "coin", PLAN["base_seed"])
                self.write_comparison()
            if set(self.output_trials()) == expected:
                if self.job["kind"] == "greedy" or (self.source_job / "RUN_COMPLETE.json").exists():
                    self.finish()
                    return
            source_state = json.loads((self.source_job / "STATUS.json").read_text())
            if source_state["phase"] == "complete" and available != expected:
                raise RuntimeError("Completed source is missing saved rankings for some expected trials")
            self.status("waiting_source", available_source_trials=len(available),
                        source_phase=source_state["phase"])
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    assert socket.gethostname().lower() == PLAN["host"].lower()
    job = next(j for j in PLAN["jobs"] if j["id"] == args.job_id)
    directory = Path(job["directory"])
    lock = (directory / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / "RUN_COMPLETE.json").exists():
        print("Already complete", args.job_id)
        return
    os.nice(10)
    runtime = None
    try:
        runtime = Runtime(job)
        runtime.run(args.validate_only)
    except Exception as error:
        if runtime is not None:
            runtime.status("failed", error=str(error), finished_utc=now())
        save(directory / "RUN_FAILED.json", {"error": str(error), "failed_utc": now(),
                                             "worker_pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
