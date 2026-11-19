"""Export matched LLM/rule comparisons and audit the fixed priority rule."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from baseline_support import SOURCE_KEY, read_rows, trial_rows

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "BASELINE_PLAN.json").read_text())


def metrics(rows):
    rows = list(rows)
    ok = [r for r in rows if not int(r["Failed"])]
    return {"trials": len(rows), "failures": len(rows) - len(ok),
            "mean_successful_steps": sum(int(r["Steps"]) for r in ok) / len(ok) if ok else None,
            "optimal_trials": sum(not int(r["Failed"]) and int(r["Steps"]) == int(r["TrueOptimal"])
                                  for r in rows)}


def main():
    summary = []
    for job in PLAN["jobs"]:
        d = Path(job["directory"])
        rows = trial_rows(d / f"results/{job['result_key']}_team_results.csv")
        current = metrics(rows.values())
        state = json.loads((d / "STATUS.json").read_text())
        source = trial_rows(Path(job["source_job_directory"]) / f"results/{SOURCE_KEY}_team_results.csv")
        original = metrics(source[k] for k in rows) if job["kind"] == "paired_rule" else None
        worlds = Counter(k[0] for k in rows)
        conflicts = read_rows(d / f"results/{job['result_key']}_conflicts.csv")
        for conflict in conflicts:
            players = [int(value[1:]) for value in conflict["players"].split()]
            actions = conflict["final_actions"].split()
            claimers = [player for player, action in zip(players, actions) if action == "CLAIM"]
            assert claimers == [min(players)], "Priority rule was violated"
            assert all(name == f"A{min(players)}" for name in conflict["final_names"].split())
            assert not int(conflict.get("coin_fallback") or 0)
            assert not int(conflict["nash_fallback"])
        audit = {"audited_utc": datetime.now(timezone.utc).isoformat(),
                 "conflicts_checked": len(conflicts), "lowest_competing_id_always_wins": True,
                 "random_fallbacks": 0, "new_model_calls": state.get("model_calls", 0)}
        (d / "PRIORITY_RULE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
        comparable = [key for key in rows if job["kind"] == "paired_rule"
                      and not int(rows[key]["Failed"]) and not int(source[key]["Failed"])]
        summary.append({"Baseline": job["id"], "Model": job["model"], "Kind": job["kind"],
            "Phase": state["phase"], "CompleteWorlds": sum(n == 5 for n in worlds.values()),
            "Trials": len(rows), "RuleMeanSuccessfulSteps": current["mean_successful_steps"],
            "RuleOptimalTrials": current["optimal_trials"], "RuleTaskFailures": current["failures"],
            "LLMMeanSuccessfulStepsSameTrials": original["mean_successful_steps"] if original else "",
            "LLMOptimalTrialsSameTrials": original["optimal_trials"] if original else "",
            "LLMTaskFailuresSameTrials": original["failures"] if original else "",
            "BothSuccessfulPairs": len(comparable),
            "MeanRuleMinusLLMStepsBothSuccessful":
                sum(int(rows[k]["Steps"]) - int(source[k]["Steps"]) for k in comparable) / len(comparable)
                if comparable else "", "RuleConflicts": len(conflicts), "NewModelCalls": state.get("model_calls", 0)})
    with (ROOT / "baseline_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (ROOT / "baseline_summary.json").write_text(json.dumps({
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "Each row compares the same exact case/trial pairs and initial rankings. Means exclude failed tasks; failure counts are separate. Deterministic greedy repeats are not independent samples.",
        "baselines": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
