"""Verify published file checksums, all 7500 trials, and paired initial rankings."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA / 'tools/original_baseline_tools'))
from baseline_support import ranking_digest, ranking_rows, read_rows, trial_rows

EXPECTED = {(f'case_{case}', trial) for case in range(1, 101) for trial in range(1, 6)}


def main():
    manifest = json.loads((DATA / 'MANIFEST.json').read_text())
    for relative, metadata in manifest['files'].items():
        path = DATA / relative
        assert path.is_file(), f'Missing {relative}'
        assert path.stat().st_size == metadata['bytes'], f'Size changed: {relative}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256'], f'Changed: {relative}'
    models = [r for r in json.loads((DATA / 'provenance/model_runs.json').read_text()) if r['production_included']]
    baselines = json.loads((DATA / 'provenance/baseline_runs.json').read_text())
    summary = {r['Configuration']: r for r in read_rows(DATA / 'reports/summary.csv')}
    total = 0
    reference = {}
    for group, jobs in [('models', models), ('baselines', baselines)]:
        for job in jobs:
            directory = DATA / group / job['id']
            key = job.get('result_key', 'agent_rank_once_game_llm_matching_C_coin')
            rows = trial_rows(directory / 'results' / (key + '_team_results.csv'))
            assert set(rows) == EXPECTED and len(rows) == 500, job['id']
            success = [r for r in rows.values() if not int(r['Failed'])]
            actual_mean = mean(int(r['Steps']) for r in success)
            reported = summary[job['id']]
            assert abs(actual_mean - float(reported['MeanSuccessfulSteps'])) < 1e-10
            assert len(rows) - len(success) == int(reported['FailedTrials'])
            assert sum(int(r['Steps']) == int(r['TrueOptimal']) for r in success) == int(reported['OptimalTrials'])
            for pair, row in rows.items():
                values = (int(row['TrueOptimal']), int(row['Greedy']))
                assert reference.setdefault(pair, values) == values
                assert int(row['Failed']) in (0, 1)
                assert not int(row['Failed']) or int(row['Steps']) == 100
            conflicts = read_rows(directory / 'results' / (key + '_conflicts.csv'))
            assert len(conflicts) == int(reported['Conflicts'])
            assert sum(int(r.get('coin_fallback') or 0) for r in conflicts) == int(reported['CoinFallbacks'])
            if group == 'baselines':
                assert all(not int(r.get('coin_fallback') or 0) and not int(r['nash_fallback']) for r in conflicts)
                if job['kind'] == 'paired_rule':
                    source = DATA / 'models' / job['source_job'] / 'results'
                    rankings = ranking_rows(source / 'agent_rank_once_game_llm_matching_C_coin_reasoning.csv')
                    used = [json.loads(line) for line in (directory / 'paired_rankings.jsonl').read_text().splitlines()]
                    assert len(used) == 500
                    assert {(r['Case'], int(r['Trial'])) for r in used} == EXPECTED
                    for row in used:
                        recorded = rankings[(row['Case'], int(row['Trial']))]
                        assert row['rankings_sha256'] == ranking_digest(recorded)
                        assert row['rankings'] == {str(a): r['ranking'] for a, r in recorded.items()}
                    for conflict in conflicts:
                        players = [int(p[1:]) for p in conflict['players'].split()]
                        winners = [p for p, action in zip(players, conflict['final_actions'].split()) if action == 'CLAIM']
                        assert winners == [min(players)]
            total += len(rows)
    assert total == 7500 and len(models) == 7 and len(baselines) == 8
    print(f'PASS: {len(manifest["files"])} file checksums, 7500 complete trial records, identical maps/references, exact paired rankings, and fixed priority decisions.')


if __name__ == '__main__':
    main()
