"""Replay saved-ranking controls on Linux without running any LLM inference."""
import argparse
import json
import shutil
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='New directory outside the published dataset')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--validate-only', action='store_true',
                        help='Verify original trajectories and import/configuration; do not rerun 4000 controls')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('The archived runner uses Linux process locks and /proc; run it on Linux.')
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.expanduser().resolve()
    if output.is_relative_to(DATA) or DATA.is_relative_to(output):
        parser.error('--output must be outside the published dataset')
    output.mkdir(parents=True, exist_ok=False)
    profiles = json.loads((DATA / 'provenance/source_profiles.json').read_text())
    jobs = json.loads((DATA / 'provenance/baseline_runs.json').read_text())
    original = json.loads((DATA / 'provenance/original_baseline_plan.json').read_text())
    for profile, metadata in profiles.items():
        source = output / 'sources' / profile
        source.mkdir(parents=True)
        with zipfile.ZipFile(DATA / metadata['archive']) as archive:
            for member in archive.namelist():
                candidate = (source / member).resolve()
                if not candidate.is_relative_to(source):
                    raise ValueError('Unsafe path in source snapshot')
            archive.extractall(source)
    relocated = []
    for job in jobs:
        directory = output / 'jobs' / job['id']
        (directory / 'results').mkdir(parents=True)
        relocated.append({**job, 'directory': str(directory),
                          'source_directory': str(output / 'sources' / job['source_profile']),
                          'source_hashes': str(DATA / profiles[job['source_profile']]['file_hashes']),
                          'source_job_directory': str(DATA / 'models' / (
                              job['source_job'] or 'ministral3_8b_instruct_multiline_4k')),
                          'host': socket.gethostname()})
    plan = {**original, 'root': str(output), 'campaign_root': str(DATA),
            'python': sys.executable, 'host': socket.gethostname(), 'workers': args.workers,
            'local_work_root': str(output / 'work'), 'jobs': relocated,
            'benchmark_job_directory': str(DATA / 'models/gemma4_12b_nothink')}
    (output / 'BASELINE_PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    for source in (DATA / 'tools/original_baseline_tools').glob('*.py'):
        shutil.copy2(source, output / source.name)
    if args.validate_only:
        for job in relocated:
            subprocess.run([sys.executable, str(output / 'baseline_worker.py'),
                            job['id'], '--validate-only'], check=True, cwd=output)
        print('All eight controls validated with zero model calls. Output:', output)
        return
    subprocess.run([sys.executable, str(output / 'baseline_controller.py')], check=True, cwd=output)
    state = json.loads((output / 'CONTROLLER_STATUS.json').read_text())
    if set(state['exit_codes']) != {job['id'] for job in relocated} or any(state['exit_codes'].values()):
        raise RuntimeError('Some replay workers failed; inspect their evaluation.log files')
    subprocess.run([sys.executable, str(output / 'summarize_baselines.py')], check=True, cwd=output)
    print('All 4000 controls rerun with zero model calls. Output:', output)


if __name__ == '__main__':
    main()
