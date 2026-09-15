"""Build the report's historical first-five references from saved RoboLab logs."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path


SOURCES = (
    ('pi05_only', 'π0.5 only', 'pi05', '#e9a150',
     'robolab120_pi05_jointpos_specific_10ep_20260603'),
    ('cosmos_nano_policy', 'Cosmos3-Nano-Policy', 'cosmos3', '#9270c4',
     'robolab120_cosmos3_official_specific_10ep_20260603'),
    ('dreamzero', 'DreamZero', 'dreamzero', '#7791a8',
     'robolab120_dreamzero_specific_10ep_torchattn_20260603'),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_first(root, tasks, expected_policy, limit=5):
    """Order by run/episode/env_id, then verify every selected terminal log."""
    root = Path(root)
    source = root / 'episode_results.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    result = []
    for task in tasks:
        candidates = [r for r in rows if r['task_name'] == task]
        keys = [(r['run'], r['episode'], r['env_id']) for r in candidates]
        if len(set(keys)) != len(keys):
            raise ValueError(f'{task}: duplicate episode identifiers')
        if len(candidates) < limit:
            raise ValueError(f'{task}: needs {limit} episodes, found {len(candidates)}')
        selected = sorted(candidates, key=lambda r: (r['run'], r['episode'], r['env_id']))[:limit]
        trials = []
        for row in selected:
            if row['policy'] != expected_policy or row['instruction_type'] != 'specific':
                raise ValueError(f'{task}: unexpected policy or instruction protocol')
            if type(row['success']) is not bool:
                raise ValueError(f'{task}: missing or non-boolean outcome')
            log_file = f"{task}/log_{row['run']}_env{row['env_id']}.json"
            log = json.loads((root / log_file).read_text())
            if (log['task'], log['run'], log['env_id'], log['success'], log['final_step']) != (
                    task, row['run'], row['env_id'], row['success'], row['episode_step']):
                raise ValueError(f'{task}: episode summary disagrees with terminal log')
            trials.append({key: row[key] for key in ('run', 'episode', 'env_id', 'success')})
            trials[-1].update(steps=row['episode_step'], log_file=log_file,
                              log_sha256=sha256(root / log_file))
        result.append({'task': task, 'episodes': limit, 'available_episodes': len(candidates),
                       'successes': sum(r['success'] for r in selected), 'trials': trials,
                       'env_cfg_sha256': sha256(root / task / 'env_cfg.json')})
    return {'source_run': root.name, 'source_file': source.name,
            'source_sha256': sha256(source), 'source_episodes': len(rows), 'tasks': result}


def augment(report, baseline_root):
    data = copy.deepcopy(report)
    tasks = [t['task'] for t in data['tasks']]
    audit = {'schema': 'robolab.report.first_five_baselines.v1',
             'selection': 'First five ordered by run, episode, env_id; success is not a selection criterion.',
             'source_cohort': '2026-06-03 historical specific-instruction evaluations',
             'initial_states_paired': False, 'videos_added': False, 'methods': []}
    for method_id, label, policy, color, folder in SOURCES:
        selection = select_first(Path(baseline_root) / folder, tasks, policy)
        selection.update(method=method_id, label=label, policy=policy)
        audit['methods'].append(selection)
        for task, counts in zip(data['tasks'], selection['tasks']):
            previous = task['successes'].get(method_id)
            if previous is not None and previous != counts['successes']:
                raise ValueError(f'{method_id}/{task["task"]}: would change an existing result')
            task['successes'][method_id] = counts['successes']
        method = next((m for m in data['methods'] if m['id'] == method_id), None)
        values = {'id': method_id, 'label': label, 'color': color,
                  'successes': sum(t['successes'] for t in selection['tasks']),
                  'episodes': len(tasks) * 5, 'source_kind': 'historical',
                  'source_run': folder, 'source_sha256': selection['source_sha256']}
        if method is None:
            data['methods'].append(values)
        else:
            method.update(values)
    data.update(schema='robolab.leaderboard.v2', episodes_per_task=5,
                historical_baseline_selection='robolab-baselines.json')
    data['provenance']['historical_baseline_sources'] = {
        m['method']: m['source_sha256'] for m in audit['methods']}
    data['notes'] = [n.replace('across all three methods', 'across methods')
                     for n in data['notes'] if not n.startswith('Pi05-only uses')]
    for note in [
        'All three historical baselines use the same June 2026 evaluation cohort as the existing pi0.5 reference, selecting the first five episodes per task by run, episode, env_id, including failures.',
        'Task names and episode counts match; historical and Astra evaluations are not verified to share identical initial states, task versions or control settings.',
        'Only scores are added for Cosmos3-Nano-Policy and DreamZero; the existing 150-video gallery is unchanged.',
    ]:
        if note not in data['notes']:
            data['notes'].append(note)
    return data, audit


def write_report(report_root, data, audit):
    root = Path(report_root)
    for name, value in [('robolab.json', data), ('robolab-baselines.json', audit)]:
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    with (root / 'robolab-scores.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['task', 'task_label', 'episodes_per_method',
                         *[m['id'] + '_successes' for m in data['methods']]])
        for task in data['tasks']:
            writer.writerow([task['task'], task['label'], 5,
                             *[task['successes'][m['id']] for m in data['methods']]])
    with (root / 'robolab-leaderboard.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['scope', 'method', 'successes', 'episodes', 'success_rate_percent'])
        for task in [None, *data['tasks']]:
            for method in data['methods']:
                successes = task['successes'][method['id']] if task else method['successes']
                episodes = 5 if task else method['episodes']
                writer.writerow([task['task'] if task else 'overall', method['label'],
                                 successes, episodes, 100 * successes / episodes])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--report-root', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.report_root / 'robolab.json').read_text())
    data, audit = augment(report, args.baseline_root)
    write_report(args.report_root, data, audit)
    print(json.dumps({m['label']: f"{m['successes']}/{m['episodes']}" for m in data['methods']}))


if __name__ == '__main__':
    main()
