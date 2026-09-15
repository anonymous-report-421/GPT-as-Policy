"""Add audited RoboLab rows and its paper reference to the public report snapshot.

Run after refreshing the upstream public article. Raw episodes and selection
manifests are read-only; this step changes presentation names and derived rows.
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {
    'pure_astra': ('gpt', 'GPT 6 Astra（direct）'),
    'astra_pi05': ('mix', 'π₀.₅ + GPT 6 Astra'),
    'pi05_only': ('Pi-05', 'π₀.₅'),
    'cosmos_nano_policy': ('Cosmos3-Nano-Policy', 'Cosmos3-Nano-Policy'),
    'dreamzero': ('DreamZero', 'DreamZero'),
}
OLD_LABELS = {'Pure Astra': NAMES['pure_astra'][1],
              'Astra + π0.5': NAMES['astra_pi05'][1],
              'π0.5 only': NAMES['pi05_only'][1]}
REFERENCE = {
    'id': 'robolab', 'label': 'RoboLab',
    'title': 'RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies',
    'author': 'Jenai Xuning Yang, Rishit Dagli, Alex Zook, Hugo Hadfield, Ankit Goyal, Stan Birchfield, Fabio Ramos, Jonathan Tremblay',
    'type': '2026. arXiv:2604.09860 / Robotics: Science and Systems (RSS)',
    'url': 'https://arxiv.org/abs/2604.09860',
    'summary': 'RoboLab 以高保真仿真评测通用机器人策略的视觉、程序与关系理解；论文评测采用 DROID 形态的单臂 Franka Panda。本文另选 10 个任务，每方法每任务 5 个最终评测槽位，不是 RoboLab-120 全榜复现。',
}

def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def revise(root, snapshot_path):
    data = json.loads(snapshot_path.read_text())
    results = json.loads((root/'robolab.json').read_text())
    assert results['episodes_per_task'] == 5 and len(results['tasks']) == 10
    assert {m['id'] for m in results['methods']} == NAMES.keys()
    for method in results['methods']:
        assert method['episodes'] == 50
        assert method['successes'] == sum(t['successes'][method['id']] for t in results['tasks'])
        assert all(0 <= t['successes'][method['id']] <= 5 for t in results['tasks'])
        method['label'] = NAMES[method['id']][1]
    models = [{**m, 'name': NAMES[m['id']][0], 'sr': 100*m['successes']/m['episodes']} for m in results['methods']]
    source = {
        'label': 'RoboLab · selected final slots and first-five historical baselines',
        'files': ['robolab.json', 'robolab-scores.csv', 'robolab-baselines.json', 'provenance.json'],
        'selection': '10 tasks × 5 slots per policy; includes failures and authorized retries',
        'tables': [{'name': REFERENCE['title'], 'href': REFERENCE['url']}],
    }
    data['queries']['robolab_models'] = {'label': 'RoboLab policies', 'rows': models, 'source': source}
    data['queries']['robolab_tasks'] = {'label': 'RoboLab task results', 'rows': results['tasks'], 'source': source}
    data['reportData']['references'] = [REFERENCE if r['id']=='robolab' else r for r in data['reportData']['references']]
    data['reportData']['robolab'] = results
    data['reportData']['robolab_status'] = 'reported_supplementary'
    dump(snapshot_path, data)
    dump(root/'robolab.json', results)
    leaderboard = root/'robolab-leaderboard.csv'
    with leaderboard.open(newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    for row in rows:
        row['method'] = OLD_LABELS.get(row['method'], row['method'])
    with leaderboard.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({m['label']: m['sr'] for m in models}, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    revise(args.report_root, args.snapshot)

if __name__ == '__main__':
    main()
