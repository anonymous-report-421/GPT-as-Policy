"""Freeze and verify model-independent, paired RoboDojo evaluation fixtures.

EvalEnv's eval_seed selects a published layout group; reset(seed=[layout_id])
selects its numerically sorted entry. Neither is a free-running episode RNG.
This module is CPU-only and never imports the simulator or loads trajectory PKLs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .io import require, sha256

TASKS = (
    'organize_table', 'classify_objects_by_language', 'imitate_sorting_sequence',
    'arrange_largest_number', 'pack_objects_into_box', 'classify_objects',
    'build_tower', 'make_kong', 'fold_clothes', 'put_bottles_into_dustbin',
)
GENERALIZATION_TASKS = frozenset({
    'arrange_largest_number', 'pack_objects_into_box', 'fold_clothes',
})
SCHEMA = 'hybrid_rollout.robodojo.evaluation_panel.v1'


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def layout_files(source, runtime_task, eval_seed):
    """Same full-match + numerical ordering as native SeedManager.init_eval."""
    directory = Path(source)/'Assets/Eval_Layout/RoboDojo/arx_x5'/str(eval_seed)
    pattern = re.compile(rf'{re.escape(runtime_task)}_\d+\.json')
    return sorted((p for p in directory.iterdir() if pattern.fullmatch(p.name)),
                  key=lambda p: int(p.stem.rsplit('_', 1)[-1]))


def case_specs(eval_seed=0):
    require(type(eval_seed) is int and eval_seed in (0, 1, 2), 'Unknown published eval_seed group')
    for replica, task in enumerate(TASKS):
        for index in range(6):
            random = task in GENERALIZATION_TASKS and index >= 3
            layout_id = index % 3 if task in GENERALIZATION_TASKS else index
            variant = 'random' if random else 'standard'
            yield dict(case_id=f'{task}__{variant}__g{eval_seed}__l{layout_id}',
                       task=task, runtime_task=task + ('_random' if random else ''),
                       variant=variant, replica_id=replica, rollout_index=index,
                       eval_seed=eval_seed, layout_id=layout_id, reset_seed=layout_id,
                       simulator_initial_seed=0, policy_rng_seed=0)


def file_record(source, path):
    return dict(path=path.relative_to(source).as_posix(), sha256=sha256(path))


def freeze_panel(source, panel_id, eval_seed=0):
    source = Path(source).resolve()
    require(re.fullmatch(r'[A-Za-z0-9_-]+', panel_id), 'Invalid panel_id')
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    # Include local changes to environment/task/config code, not just a git HEAD.
    files = sorted({p for folder in ('env', 'env_cfg', 'task/RoboDojo', 'utils', 'src/eval_client')
                    for p in (source/folder).rglob('*') if p.suffix in ('.py', '.yaml', '.yml')})
    native_files = [file_record(source, p) for p in files if p.is_file()]
    require(native_files, 'Missing native source files')
    cases = []
    for spec in case_specs(eval_seed):
        candidates = layout_files(source, spec['runtime_task'], eval_seed)
        require(spec['layout_id'] < len(candidates), f'Missing layout for {spec["case_id"]}')
        layout = candidates[spec['layout_id']]
        trajectories = []
        if spec['task'] == 'imitate_sorting_sequence':
            trajectories = [source/'Assets/Traj/RoboDojo/imitate_sorting_sequence'/
                            str(eval_seed)/f'{spec["layout_id"]}.pkl']
        elif spec['task'] == 'make_kong':
            trajectories = [source/'Assets/Traj/RoboDojo/make_kong'/f'{i}.pkl' for i in range(4)]
        cases.append(dict(spec, layout=file_record(source, layout),
                          native_support_trajectories=[file_record(source, p) for p in trajectories]))
    panel = dict(schema=SCHEMA, panel_id=panel_id, scope='60_episode_official_layout_subset',
                 selection='Group fixed before evaluation; native numerical layout order; no outcome selection',
                 eval_seed=eval_seed, native_source_commit=head, native_source_files=native_files,
                 seed_semantics=dict(eval_seed='published layout group', layout_id='native sorted layout index',
                     reset_seed='layout_id, passed to native reset', simulator_initial_seed='native startup seed',
                     policy_rng_seed='OpenPI JAX initial key; fresh process, no warmup inference'),
                 pairing='Reuse this manifest unchanged across methods; compare observed initial fingerprints too',
                 invalid_layout_policy='Record and exclude native-invalid cases; no automatic replacement or retry',
                 cases=cases)
    panel['panel_sha256'] = canonical_sha256(panel)
    validate_panel(panel)
    return panel


def validate_panel(panel):
    require(panel.get('schema') == SCHEMA, 'Unknown evaluation manifest schema')
    require(panel.get('panel_sha256') == canonical_sha256(
        {k: v for k, v in panel.items() if k != 'panel_sha256'}), 'Evaluation manifest hash mismatch')
    expected = list(case_specs(panel['eval_seed']))
    require(len(panel.get('cases', [])) == 60, 'Expected exactly 60 frozen cases')
    for actual, spec in zip(panel['cases'], expected):
        require(all(actual.get(k) == v for k, v in spec.items()), 'Evaluation case ordering/seed/variant changed')
        for record in [actual['layout'], *actual['native_support_trajectories']]:
            require(not Path(record['path']).is_absolute() and '..' not in Path(record['path']).parts,
                    'Asset path must remain below the native source')
    return panel


def read_panel(path, expected_sha=None):
    panel = validate_panel(json.loads(Path(path).read_text()))
    require(expected_sha is None or panel['panel_sha256'] == expected_sha, 'Unexpected evaluation panel identity')
    return panel


def verify_assets(panel, source, cases=None):
    """Fail before paying for inference if an input or its native index changed."""
    source = Path(source).resolve()
    validate_panel(panel)
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == panel['native_source_commit'], 'RoboDojo native commit changed')
    for record in panel['native_source_files']:
        require(sha256(source/record['path']) == record['sha256'], f'Native source changed: {record["path"]}')
    for case in panel['cases'] if cases is None else cases:
        resolved = layout_files(source, case['runtime_task'], case['eval_seed'])[case['layout_id']]
        require(resolved == source/case['layout']['path'], f'Native layout ordering changed: {case["case_id"]}')
        for record in [case['layout'], *case['native_support_trajectories']]:
            require(sha256(source/record['path']) == record['sha256'], f'Asset changed: {record["path"]}')


def replica_cases(panel, task, replica):
    rows = [c for c in panel['cases'] if c['replica_id'] == int(replica)]
    require(len(rows) == 6 and all(c['task'] == task for c in rows), 'Task/replica does not match frozen panel')
    return rows


def case_identity(panel, case):
    return dict(panel_id=panel['panel_id'], panel_sha256=panel['panel_sha256'],
                **{k: case[k] for k in ('case_id', 'task', 'runtime_task', 'variant', 'eval_seed',
                    'layout_id', 'reset_seed', 'simulator_initial_seed', 'policy_rng_seed')},
                layout_sha256=case['layout']['sha256'])


def selected_cases(panel, task, replica, count=1, case_id=None):
    rows = replica_cases(panel, task, replica)
    if case_id is not None:
        require(count == 1, 'An explicit case_id requires exactly one rollout')
        selected = [c for c in rows if c['case_id'] == case_id]
        require(len(selected) == 1, 'case_id does not match the frozen task/replica')
        return selected
    require(1 <= count <= 6, 'Rollout count must be in 1..6')
    return rows[:count]


def archive_path(shared, experiment, case, replica, attempt):
    return (Path(shared)/'results'/experiment/case['task']/case['variant']/f'eval_seed_{case["eval_seed"]}'/
            f'layout_{case["layout_id"]}'/f'replica_{replica}'/f'attempt_{attempt}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--panel-id', default='robodojo_panel60_v1')
    p.add_argument('--eval-seed', type=int, default=0)
    args = p.parse_args()
    if args.freeze:
        panel = freeze_panel(args.source, args.panel_id, args.eval_seed)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open('x') as stream:  # Never replace a paired evaluation panel.
            json.dump(panel, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    panel = read_panel(args.manifest)
    verify_assets(panel, args.source)
    print(json.dumps(dict(manifest=str(args.manifest), panel_sha256=panel['panel_sha256'],
                          cases=len(panel['cases']), assets_verified=True)))


if __name__ == '__main__':
    main()
