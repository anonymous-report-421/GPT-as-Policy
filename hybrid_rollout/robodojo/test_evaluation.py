"""CPU-only paired fixture, native ordering, drift and recording checks."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from .evaluation import (TASKS, GENERALIZATION_TASKS, archive_path, canonical_sha256,
    case_identity, case_specs, freeze_panel, layout_files, read_panel, replica_cases,
    validate_panel, verify_assets)


def fake_native(tmp_path):
    source = tmp_path/'native'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture Test',
        '-c', 'user.email=fixture@example.invalid', 'commit', '-q', '--allow-empty', '-m', 'fixture'], check=True)
    (source/'env').mkdir()
    (source/'env/config.py').write_text('NATIVE_VERSION = 1\n')
    for case in case_specs():
        directory = source/'Assets/Eval_Layout/RoboDojo/arx_x5/0'
        directory.mkdir(parents=True, exist_ok=True)
        (directory/f'{case["runtime_task"]}_{case["layout_id"]}.json').write_text(json.dumps(case))
        if case['task'] == 'imitate_sorting_sequence':
            path = source/'Assets/Traj/RoboDojo/imitate_sorting_sequence/0'/f'{case["layout_id"]}.pkl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'opaque; never unpickle in controller')
    for index in range(4):
        path = source/'Assets/Traj/RoboDojo/make_kong'/f'{index}.pkl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'opaque native support trajectory')
    return source


def fake_panel_file(tmp_path):
    source = fake_native(tmp_path)
    panel = freeze_panel(source, 'test_panel')
    path = tmp_path/'evaluation.json'
    path.write_text(json.dumps(panel))
    return source, path, panel


def test_panel_exact_stratification_and_common_seeds():
    rows = list(case_specs())
    assert len(rows) == len({r['case_id'] for r in rows}) == 60
    assert {r['eval_seed'] for r in rows} == {0}
    for task in TASKS:
        selected = [r for r in rows if r['task'] == task]
        if task in GENERALIZATION_TASKS:
            assert [(r['variant'], r['layout_id']) for r in selected] == [
                (variant, i) for variant in ('standard', 'random') for i in range(3)]
        else:
            assert [r['layout_id'] for r in selected] == list(range(6))
        assert all(r['reset_seed'] == r['layout_id'] for r in selected)
        assert all(r['simulator_initial_seed'] == r['policy_rng_seed'] == 0 for r in selected)


def test_freeze_repeatable_and_layout_drift_rejected(tmp_path):
    source, path, panel = fake_panel_file(tmp_path)
    assert freeze_panel(source, 'test_panel') == panel
    verify_assets(read_panel(path, panel['panel_sha256']), source)
    case = panel['cases'][0]
    (source/case['layout']['path']).write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Asset changed'):
        verify_assets(panel, source)


def test_numerical_index_not_filename_suffix(tmp_path):
    directory = tmp_path/'Assets/Eval_Layout/RoboDojo/arx_x5/0'
    directory.mkdir(parents=True)
    for name in ('build_tower_10.json', 'build_tower_2.json', 'build_tower_random_0.json'):
        (directory/name).write_text('{}')
    assert [p.name for p in layout_files(tmp_path, 'build_tower', 0)] == ['build_tower_2.json', 'build_tower_10.json']


def test_wrong_hash_replica_and_mutated_seed_fail_closed(tmp_path):
    _, path, panel = fake_panel_file(tmp_path)
    with pytest.raises(ValueError, match='Unexpected evaluation panel'):
        read_panel(path, 'bad')
    with pytest.raises(ValueError, match='Task/replica'):
        replica_cases(panel, 'fold_clothes', 6)
    broken = copy.deepcopy(panel)
    broken['cases'][0]['layout_id'] = 9
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_panel(broken)
    broken['panel_sha256'] = canonical_sha256({k: v for k, v in broken.items() if k != 'panel_sha256'})
    with pytest.raises(ValueError, match='ordering/seed/variant'):
        validate_panel(broken)


def test_pair_key_independent_of_method_and_output_isolation(tmp_path):
    _, _, panel = fake_panel_file(tmp_path)
    paths = [archive_path(tmp_path, 'hybrid', c, c['replica_id'], 0) for c in panel['cases']]
    assert len(set(paths)) == 60
    case = panel['cases'][0]
    assert archive_path(tmp_path, 'pi05', case, 0, 0) != paths[0]
    assert case_identity(panel, case)['case_id'] == case['case_id']
    assert not any(k in case_identity(panel, case) for k in ('model', 'checkpoint', 'experiment_id'))


def test_checked_in_panel_matches_selection():
    panel = read_panel(Path(__file__).parent/'eval_panels/robodojo_panel60_v1.json')
    assert len(panel['cases']) == 60
