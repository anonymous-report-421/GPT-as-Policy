import copy

import pytest

from . import campaign
from .evaluation import case_identity
from .input_guard_queue import release_verified_guards
from .io import write_json
from .test_campaign import dispatcher_fixture


def fixture(tmp_path,monkeypatch):
    plan,state,job,panel,sub,batch,archive=dispatcher_fixture(tmp_path,monkeypatch)
    slot=state['slots'][0];slot['status']='pause_unknown_failure';slot['job_id']=job['name']
    (archive/'sim').mkdir()
    batch.parent.mkdir(parents=True,exist_ok=True)
    case=plan['queue'][0]['case'];identity=case_identity(panel,case)
    write_json(archive/'sim/evaluation_outcome.json',dict(evaluation_case=identity,complete=False))
    write_json(archive/'controller/result.json',dict(evaluation_case=identity,complete=False))
    write_json(archive/'controller/failure.json',{'error':'InputError: invalid proposal_path\nFive rejected tool calls'})
    write_json(archive/'artifact_manifest.json',{'status':'verified'})
    write_json(batch,dict(state='failed',attempt=0,exit_code=1,episodes=[dict(archive=str(archive))],
        final_reset=dict(all_owned_processes_exited=True,ports_released=True)))
    return plan,state,job,sub,batch,archive


def test_known_guard_retries_once_then_defers_without_blocking_slot(tmp_path,monkeypatch):
    plan,state,job,sub,batch,archive=fixture(tmp_path,monkeypatch)
    old=copy.deepcopy(state['slots'][0]);raw=(archive/'controller/failure.json').read_bytes()
    release_verified_guards(plan,state,tmp_path,[job])
    assert state['slots'][0]['status']=='idle'
    assert state['pending_retries'][0]['attempt']==1
    assert state['input_guard_retry_counts']=={'0':1}
    state['pending_retries']=[];state['slots'][0].update(old,attempt=1)
    write_json(batch,dict(campaign.optional(batch),attempt=1))
    release_verified_guards(plan,state,tmp_path,[job])
    assert state['slots'][0]['status']=='idle' and not state['pending_retries']
    assert state['deferred_cases']['0']['original_slot']['attempt']==1
    assert (archive/'controller/failure.json').read_bytes()==raw


@pytest.mark.parametrize('reason',['live','native_complete','unclean','unknown','wrong_attempt','global_stop','slot_stop','other_owner'])
def test_unverified_or_out_of_scope_attempt_is_not_retried(tmp_path,monkeypatch,reason):
    plan,state,job,sub,batch,archive=fixture(tmp_path,monkeypatch)
    if reason=='live':job['state']='RUNNING'
    elif reason=='native_complete':
        write_json(archive/'sim/evaluation_outcome.json',dict(campaign.optional(archive/'sim/evaluation_outcome.json'),complete=True))
    elif reason=='unclean':write_json(batch,dict(campaign.optional(batch),final_reset={}))
    elif reason=='unknown':write_json(archive/'controller/failure.json',{'error':'HTTP 502'})
    elif reason=='wrong_attempt':state['slots'][0]['attempt']=3
    elif reason=='global_stop':write_json(tmp_path/'STOP',{})
    elif reason=='slot_stop':write_json(tmp_path/'STOP_SLOT_4',{})
    elif reason=='other_owner':job['ownership']['user_name']='someone_else'
    before=copy.deepcopy(state)
    release_verified_guards(plan,state,tmp_path,[job])
    assert state==before


def test_manual_retry_counts_toward_limit_even_if_another_slot_claimed_it(tmp_path,monkeypatch):
    plan,state,job,sub,batch,archive=fixture(tmp_path,monkeypatch)
    state['slots'].append(dict(id=12,status='idle',assigned=[0],history=[dict(index=0,action='user_authorized_input_guard_retry')]))
    release_verified_guards(plan,state,tmp_path,[job])
    assert state['slots'][0]['status']=='idle'
    assert '0' in state['deferred_cases'] and not state.get('pending_retries')


def zero_fixture(tmp_path, monkeypatch):
    from .io import sha256
    plan, state, job, sub, batch, archive = fixture(tmp_path, monkeypatch)
    result = dict(campaign.optional(archive/'controller/result.json'), episode_id='zeroepisode',
        reason='controller_error', terminated=False, truncated=False, video_frames=1,
        evaluation_method='gpt_only', context_version='v3', pi05_enabled=False, pi05_inference_calls=0,
        **{k:0 for k in ('step_id', 'decisions', 'student_steps', 'edited_steps',
                        'recovery_steps', 'predictions', 'gpt_joint_steps', 'gpt_eef_steps')})
    write_json(archive/'controller/result.json', result)
    write_json(archive/'controller/run.json', dict(evaluation_method='gpt_only',
        action_space='eef_only', context_version='v3', pi05_enabled=False, pi05_inference_calls=0))
    write_json(archive/'controller/history.json', [])
    write_json(archive/'sim/reset.json', dict(episode_id='zeroepisode', step_id=0))
    write_json(archive/'sim/summary.json', {k:result[k] for k in
        ('episode_id','step_id','reason','complete','terminated','truncated','video_frames')})
    write_json(archive/'sim/evaluation_outcome.json', dict(
        campaign.optional(archive/'sim/evaluation_outcome.json'), reason='controller_error',
        native_control_steps=0, valid_for_success_rate=False, native_success=None, native_score=None))
    first = archive/'sim/zeroepisode/observations/000000.npz'
    first.parent.mkdir(parents=True)
    first.write_bytes(b'initial-observation-fixture')
    (archive/'sim/sensors.mp4').write_bytes(b'one-frame-fixture')
    checks = {k:True for k in ('has_controller_result','has_sim_reset','has_sensor_video',
        'has_source_snapshot','observation_indices_contiguous','action_indices_contiguous',
        'observation_payload_keys_complete','video_frame_contract','gpt_only_no_pi05_service',
        'gpt_only_direct_tool_contract','paired_evaluation_identity_matches','frozen_layout_copy_matches')}
    checks['gpt_only_eef_action_contract'] = False
    paths = [first, archive/'sim/sensors.mp4'] + [archive/n for n in
        ('controller/run.json','controller/result.json','controller/history.json',
         'sim/reset.json','sim/summary.json','sim/evaluation_outcome.json')]
    write_json(archive/'artifact_manifest.json', dict(status='incomplete', checks=checks,
        episode_id='zeroepisode', step_id=0, raw_observation_count=1, raw_action_count=0,
        observation_payload_errors=[], all_immutable_artifacts=[dict(
            path=p.relative_to(archive).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in paths]))
    return plan, state, job, sub, batch, archive


def test_zero_step_guard_retries_without_promoting_or_modifying_incomplete_archive(tmp_path, monkeypatch):
    plan, state, job, sub, batch, archive = zero_fixture(tmp_path, monkeypatch)
    original = {p:p.read_bytes() for p in archive.rglob('*') if p.is_file()}
    release_verified_guards(plan, state, tmp_path, [job])
    assert state['slots'][0]['status'] == 'idle'
    assert state['pending_retries'][0]['attempt'] == 1
    assert state['input_guard_queue_events'][0]['archive_retry_evidence'] == 'zero_step_eef_guard_verified_no_execution'
    assert all(p.read_bytes() == data for p, data in original.items())


@pytest.mark.parametrize('condition', ['nonzero', 'bad_extra_check', 'missing_check', 'accepted_response',
    'raw_action', 'mutated_initial', 'missing_history', 'native_terminal', 'wrong_episode',
    'other_method', 'manual_stop', 'missing_reset', 'global_stop', 'slot_stop'])
def test_zero_step_exception_rejects_other_missing_evidence_or_execution(tmp_path, monkeypatch, condition):
    plan, state, job, sub, batch, archive = zero_fixture(tmp_path, monkeypatch)
    def update(name, **values):
        write_json(archive/name, dict(campaign.optional(archive/name), **values))
    if condition == 'nonzero': update('controller/result.json', step_id=1)
    if condition in ('bad_extra_check', 'missing_check'):
        m=campaign.optional(archive/'artifact_manifest.json')
        if condition == 'bad_extra_check': m['checks']['video_frame_contract']=False
        else: del m['checks']['gpt_only_no_pi05_service']
        write_json(archive/'artifact_manifest.json', m)
    if condition == 'accepted_response': write_json(archive/'controller/response_000.json', {'mode':'eef'})
    if condition == 'raw_action': write_json(archive/'sim/zeroepisode/action_000000.json', {})
    if condition == 'mutated_initial': (archive/'sim/zeroepisode/observations/000000.npz').write_bytes(b'changed')
    if condition == 'missing_history': (archive/'controller/history.json').unlink()
    if condition == 'native_terminal': update('sim/evaluation_outcome.json', complete=True)
    if condition == 'wrong_episode': update('sim/reset.json', episode_id='another')
    if condition == 'other_method': update('controller/run.json', evaluation_method='pi05_plus_gpt')
    if condition == 'manual_stop': job['state']=next(iter(campaign.MANUAL_STOP))
    if condition == 'missing_reset': write_json(batch, dict(campaign.optional(batch), final_reset={}))
    if condition == 'global_stop': write_json(tmp_path/'STOP', {})
    if condition == 'slot_stop': write_json(tmp_path/'STOP_SLOT_4', {})
    before=copy.deepcopy(state)
    release_verified_guards(plan, state, tmp_path, [job])
    assert state == before
