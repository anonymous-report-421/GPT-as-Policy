"""Bounded, opt-in recycling of terminal input-guard slots; no controller changes."""
import copy
import time
from pathlib import Path

from . import campaign
from .evaluation import case_identity, read_panel
from .io import sha256


def guard_archive_evidence(archive, result, outcome):
    """Accept either a verified trajectory or strictly proven zero execution.

    The EEF artifact check deliberately requires a nonempty accepted response
    list. Five invalid calls before the first action therefore leave that one
    check false. Such an interrupted attempt may be retried, never counted as
    a completed trajectory, and its original manifest must remain unchanged.
    """
    manifest = campaign.optional(archive/'artifact_manifest.json')
    if manifest.get('status') == 'verified':
        return 'verified_action_archive'
    checks = manifest.get('checks', {})
    required = {'has_controller_result', 'has_sim_reset', 'has_sensor_video',
        'has_source_snapshot', 'observation_indices_contiguous',
        'action_indices_contiguous', 'observation_payload_keys_complete',
        'video_frame_contract', 'gpt_only_no_pi05_service',
        'gpt_only_direct_tool_contract', 'gpt_only_eef_action_contract',
        'paired_evaluation_identity_matches', 'frozen_layout_copy_matches'}
    run = campaign.optional(archive/'controller/run.json')
    reset = campaign.optional(archive/'sim/reset.json')
    summary = campaign.optional(archive/'sim/summary.json')
    episode_id = result.get('episode_id')
    if (manifest.get('status') != 'incomplete' or not required <= checks.keys()
            or {k for k, v in checks.items() if v is not True} != {'gpt_only_eef_action_contract'}
            or checks['gpt_only_eef_action_contract'] is not False
            or manifest.get('raw_observation_count') != 1 or manifest.get('raw_action_count') != 0
            or manifest.get('observation_payload_errors') != []
            or not isinstance(episode_id, str) or not episode_id.isalnum()
            or any(x.get('episode_id') != episode_id for x in (manifest, reset, summary))
            or any(type(result.get(k)) is not int or result[k] != 0 for k in
                ('step_id', 'decisions', 'student_steps', 'edited_steps', 'recovery_steps',
                 'predictions', 'gpt_joint_steps', 'gpt_eef_steps'))
            or any(x.get('step_id') != 0 for x in (manifest, reset, summary))
            or any(x.get('complete') is not False or x.get('reason') != 'controller_error'
                   for x in (result, summary, outcome))
            or any(x.get(k) is not False for x in (result, summary) for k in ('terminated', 'truncated'))
            or any(x.get('video_frames') != 1 for x in (result, summary))
            or outcome.get('native_control_steps') != 0
            or outcome.get('valid_for_success_rate') is not False
            or outcome.get('native_success') is not None or outcome.get('native_score') is not None
            or run.get('action_space') != 'eef_only'
            or any(x.get('evaluation_method') != 'gpt_only' or x.get('pi05_enabled') is not False
                   or x.get('pi05_inference_calls') != 0 or x.get('context_version') != 'v3'
                   for x in (run, result))
            or campaign.optional(archive/'controller/history.json') != []):
        return None
    episode = archive/'sim'/episode_id
    first = episode/'observations/000000.npz'
    if (list((episode/'observations').glob('*.npz')) != [first]
            or list(episode.glob('action_*.json'))
            or any((archive/'controller').glob('response_*.json'))
            or any((archive/'controller').glob('execution_*.npz'))):
        return None
    inventory = {x['path']: x for x in manifest.get('all_immutable_artifacts', [])}
    paths = [first, archive/'sim/sensors.mp4'] + [archive/n for n in
        ('controller/run.json', 'controller/result.json', 'controller/history.json',
         'sim/reset.json', 'sim/summary.json', 'sim/evaluation_outcome.json')]
    for path in paths:
        record = inventory.get(path.relative_to(archive).as_posix(), {})
        if (path.is_symlink() or not path.is_file() or record.get('bytes') != path.stat().st_size
                or record.get('sha256') != sha256(path)):
            return None
    return 'zero_step_eef_guard_verified_no_execution'


def release_verified_guards(plan, state, directory, jobs):
    directory=Path(directory)
    if (directory/'STOP').exists():return
    candidates=[s for s in state['slots'] if s['status']=='pause_unknown_failure'
        and not (directory/f"STOP_SLOT_{s['id']}").exists()]
    if not candidates:return
    panel=read_panel(plan['eval_manifest'])
    jobs={j['name']:j for j in jobs}
    for slot in candidates:
        job=jobs.get(slot.get('job_id'),{})
        sub=campaign.optional(slot['submission']);batch=campaign.optional(sub['batch_result'])
        if len(batch.get('episodes',[]))!=1:continue
        episode=batch['episodes'][0];archive=Path(episode['archive'])
        result=campaign.optional(archive/'controller/result.json')
        outcome=campaign.optional(archive/'sim/evaluation_outcome.json')
        error=campaign.optional(archive/'controller/failure.json').get('error','')
        expected=case_identity(panel,plan['queue'][slot['index']]['case'])
        if (job.get('state') not in campaign.TERMINAL - campaign.MANUAL_STOP or batch.get('exit_code')==130
                or batch.get('state')!='failed'
                or str(batch.get('attempt'))!=str(slot['attempt'])
                or job.get('ownership',{}).get('user_name')!='sujiayi'
                or job.get('display_name')!=sub['environment']['ROLLOUT_EXPERIMENT_ID']+'_r'+sub['environment']['ROLLOUT_REPLICA_ID']
                or sub.get('evaluation_panel_sha256')!=panel['panel_sha256']
                or outcome.get('evaluation_case')!=expected or result.get('evaluation_case')!=expected
                or outcome.get('complete') is not False or result.get('complete') is not False
                or 'InputError' not in error or 'Five rejected tool calls' not in error
                or not all(batch.get('final_reset',{}).get(k) is True for k in ['ports_released','all_owned_processes_exited'])):
            continue  # Unknown/auth/quota/native outcomes must never be disguised as guard retries.
        evidence = guard_archive_evidence(archive, result, outcome)
        if evidence is None:
            continue
        index=slot['index'];key=str(index)
        if (index not in slot.get('assigned',[]) or key in state.get('deferred_cases',{})
                or any(r['index']==index for r in state.get('pending_retries',[]))
                or any(s is not slot and s.get('index')==index and s['status'] in
                       {'ready','submitted','submitting','running'} for s in state['slots'])):
            continue
        counts=state.setdefault('input_guard_retry_counts',{})
        # The initial explicitly requested C1 retry already consumes its one retry.
        prior_manual=any(h.get('index')==index and h.get('action')=='user_authorized_input_guard_retry'
            for s in state['slots'] for h in s.get('history',[]))
        count=max(int(counts.get(key,0)),int(prior_manual))
        event=dict(index=index,case_id=expected['case_id'],attempt=slot['attempt'],
            job_id=slot['job_id'],submission=slot['submission'],archive=str(archive),
            utc=campaign.utc(),raw_results_unchanged=True,native_complete=False,
            archive_retry_evidence=evidence)
        if count<1:
            counts[key]=count+1
            state.setdefault('pending_retries',[]).append(dict(index=index,attempt=slot['attempt']+1,
                retry_number=slot.get('retry_number',0),next_retry_at=time.time()+300,
                reason='bounded_input_guard_retry_after_verified_terminal_reset'))
            event.update(action='input_guard_retry_once',next_attempt=slot['attempt']+1)
        else:
            counts[key]=count
            state.setdefault('deferred_cases',{})[key]=dict(status='awaiting_operator_analysis',
                reason='Input guard repeated after one authorized retry; slot released without another paid attempt',
                original_slot=copy.deepcopy(slot),native_outcome=None)
            event.update(action='input_guard_deferred_after_one_retry')
        state.setdefault('input_guard_queue_events',[]).append(event)
        slot.setdefault('history',[]).append(event)
        slot.update(status='idle',submission=None,job_id=None,platform_state=None,
            next_retry_at=0,last_action=event['action'])
