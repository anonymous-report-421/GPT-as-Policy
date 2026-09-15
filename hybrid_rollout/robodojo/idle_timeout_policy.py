"""Operations-only adjudication of the authorized 900s simulator idle failure.

Never mutate native outcomes, reconnect RPC, or equate arbitrary EOF/network
errors with idle timeout. Legacy RPC lacks an explicit timeout event: require
both independent timing signatures and the exact trusted call-stack/source.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .io import sha256, write_json

SCHEMA = 'robodojo.rpc_idle_timeout_failure.v1'
REASON = 'simulator_rpc_idle_timeout_900s'
SIDECAR = 'evaluation_adjudication.json'


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def policy_for(plan):
    root = Path(plan['shared_root'])/'campaigns'/plan['experiment_prefix']
    path = root/'rpc_idle_timeout_policy.json'
    if not path.exists():
        return None
    policy = read(path)
    if (policy.get('schema') != SCHEMA or policy.get('campaign') != plan['experiment_prefix']
            or policy.get('panel_sha256') != plan['panel_sha256']
            or policy.get('retry') is not False or policy.get('idle_seconds') != 900):
        raise ValueError('Unreviewed idle-timeout policy')
    manifest = read(Path(__file__).resolve().parents[2]/'source_manifest.json')
    pinned = manifest.get('idle_timeout_policy_sha256')
    if pinned and sha256(path) != pinned:
        raise ValueError('Frozen idle-timeout policy changed')
    return policy


def detect(archive, batch, platform_state, expected, policy):
    """Conservative legacy timeout signature; unrelated or uncertain EOF stays retryable."""
    archive = Path(archive)
    if platform_state not in ('FAILED', 'SUCCEEDED') or batch.get('exit_code') == 130:
        return None
    if not all(batch.get('final_reset', {}).get(k) is True for k in
               ('all_owned_processes_exited', 'ports_released')):
        return None
    paths = ['controller/failure.json', 'controller/history.json', 'controller/run.json',
             'sim/evaluation_outcome.json', 'sim/summary.json', 'logs/sim.log',
             'component_reset.json', 'controller/debug_video/manifest.json',
             'source_snapshot/hybrid_rollout/robodojo/robodojo_server/rpc.py']
    if not all((archive/p).is_file() for p in paths):
        return None
    if (str(archive) not in policy.get('explicit_archives', []) and
            (archive/'controller/failure.json').stat().st_mtime <
            datetime.fromisoformat(policy['applies_from_utc']).timestamp()):
        return None  # Do not retrospectively replace a later completed rerun.
    outcome, summary, failure, run = (read(archive/p) for p in
        ('sim/evaluation_outcome.json', 'sim/summary.json', 'controller/failure.json', 'controller/run.json'))
    if (outcome.get('evaluation_case') != expected or run.get('evaluation_case') != expected
            or outcome.get('complete') is not False or read(archive/'controller/result.json').get('complete')
            or outcome.get('status') != 'incomplete' or outcome.get('native_success') is not None
            or outcome.get('native_score') is not None
            or summary.get('reason') not in ('controller_disconnect', 'server_close')
            or summary.get('complete') is not False or run.get('context_version') != 'v3'):
        return None
    error = str(failure.get('error', ''))
    if not all(s in error for s in ('EOFError: RPC connection closed', 'record_codex_decision', 'robodojo_server/protocol.py')):
        return None
    rpc = archive/paths[-1]
    if sha256(rpc) != policy['rpc_source_sha256']:
        return None  # Only the reviewed server with fixed socket timeout.
    steps = outcome.get('native_control_steps')
    limit = outcome.get('native_step_limit')
    history = read(archive/'controller/history.json')
    if (type(steps) is not int or type(limit) is not int or not 0 < steps < limit
            or steps != summary.get('step_id') or steps != failure.get('step_id')
            or not isinstance(history, list) or not history or history[-1].get('end_tick') != steps):
        return None
    log = (archive/'logs/sim.log').read_text(errors='replace')
    records = []
    for match in re.finditer(r'\{"schema": "robodojo_rollout.codex_decision.v1"[^\n]*', log):
        try:
            row, _ = json.JSONDecoder().raw_decode(match.group())
        except ValueError:
            continue
        if row.get('event') == 'codex_decision':
            records.append((match.end(), row))
    if not records:
        return None
    pos, last = records[-1]
    tail = log[pos:]
    if ('Simulation App Shutting Down' not in tail or 'Traceback (most recent call last)' in tail
            or last.get('decision') != history[-1].get('decision')
            or last.get('step_id') != history[-1].get('start_tick')):
        return None
    # Summary is written immediately as the server closes; history immediately
    # after the last action ACK. Record these original timestamps before any copy.
    ack_time = (archive/'controller/history.json').stat().st_mtime
    close_time = (archive/'sim/summary.json').stat().st_mtime
    idle = close_time - ack_time
    stamps = re.findall(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)', tail)
    if not stamps:
        return None
    last_decision = datetime.fromisoformat(last['recorded_at']).timestamp()
    shutdown = datetime.fromisoformat(stamps[-1].replace('Z', '+00:00')).timestamp()
    gap = shutdown - last_decision
    if not (895 <= idle <= 915 and 900 <= gap <= 940 and abs(shutdown - close_time) <= 10):
        return None
    video = read(archive/'controller/debug_video/manifest.json')
    if video.get('status') != 'completed' or video.get('frames') != steps + 1:
        return None
    if not all(read(archive/'component_reset.json').get(k) is True for k in
               ('all_owned_processes_exited', 'ports_released')):
        return None
    return dict(schema=SCHEMA, reason=REASON, evaluation_failure=True, evaluation_success=False,
        retry=False, native_complete=False, native_success=None, native_score=None,
        evaluation_case=expected, control_steps=steps, native_step_limit=limit,
        video_frames=video['frames'], evidence_basis='reviewed_legacy_rpc_source_and_two_independent_timing_signatures',
        observed_idle_seconds=idle, decision_to_shutdown_seconds=gap,
        last_ack_file_mtime_utc=datetime.fromtimestamp(ack_time, timezone.utc).isoformat(),
        simulator_close_file_mtime_utc=datetime.fromtimestamp(close_time, timezone.utc).isoformat(),
        original_sha256={p: sha256(archive/p) for p in paths},
        video_label='Evaluation failure: simulator RPC idle timeout (900 s); no retry. Native score unavailable.')


def verified_adjudication(archive):
    archive = Path(archive)
    path = archive/SIDECAR
    if not path.exists():
        return None
    row = read(path)
    if (row.get('schema') != SCHEMA or row.get('reason') != REASON
            or row.get('evaluation_success') is not False or row.get('retry') is not False
            or row.get('native_complete') is not False or row.get('native_success') is not None
            or row.get('native_score') is not None or not row.get('original_sha256')):
        raise ValueError('Invalid timeout adjudication')
    for relative, digest in row['original_sha256'].items():
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts or sha256(archive/p) != digest:
            raise ValueError('Adjudicated evidence changed: '+relative)
    return row


def adjudicate(plan, batch, archive, job, expected):
    policy = policy_for(plan)
    if not policy:
        return None
    old = verified_adjudication(archive)
    if old:
        if old.get('evaluation_case') != expected or old.get('job_id') != job['name']:
            raise ValueError('Adjudication identity mismatch')
        return old
    row = detect(archive, batch, job['state'], expected, policy)
    if row is None:
        return None
    row.update(recorded_utc=datetime.now(timezone.utc).isoformat(), job_id=job['name'],
        platform_state=job['state'], archive=str(archive),
        policy_sha256=sha256(Path(plan['shared_root'])/'campaigns'/plan['experiment_prefix']/'rpc_idle_timeout_policy.json'))
    write_json(Path(archive)/SIDECAR, row)
    write_json(Path(archive)/'logs/evaluation_failure.json', row)
    return row


def include_adjudicated(plan, selection, jobs):
    """Add accepted evaluation failures without pretending they reached native termination."""
    policy = policy_for(plan)
    if not policy:
        return selection
    from .evaluation import archive_path, case_identity, read_panel
    root = Path(plan['shared_root'])/'campaigns'/plan['experiment_prefix']
    state = read(root/'state.json')
    panel = read_panel(plan['eval_manifest'], plan['panel_sha256'])
    chosen = list(selection['cases'])
    errors = list(selection['errors'])
    found = {r['case_id'] for r in chosen}
    for index, item in state.get('adjudicated_failures', {}).items():
        case = plan['queue'][int(index)]['case']
        expected = case_identity(panel, case)
        archive = Path(item['archive'])
        intended = archive_path(Path(plan['shared_root']), f"{plan['experiment_prefix']}_c{index}_a{item['attempt']}",
                                case, case['replica_id'], item['attempt'])
        if archive != intended or case['case_id'] in found:
            errors.append(dict(case_id=case['case_id'], error='Conflicting idle-timeout adjudication'))
            continue
        row = verified_adjudication(archive)
        matches = [j for j in jobs if j.get('name') == item['job_id']
                   and j.get('ownership', {}).get('user_name') == 'sujiayi']
        if (not row or row['evaluation_case'] != expected or len(matches) != 1
                or matches[0]['state'] not in ('SUCCEEDED', 'FAILED')
                or row['policy_sha256'] != sha256(root/'rpc_idle_timeout_policy.json')):
            continue
        run = read(archive/'controller/run.json')
        worker = read(archive/'controller/codex_workspace/worker.json')
        fingerprint = read(archive/'sim/initial_observation_fingerprint.json')
        chosen.append(dict(case_id=case['case_id'], task=case['task'], variant=case['variant'],
            evaluation_case=expected, method=plan['evaluation_method'], context_version='v3',
            reused_v1_success=False, native_success=None, evaluation_success=False,
            native_complete=False, native_score=None, evaluation_failure_reason=REASON,
            control_steps=row['control_steps'], archive=str(archive), attempt=item['attempt'],
            job_id=item['job_id'], adjudication_sha256=sha256(archive/SIDECAR),
            outcome_sha256=sha256(archive/'sim/evaluation_outcome.json'),
            artifact_manifest_sha256=sha256(archive/'artifact_manifest.json'),
            initial_observation_fields=fingerprint.get('fields'),
            initial_observation_fingerprint_sha256=sha256(archive/'sim/initial_observation_fingerprint.json'),
            shared_settings={k:run.get(k) for k in ('robot_profile', 'max_episode_steps',
                'control_dt', 'teacher_model', 'teacher_reasoning_effort')},
            image_max_edge=worker.get('codex_image_max_edge'), teacher_prompt_sha256=run.get('teacher_prompt_sha256')))
        found.add(case['case_id'])
    missing = [r for r in selection['missing'] if r['case_id'] not in found]
    return dict(selection, cases=chosen, selected_count=len(chosen), missing=missing, errors=errors,
        native_complete_count=selection['selected_count'],
        adjudicated_failure_count=len(chosen)-selection['selected_count'],
        complete=len(chosen) == 50 and not missing and not errors,
        outcome_rule='Native outcomes plus user-authorized 900s simulator idle failures; native score stays null for idle failures.')
