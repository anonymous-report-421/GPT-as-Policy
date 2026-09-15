"""Public structured decisions only; no model reasoning stream or physics calls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from ..io import write_json


def decision_event(*, episode_id, step_id, decision, prediction_id, response,
                   origin='robodojo_server'):
    mode = response['mode']
    return dict(schema='robodojo_rollout.codex_decision.v1', event='codex_decision',
        recorded_at=datetime.now(timezone.utc).isoformat(), origin=origin,
        status='decision_received_not_execution_ack', episode_id=episode_id,
        step_id=step_id, decision=decision, prediction_id=prediction_id,
        request_id=response['request_id'], mode=mode, reason=response['reason'],
        requested_steps=response['steps'],
        evaluation_method=response.get('evaluation_method', 'pi05_plus_gpt'),
        codex_override=mode in ('edit', 'eef') and response.get('evaluation_method') != 'gpt_only',
        prefix_only=mode == 'student' and response['steps'] < 50,
        numeric_action_changed=None, response=response)


def emit(event):
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def record_decision(session, args):
    """Persist before acknowledging receipt; never imply action execution."""
    session._check_identity(args['episode_id'], args['step_id'])
    decision = args['decision']
    if type(decision) is not int or decision != getattr(session, 'codex_decision_count', 0):
        raise ValueError('Expected next Codex decision index; no duplicate recording')
    response = args['response']
    if not isinstance(response, dict) or response.get('mode') not in ('student', 'edit', 'eef', 'stop', 'joint'):
        raise ValueError('Invalid Codex response')
    if not isinstance(response.get('reason'), str) or not response['reason'].strip():
        raise ValueError('Codex reason is required')
    event = decision_event(episode_id=session.episode_id, step_id=session.step_id,
        decision=decision, prediction_id=args['prediction_id'], response=response)
    path = session.episode_dir/'codex_decisions'/f'{decision:03d}.json'
    if path.exists():
        raise ValueError('Refusing to overwrite an existing decision')
    path.parent.mkdir(exist_ok=True)
    write_json(path, event)
    session.codex_decision_count = decision + 1
    emit(dict(event, record_path=str(path)))
    return dict(episode_id=session.episode_id, step_id=session.step_id, physical_steps=0,
                decision=decision, record_path=str(path), status=event['status'])


def mirror_available(controller, seen):
    """Compatibility for immutable running snapshots; explicitly NOT server receipt."""
    for path in sorted(controller.glob('response_*.json')):
        if path.name in seen:
            continue
        decision = int(path.stem.split('_')[1])
        request = json.loads((controller/f'request_{decision:03d}.json').read_text())
        response = json.loads(path.read_text())
        event = decision_event(episode_id=request['episode_id'], step_id=request['step_id'],
            decision=decision, prediction_id=request['prediction_id'], response=response,
            origin='controller_record_mirror')
        event['status'] = 'controller_response_observed_not_server_receipt_or_execution_ack'
        emit(dict(event, response_path=str(path)))
        seen.add(path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not (root/'controller').is_dir():
        parser.error('run-root must contain an existing controller directory')
    seen = set()
    while True:
        mirror_available(root/'controller', seen)
        if not args.follow or (root/'job_exit_status.txt').exists():
            mirror_available(root/'controller', seen)
            break
        time.sleep(2)


if __name__ == '__main__':
    main()
