"""Independent evidence audit for user-authorized simulator idle failures.

No fake result.json or relaxed native-complete gate: this is a separate audit
class. It verifies all recorded payloads plus the missing-result substitutes.
"""
import json
import argparse
from pathlib import Path

from .campaign import optional, utc
from .idle_timeout_policy import verified_adjudication, REASON, SIDECAR, policy_for
from .io import require, sha256, write_json


def adjudicated(row, plan=None):
    root = Path(row['archive']).resolve()
    evidence = verified_adjudication(root)
    require(evidence is not None, 'Missing authorized idle-failure adjudication')
    require(row.get('evaluation_failure_reason') == REASON
        and row.get('native_complete') is False and row.get('native_success') is None
        and row.get('native_score') is None and row.get('evaluation_success') is False,
        'Not the selected user-authorized idle failure')
    require(sha256(root/SIDECAR) == row.get('adjudication_sha256'), 'Adjudication changed')
    require(evidence['evaluation_case'] == row['evaluation_case']
        and evidence['control_steps'] == row['control_steps']
        and evidence['job_id'] == row['job_id'] and evidence['archive'] == str(root),
        'Adjudication selection identity changed')
    if plan is not None:
        require(policy_for(plan) is not None, 'No active authorization for idle failure')
        path=Path(plan['shared_root'])/'campaigns'/plan['experiment_prefix']/'rpc_idle_timeout_policy.json'
        require(evidence['policy_sha256'] == sha256(path), 'Idle-failure policy changed')
        require(row['case_id'] in plan['primary_case_ids'], 'Adjudicated case outside frozen plan')
    require(sha256(root/'artifact_manifest.json') == row['artifact_manifest_sha256'], 'Artifact manifest changed')
    return evidence


def audit_idle(row, plan=None):
    from .archive_audit import audit_saved_payload
    from .evaluation import case_identity, read_panel
    from .robodojo_server.debug_recorder import DecisionTimeline
    root=Path(row['archive']).resolve()
    evidence=adjudicated(row, plan)
    manifest=optional(root/'artifact_manifest.json')
    require(manifest.get('status') == 'incomplete', 'Original incomplete status must remain truthful')
    allowed_false={'has_controller_result','gpt_only_no_pi05_service',
        'gpt_only_eef_action_contract','paired_evaluation_identity_matches'}
    checks=manifest.get('checks',{})
    require({k for k,v in checks.items() if v is not True} == allowed_false
        and all(checks[k] is False for k in allowed_false), 'Unexpected incomplete artifact checks')
    require(not (root/'controller/result.json').exists(), 'Do not synthesize a missing controller result')
    inventory=manifest['all_immutable_artifacts']
    paths=[r['path'] for r in inventory]
    require(len(paths)==len(set(paths)), 'Duplicate immutable inventory paths')
    for item in inventory:
        p=Path(item['path']);path=(root/p).resolve()
        require(not p.is_absolute() and path.is_relative_to(root), 'Inventory escapes archive')
        require(sha256(path)==item['sha256'] and path.stat().st_size==item['bytes'],
            'Immutable artifact changed: '+item['path'])
    selected=optional(root/'evaluation_case.json')
    panel=read_panel(root/'evaluation_manifest.json')
    identity=case_identity(panel,selected['case'])
    run=optional(root/'controller/run.json');reset=optional(root/'sim/reset.json')
    failure=optional(root/'controller/failure.json');outcome=optional(root/'sim/evaluation_outcome.json')
    fingerprint=optional(root/'sim/initial_observation_fingerprint.json')
    require(selected['case'] in panel['cases'] and selected['identity']==identity
        and all(x==identity for x in [row['evaluation_case'],run['evaluation_case'],
            reset['metadata']['evaluation_case'],outcome['evaluation_case'],fingerprint['evaluation_case']]),
        'Frozen case/seed identities differ')
    require(run.get('evaluation_method')=='gpt_only' and run.get('context_version')=='v3'
        and run.get('pi05_enabled') is False and run.get('pi05_inference_calls')==0
        and run.get('action_space')=='eef_only', 'Unexpected method for this idle-failure audit')
    counts=failure['counters'];step=evidence['control_steps']
    require(failure.get('completed') is False and failure['step_id']==step
        and counts.get('gpt_eef_steps')==step and all(counts.get(k)==0 for k in
        ('student_steps','edited_steps','recovery_steps','predictions','gpt_joint_steps')),
        'Failure counters do not establish GPT-only execution')
    require(not any((root/p).exists() for p in ('policy.pid','policy_identity.json','logs/pi05.log'))
        and not list((root/'controller').glob('proposal_*.npz')), 'Unexpected pi05 artifacts')
    responses=[optional(p) for p in (root/'controller').glob('response_*.json')]
    require(bool(responses) and all(r.get('mode')=='eef' and r.get('evaluation_method')=='gpt_only'
        and type(r.get('steps')) is int and 1<=r['steps']<=5 for r in responses), 'Invalid EEF-only contract')
    timeline=DecisionTimeline(root/'controller')
    require(timeline.adjudication is not None and timeline.segments
        and timeline.segments[0]['start_tick']==0 and timeline.segments[-1]['end_tick']==step
        and sum(r['executed_steps'] for r in timeline.segments)==step, 'Missing executed action segment')
    payload=audit_saved_payload(root,step,manifest)
    require(payload['observations']==step+1==manifest['raw_observation_count']
        and payload['actions']==step==manifest['raw_action_count'], 'Raw count disagreement')
    return dict(archive=str(root),checked_utc=utc(),passed=True,**payload,
        context_version='v3',evaluation_method='gpt_only',native_complete=False,
        native_success=None,native_score=None,evaluation_success=False,
        evaluation_failure_reason=REASON,adjudication_sha256=row['adjudication_sha256'],
        artifact_manifest_sha256=sha256(root/'artifact_manifest.json'),
        immutable_files_verified=len(inventory),
        checks='Independent authorized idle-failure audit: every immutable file SHA/size; every NPZ member decoded/CRC; action JSON; actual video frame decode count; fixed case/seed; no-pi05 EEF-only counters and all execution segments. Native outcome remains incomplete.',
        original_data_modified=False,model_calls=0,simulation_runs=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--approved-sha256',required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(sha256(args.plan)==args.approved_sha256,'Plan changed')
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.archive.resolve()),
        'Use a new audit report outside the archive')
    plan=optional(args.plan);progress=optional(args.plan.parent/'progress.json')
    rows=[r for k in ('hybrid','direct') for r in progress.get(k,{}).get('cases',[])
          if Path(r['archive']).resolve()==args.archive.resolve()]
    require(len(rows)==1,'Require exactly one selected adjudicated archive')
    report=audit_idle(rows[0],plan)
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='videos'}))
