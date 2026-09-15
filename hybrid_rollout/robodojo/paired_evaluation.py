"""Select exactly one native-complete outcome per fixed case and method.

Historical v1 failures are preserved, never added to the 50-case denominator.
The eight explicitly frozen v1-success exemptions are labeled, not presented
as fresh v2 runs or an unbiased all-v2 comparison.
"""
from collections import Counter
from pathlib import Path

from . import campaign
from .case_ledger import build_ledger
from .evaluation import case_identity, read_panel
from .io import sha256, write_json


def select_panel(shared, panel, case_ids, method, exempt_v1=(), *, jobs=None,
                 context_version='v2', experiment_prefix=None):
    ledger = build_ledger(shared, panel, method=method)
    if len(case_ids) != 50 or len(set(case_ids)) != 50:
        raise ValueError('Require exactly 50 frozen distinct evaluation cases')
    if not set(exempt_v1) <= set(case_ids) or (method == 'gpt_only' and exempt_v1):
        raise ValueError('Invalid context exemptions')
    rows = {r['case_id']: r for r in ledger['cases']}
    if context_version == 'v3' and (exempt_v1 or not experiment_prefix or jobs is None):
        raise ValueError('v3 selection needs exact campaign, live platform evidence and no v1 exemptions')
    definitions = {c['case_id']: c for c in panel['cases']}
    chosen, missing, errors = [], [], list(ledger['errors'])
    for cid in case_ids:
        row = rows[cid]
        if context_version != 'v3' and row['state'] == 'duplicate_completed_review_required':
            errors.append(dict(case_id=cid, error='Unexpected duplicate completion'))
            continue
        candidates = [a for a in row['attempts'] if a['state'] == 'completed' and
            ((cid in exempt_v1 and a.get('context_version', 'v1') == 'v1' and a['native_success'] is True)
             or (cid not in exempt_v1 and a.get('context_version') == context_version))]
        if context_version == 'v3':
            # An archival/network exit may fail AFTER a native terminal result.
            # Do not discard that outcome or sample a replacement success.
            candidates = [a for a in row['attempts'] if a.get('context_version') == 'v3'
                and a.get('experiment', '').startswith(experiment_prefix+'_') and a.get('archive')
                and campaign.optional(Path(a['archive'])/'sim/evaluation_outcome.json').get('complete') is True]
        if len(candidates) != 1:
            missing.append(dict(case_id=cid, eligible_complete_count=len(candidates)))
            continue
        item = candidates[0]
        archive = Path(item['archive'])
        case = definitions[cid]
        identity = case_identity(panel, case)
        result = campaign.optional(archive/'controller/result.json')
        outcome = campaign.optional(archive/'sim/evaluation_outcome.json')
        run = campaign.optional(archive/'controller/run.json')
        worker = campaign.optional(archive/'controller/codex_workspace/worker.json')
        fingerprint = campaign.optional(archive/'sim/initial_observation_fingerprint.json')
        artifact = campaign.optional(archive/'artifact_manifest.json')
        if artifact.get('status') != 'verified':
            missing.append(dict(case_id=cid, reason='artifact_not_verified'))
            continue
        if method == 'gpt_only' and not all(artifact.get('checks', {}).get(k) is True for k in
                ('gpt_only_no_pi05_service', 'gpt_only_direct_tool_contract')):
            errors.append(dict(case_id=cid, error='GPT-only policy-exclusion evidence missing'))
            continue
        if context_version == 'v3' and method == 'gpt_only' and (
                run.get('action_space') != 'eef_only'
                or artifact.get('checks', {}).get('gpt_only_eef_action_contract') is not True):
            errors.append(dict(case_id=cid, error='Missing EEF-only evidence'))
            continue
        job_id = None
        if jobs is not None:
            matches = [j for j in jobs if j.get('display_name') == item['experiment']+'_r'+str(case['replica_id'])
                       and j.get('ownership', {}).get('user_name') == 'sujiayi']
            if len(matches) != 1:
                missing.append(dict(case_id=cid, reason='platform_identity_unresolved'))
                continue
            job = matches[0]
            verdict = campaign.assess(campaign.optional(item['batch_path']), archive,
                job['state'], identity)
            if verdict != 'completed':
                missing.append(dict(case_id=cid, reason=verdict))
                continue
            job_id = job['name']
        chosen.append(dict(case_id=cid, task=case['task'], variant=case['variant'],
            evaluation_case=identity, method=method, context_version=item['context_version'],
            reused_v1_success=cid in exempt_v1, native_success=outcome['native_success'],
            native_score=outcome.get('native_score'), control_steps=result['step_id'],
            archive=str(archive), attempt=item['attempt'], job_id=job_id,
            result_sha256=sha256(archive/'controller/result.json'),
            outcome_sha256=sha256(archive/'sim/evaluation_outcome.json'),
            artifact_manifest_sha256=sha256(archive/'artifact_manifest.json'),
            initial_observation_fields=fingerprint.get('fields'),
            initial_observation_fingerprint_sha256=sha256(archive/'sim/initial_observation_fingerprint.json'),
            shared_settings={k:run.get(k) for k in ('robot_profile', 'max_episode_steps',
                'control_dt', 'teacher_model', 'teacher_reasoning_effort')},
            image_max_edge=worker.get('codex_image_max_edge'),
            teacher_prompt_sha256=run.get('teacher_prompt_sha256')))
    return dict(method=method, case_count=50, selected_count=len(chosen),
        native_successes=sum(c['native_success'] for c in chosen),
        complete=len(chosen)==50 and not missing and not errors,
        platform_verified=jobs is not None, missing=missing, errors=errors, cases=chosen)


def comparison(hybrid, direct):
    if not all(p['complete'] and p['platform_verified'] for p in (hybrid,direct)):
        raise ValueError('Both full panels require verified native terminal outcomes')
    left={r['case_id']:r for r in hybrid['cases']}
    right={r['case_id']:r for r in direct['cases']}
    if left.keys()!=right.keys() or any(left[k]['evaluation_case']!=right[k]['evaluation_case'] for k in left):
        raise ValueError('Comparison cases/seeds/settings are not aligned')
    if any(left[k].get('shared_settings') != right[k].get('shared_settings') or
           left[k].get('image_max_edge') != right[k].get('image_max_edge') for k in left):
        raise ValueError('Comparison robot/horizon/model/preview settings changed')
    mismatched_initial = [k for k in left if left[k].get('initial_observation_fields') !=
                           right[k].get('initial_observation_fields')]
    per_task=[]
    def success(row):
        return row.get('evaluation_success', row['native_success']) is True
    for task in sorted({r['task'] for r in left.values()}):
        ids=[k for k,v in left.items() if v['task']==task]
        if len(ids)!=5:raise ValueError('Each task must have exactly five cases')
        per_task.append(dict(task=task, n=5, hybrid_successes=sum(success(left[k]) for k in ids),
            gpt_only_successes=sum(success(right[k]) for k in ids),
            gpt_only_idle_timeout_failures=sum(right[k].get('evaluation_failure_reason') ==
                'simulator_rpc_idle_timeout_900s' for k in ids)))
    paired=Counter((success(left[k]),success(right[k])) for k in left)
    v3 = all(r['context_version'] == 'v3' for r in [*left.values(), *right.values()])
    def mean_score(rows):
        values = [r.get('native_score') for r in rows]
        return sum(values)/len(values) if values and all(type(v) in (int, float) for v in values) else None
    for row in per_task:
        row['hybrid_mean_score'] = mean_score([r for r in left.values() if r['task'] == row['task']])
        row['gpt_only_mean_score'] = mean_score([r for r in right.values() if r['task'] == row['task']])
    return dict(schema='robodojo.paired50_comparison.v3' if v3 else 'robodojo.paired50_comparison.v1', cases=50, tasks=10,
        hybrid_mean_score=mean_score(left.values()), gpt_only_mean_score=mean_score(right.values()),
        direct_native_complete_count=direct.get('native_complete_count', len(direct['cases'])),
        direct_adjudicated_failure_count=direct.get('adjudicated_failure_count', 0),
        score_note='Native scores missing after adjudicated idle failures remain null; means requiring them are unavailable, not zero-filled.',
        hybrid_successes=hybrid['native_successes'],gpt_only_successes=direct['native_successes'],
        hybrid_success_rate=hybrid['native_successes']/50,gpt_only_success_rate=direct['native_successes']/50,
        hybrid_only_success=paired[True,False],gpt_only_success=paired[False,True],
        both_success=paired[True,True],both_failure=paired[False,False], per_task=per_task,
        reused_v1_success_count=sum(r['reused_v1_success'] for r in left.values()),
        initial_observation_bitwise_mismatch_cases=mismatched_initial,
        initial_observation_note='Same frozen layouts/seeds do not guarantee bitwise-identical physics; report recorded RGB/proprio/EEF fingerprints, never silently replace a mismatched case.',
        caveat=('Both panels use v3 context and the same fifty fixed cases. GPT-only uses EEF actions only; '
                'the hybrid policy may execute longer pi05 chunks. Thus decision count and token cost are not fixed equal. '
                'User-adjudicated simulator RPC idle timeouts count as evaluation failures, not native terminal outcomes; missing native scores are not imputed. '
                'Five cases per task provide limited precision.' if v3 else
                'Hybrid reuses explicitly selected v1 successes; this is not a fresh homogeneous v2 or unbiased all-v2 estimate. Five cases per task provide limited precision.'),
        hybrid=hybrid,direct=direct)
