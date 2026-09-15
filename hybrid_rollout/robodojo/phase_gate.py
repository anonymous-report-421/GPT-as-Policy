"""Fail closed before GPT-only creation until all 50 mixed outcomes are verified."""
from pathlib import Path

from .evaluation import case_identity
from .io import sha256


def verify_certificate(path, panel, case_id=None):
    from .campaign import optional
    if not path or not Path(path).is_file():
        raise ValueError('GPT-only requires the completed hybrid50 certificate')
    document = optional(path)
    selection = document.get('selection', {})
    rows = selection.get('cases', [])
    v3 = document.get('schema') == 'robodojo.hybrid50_completion.v3'
    if (document.get('schema') not in ('robodojo.hybrid50_completion.v1', 'robodojo.hybrid50_completion.v3')
            or document.get('panel_sha256') != panel['panel_sha256']
            or selection.get('method') != 'pi05_plus_gpt'
            or selection.get('complete') is not True
            or selection.get('platform_verified') is not True
            or len(rows) != 50 or len({r['case_id'] for r in rows}) != 50):
        raise ValueError('Incomplete or incompatible hybrid50 certificate')
    definitions = {c['case_id']: c for c in panel['cases']}
    if case_id and case_id not in {r['case_id'] for r in rows}:
        raise ValueError('GPT-only case is not paired with the hybrid panel')
    for row in rows:
        archive = Path(row['archive'])
        for relative, field in (('controller/result.json', 'result_sha256'),
                ('sim/evaluation_outcome.json', 'outcome_sha256'),
                ('artifact_manifest.json', 'artifact_manifest_sha256')):
            if sha256(archive/relative) != row[field]:
                raise ValueError('Frozen hybrid outcome changed')
        result = optional(archive/'controller/result.json')
        outcome = optional(archive/'sim/evaluation_outcome.json')
        artifact = optional(archive/'artifact_manifest.json')
        expected = case_identity(panel, definitions[row['case_id']])
        if (row.get('evaluation_case') != expected
                or result.get('evaluation_case') != expected or outcome.get('evaluation_case') != expected
                or result.get('complete') is not True or outcome.get('complete') is not True
                or outcome.get('valid_for_success_rate') is not True
                or type(outcome.get('native_success')) is not bool
                or outcome['native_success'] is not row.get('native_success')
                or artifact.get('status') != 'verified'):
            raise ValueError('Invalid native hybrid outcome or case/seed mismatch')
        if v3:
            run = optional(archive/'controller/run.json')
            if (row.get('context_version') != 'v3' or row.get('reused_v1_success')
                    or run.get('context_version') != 'v3'
                    or run.get('evaluation_method', 'pi05_plus_gpt') != 'pi05_plus_gpt'):
                raise ValueError('v3 comparison requires fifty native v3 hybrid cases')
        elif row.get('reused_v1_success'):
            if row.get('context_version') != 'v1' or outcome['native_success'] is not True:
                raise ValueError('Only a native v1 success may be exempted')
        elif row.get('context_version') != 'v2':
            raise ValueError('Non-exempt hybrid cases require v2')
    return document
