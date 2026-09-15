import copy

import pytest

from . import adjudicated_archive_audit as subject
from .idle_timeout_policy import adjudicate, REASON
from .io import write_json, sha256


def fixture(tmp_path):
    from .test_idle_timeout_policy import fixture as idle_fixture
    plan,root,identity,policy,batch,job=idle_fixture(tmp_path)
    adjudicate(plan,batch,root,job,identity)
    write_json(root/'artifact_manifest.json',dict(status='incomplete'))
    return dict(archive=str(root),evaluation_case=identity,control_steps=257,job_id=job['name'],
        evaluation_failure_reason=REASON,native_complete=False,native_success=None,native_score=None,
        evaluation_success=False,adjudication_sha256=sha256(root/'evaluation_adjudication.json'),
        artifact_manifest_sha256=sha256(root/'artifact_manifest.json'))


def test_separate_adjudication_audit_does_not_convert_native_outcomes(tmp_path):
    row=fixture(tmp_path);before=copy.deepcopy(row)
    result=subject.adjudicated(row)
    assert result['native_complete'] is False and result['native_score'] is None
    assert row==before
    from .archive_audit import audit
    with pytest.raises(ValueError,match='Only native-complete'):audit(row['archive'])


@pytest.mark.parametrize('field,value',[('native_complete',True),('native_success',False),
    ('native_score',0),('evaluation_success',True),('evaluation_failure_reason','other'),
    ('adjudication_sha256','bad'),('control_steps',1100),('job_id','another'),
    ('evaluation_case',{'case_id':'other'}),('artifact_manifest_sha256','bad')])
def test_selected_adjudication_must_match_all_raw_identity_fields(tmp_path,field,value):
    row=fixture(tmp_path);row[field]=value
    with pytest.raises(ValueError):subject.adjudicated(row)


def test_final_export_requires_separate_idle_deep_and_pair_proof():
    from .test_final_debug_export import full_fixture
    from .final_debug_export import verify_full_audits
    rows,deep,pairs=full_fixture();row=rows[-1]
    row.update(evaluation_failure_reason=REASON,adjudication_sha256='adjudication')
    with pytest.raises(ValueError,match='separate deep audit'):verify_full_audits(rows,deep,pairs)
    deep['rows'][row['method']+':'+row['case_id']].update(native_complete=False,
        adjudication_sha256='adjudication',native_success=None,evaluation_success=False)
    with pytest.raises(ValueError,match='Pair audit'):verify_full_audits(rows,deep,pairs)
    pairs['rows'][-1].update(gpt_only_native_complete=False,gpt_only_adjudication_sha256='adjudication')
    verify_full_audits(rows,deep,pairs)


def test_incremental_observer_uses_separate_idle_audit_class(tmp_path,monkeypatch):
    from .test_audit_follow import fixture as follow_fixture
    from . import audit_follow
    import json
    args,rows,calls,progress=follow_fixture(tmp_path,monkeypatch,count=51)
    case=progress['direct']['cases'][0]
    case.update(evaluation_failure_reason=REASON,native_complete=False,native_success=None,
        evaluation_success=False,adjudication_sha256='idlehash')
    (tmp_path/'progress.json').write_text(json.dumps(progress))
    observed=[]
    def idle(row,plan):
        observed.append(row['archive'])
        return dict(archive=row['archive'],passed=True,context_version='v3',evaluation_method='gpt_only',
            native_success=None,native_complete=False,adjudication_sha256='idlehash',
            artifact_manifest_sha256=row['artifact_manifest_sha256'])
    monkeypatch.setattr(subject,'audit_idle',idle)
    result=audit_follow.step(**args)
    assert len(calls)==50 and observed==[case['archive']]
    assert result['passed_count']==51 and result['native_complete_passed_count']==50
    assert result['adjudicated_failure_passed_count']==1 and result['status']=='waiting'
