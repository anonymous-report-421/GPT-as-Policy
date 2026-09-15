import json
from types import SimpleNamespace
import pytest
from .acp_query import user_jobs


def test_all_pages_and_no_500_response():
    calls=[]
    def run(argv, **kwargs):
        calls.append(argv)
        n=next(a for a in argv if a.startswith('--page-token='))
        page=int(n.split('=')[1]); count=100 if page==1 else 2
        return SimpleNamespace(stdout=json.dumps([dict(name=f'pt-{page}-{i}',
            ownership=dict(user_name='sujiayi')) for i in range(count)]))
    assert len(user_jobs('/sco','workspace',runner=run))==102
    assert len(calls)==2 and all('--page-size=100' in c for c in calls)


def test_repeated_page_blocks_submission():
    rows=[dict(name=f'pt-{i}',ownership=dict(user_name='sujiayi')) for i in range(100)]
    with pytest.raises(ValueError,match='repeated'):
        user_jobs('/sco','workspace',runner=lambda *a,**kw:SimpleNamespace(stdout=json.dumps(rows)))


def test_wrong_owner_blocks_incomplete_inventory():
    with pytest.raises(ValueError,match='owner'):
        user_jobs('/sco','workspace',runner=lambda *a,**kw:SimpleNamespace(
            stdout=json.dumps([dict(name='pt-other',ownership=dict(user_name='other'))])))


def test_exact_multiple_of_page_size_accepts_cli_empty_page():
    rows=[dict(name=f'pt-{i}',ownership=dict(user_name='sujiayi')) for i in range(100)]
    calls=[]
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=json.dumps(rows) if len(calls)==1 else 'No jobs found\n')
    assert user_jobs('/sco','workspace',runner=run)==rows
    assert len(calls)==2


@pytest.mark.parametrize('response', ['', ' ', '503 Service Unavailable', 'No jobs found: authentication failed'])
def test_unknown_non_json_response_still_blocks_submission(response):
    with pytest.raises(ValueError):
        user_jobs('/sco','workspace',runner=lambda *a,**kw:SimpleNamespace(stdout=response))
