"""Offline fixtures only: review writes must never alter original rollout evidence."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from .annotation_dashboard import AnnotationStore, Catalog, Conflict, export, handler


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_run(root, version='v2', state='success', name='run', pause_frames=0, method='pi05_plus_gpt'):
    run = root/name/'build_tower/standard/eval_seed_0/layout_2/replica_6/attempt_0'
    identity = dict(case_id='build_tower__standard__g0__l2', task='build_tower', variant='standard',
                    eval_seed=0, layout_id=2, reset_seed=2, simulator_initial_seed=0, policy_rng_seed=0)
    put(run/'controller/run.json', dict(evaluation_case=identity, context_version=version,
        evaluation_method=method, student_backend='jax', teacher_model='gpt-6-astra',
        teacher_reasoning_effort='xhigh', max_episode_steps=1050, instruction='Build a tower.'))
    put(run/'controller/progress.json', dict(step_id=100))
    if state in ('success', 'failure', 'invalid', 'unverified'):
        put(run/'controller/result.json', dict(complete=True, success=state=='success', step_id=100))
    if state in ('success', 'failure', 'invalid'):
        invalid = state == 'invalid'
        put(run/'sim/evaluation_outcome.json', dict(evaluation_case=identity, complete=True,
            valid_for_success_rate=not invalid, native_success=None if invalid else state=='success',
            native_score=None if invalid else 1.0 if state=='success' else 0.5,
            status='invalid_native_layout' if invalid else 'native_completed'))
    if state == 'interrupted':
        put(run/'controller/failure.json', dict(error='Network interrupted'))
    video = run/'controller/debug_video/debug_rollout.mp4'
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b'0123456789')
    put(video.parent/'manifest.json', dict(status='complete', fps=25, frames=101,
        duration_seconds=4.04, reading_pause_frames=pause_frames, playback_speed=1,
        frame_mapping='initial frame 0; frame t>0 shows action t-1 ACK', video_sha256='fake_test_hash'))
    put(run/'controller/history.json', [dict(decision=0, start_tick=0, end_tick=15,
        response=dict(mode='student', reason='Public reason', hidden_field='must not be exported'))])
    put(run/'controller/codex_workspace/rpc_out.jsonl', dict(secret='not exposed'))
    return run


def annotation(row, revision=0):
    return dict(revision=revision, reviewed=True, selected=True, tags=['恢复', '恢复'],
        comment='Public behavior analysis', clips=[dict(video_key=row['videos'][0]['key'], start=1.04,
            end=2.08, comment='A useful segment', tags=['纠错成功'])])


@pytest.mark.parametrize('state', ['success', 'failure', 'invalid', 'unverified', 'interrupted', 'unfinished'])
def test_native_status_and_metadata(tmp_path, state):
    root = tmp_path/'results'; make_run(root, state=state)
    catalog = Catalog(root); row = catalog.scan()[0]
    assert row['status'] == state
    assert row['seeds']['reset_seed'] == 2 and row['context_version'] == 'v2'
    if state not in ('success', 'failure'):
        assert row['native_score'] is None
    detail = catalog.detail(row['id'])
    assert detail['decisions'][0]['reason'] == 'Public reason'
    assert 'must not be exported' not in json.dumps(detail)
    assert 'secret' not in json.dumps(detail)


def test_attempts_are_distinct_and_no_walk_of_raw_observations(tmp_path):
    root = tmp_path/'results'
    first = make_run(root, version='v1', name='first')
    make_run(root, version='v2', name='second', method='gpt_only')
    # A lookalike inside the original observation tree must not be catalogued.
    make_run(first/'sim/observations', name='not_a_run')
    c = Catalog(root); rows = c.scan()
    assert len(rows) == 2 and len({r['id'] for r in rows}) == 2
    assert {r['method'] for r in rows} == {'pi05_plus_gpt', 'gpt_only'}
    assert len({r['case_id'] for r in rows}) == 1


def test_partial_metadata_and_frame_mapping(tmp_path):
    root = tmp_path/'results'; run = make_run(root, pause_frames=5)
    put(run/'controller/run.json', ['unexpected schema'])
    (run/'controller/history.json').write_text('[')
    row = Catalog(root).scan()[0]
    assert row['task'] == 'build_tower' and not row['videos'][0]['step_mapping']


def test_no_symlink_or_path_escape(tmp_path):
    root = tmp_path/'results'; run = make_run(root)
    outside = tmp_path/'private'; make_run(outside, name='outside')
    (root/'linked').symlink_to(outside, target_is_directory=True)
    catalog = Catalog(root); rows = catalog.scan()
    assert len(rows) == 1
    for bad in ('../private', str(outside), 'x'*24, 'f'*24):
        with pytest.raises(FileNotFoundError): catalog.resolve(bad)
    with pytest.raises(FileNotFoundError): catalog.video_path(rows[0]['id'], '../../private')
    video = run/'controller/debug_video/debug_rollout.mp4'
    video.unlink(); video.symlink_to(outside/'secret')
    assert catalog.videos(run) == []


def test_versioned_annotations_and_originals_unchanged(tmp_path):
    root = tmp_path/'results'; make_run(root)
    original = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    catalog = Catalog(root); row = catalog.detail(catalog.scan()[0]['id'])
    store = AnnotationStore(tmp_path/'annotations', root)
    assert store.get(row['id'])['revision'] == 0
    first = store.save(row['id'], annotation(row), row)
    assert first['revision'] == 1 and first['tags'] == ['恢复']
    assert first['clips'][0]['start_step'] == 26 and first['clips'][0]['end_step'] == 52
    assert first['clips'][0]['video_sha256'] == 'fake_test_hash'
    second = annotation(row, revision=1); second['comment'] = 'Changed comment'; second['clips'] = []
    store.save(row['id'], second, row)
    fresh = AnnotationStore(tmp_path/'annotations', root)
    assert fresh.get(row['id'])['comment'] == 'Changed comment'
    assert [r['revision'] for r in fresh.history(row['id'])] == [2,1]
    assert fresh.history(row['id'])[1]['clips'][0]['comment'] == 'A useful segment'
    with pytest.raises(Conflict): store.save(row['id'], annotation(row), row)
    assert original == {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_two_store_instances_do_not_overwrite(tmp_path):
    root = tmp_path/'results'; make_run(root); c=Catalog(root); r=c.detail(c.scan()[0]['id'])
    stores = [AnnotationStore(tmp_path/'annotations', root) for _ in range(2)]
    def save(store):
        try: return store.save(r['id'], annotation(r), r)['revision']
        except Conflict: return 'conflict'
    with ThreadPoolExecutor(2) as executor: results=list(executor.map(save,stores))
    assert sorted(map(str,results)) == ['1','conflict']


@pytest.mark.parametrize('mutation', [
    lambda a: a.update(revision=-1), lambda a: a.update(selected='true'),
    lambda a: a.update(native_success=True), lambda a: a.update(tags=[1]),
    lambda a: a['clips'][0].update(start=-1), lambda a: a['clips'][0].update(end=5),
    lambda a: a['clips'][0].update(start=float('nan')), lambda a: a['clips'][0].update(video_key='../private'),
    lambda a: a['clips'][0].update(start=3,end=2), lambda a: a.update(comment='x'*20001),
])
def test_annotation_validation(tmp_path,mutation):
    root=tmp_path/'results';make_run(root);c=Catalog(root);r=c.detail(c.scan()[0]['id']);a=annotation(r);mutation(a)
    with pytest.raises(ValueError): AnnotationStore(tmp_path/'annotations',root).save(r['id'],a,r)


def test_annotation_root_must_not_overlap_results(tmp_path):
    root=tmp_path/'results';root.mkdir()
    for path in (root,root/'annotations',tmp_path):
        with pytest.raises(ValueError): AnnotationStore(path,root)


def test_corrupt_revision_not_silently_overwritten(tmp_path):
    root=tmp_path/'results';make_run(root);c=Catalog(root);r=c.detail(c.scan()[0]['id']);store=AnnotationStore(tmp_path/'annotations',root)
    store.save(r['id'],annotation(r),r)
    (store.root/r['id']/'00000001.json').write_text('{')
    with pytest.raises(ValueError): store.save(r['id'],annotation(r),r)


def test_export_selected_and_csv_formula_escaping(tmp_path):
    root=tmp_path/'results';make_run(root);make_run(root,name='second');c=Catalog(root)
    r=c.detail(c.scan()[0]['id']);store=AnnotationStore(tmp_path/'annotations',root);a=annotation(r);a['comment']='=dangerous()'
    store.save(r['id'],a,r)
    body,_=export(c,store,True,'json');data=json.loads(body)
    assert len(data['rows'])==1 and data['rows'][0]['annotation']['clips'][0]['start_step']==26
    body,_=export(c,store,True,'csv');assert "'=dangerous()" in body.decode()
    body,_=export(c,store,True,'md');assert b'1.040' in body
    assert len(json.loads(export(c,store,False,'json')[0])['rows'])==2


def test_export_context_versions_do_not_merge_or_remove_old_clips(tmp_path):
    root = tmp_path/'results'
    for v in ('v1', 'v2', 'v3'):
        make_run(root, name=v, version=v)
    c = Catalog(root); store = AnnotationStore(tmp_path/'annotations', root)
    for row in c.scan():
        store.save(row['id'], annotation(row), c.detail(row['id']))
    before = {str(p):p.read_bytes() for p in store.root.rglob('*.json')}
    for v in ('v1', 'v2', 'v3'):
        data = json.loads(export(c, store, True, 'json', version=v)[0])
        assert len(data['rows']) == 1 and data['rows'][0]['context_version'] == v
        assert data['rows'][0]['annotation']['clips']
        for fmt in ('md', 'csv'):
            assert export(c, store, True, fmt, version=v)[0]
    assert len(json.loads(export(c, store, True, 'json')[0])['rows']) == 3
    assert before == {str(p):p.read_bytes() for p in store.root.rglob('*.json')}


@pytest.fixture
def http_server(tmp_path):
    root=tmp_path/'results';make_run(root);catalog=Catalog(root);store=AnnotationStore(tmp_path/'annotations',root)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(catalog,store));server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try: yield server,catalog,store
    finally: server.shutdown();server.server_close();thread.join(timeout=5)


def request(server,method,path,body=None,headers=None,address='127.0.0.1'):
    conn=HTTPConnection(address,server.server_port,timeout=5)
    try:
        conn.request(method,path,body=body,headers=headers or {});response=conn.getresponse()
        return response.status,dict(response.getheaders()),response.read()
    finally: conn.close()


def test_http_csrf_conflict_and_no_exposed_raw_files(http_server):
    server,catalog,store=http_server
    status,headers,body=request(server,'GET','/api/runs');assert status==200
    data=json.loads(body);r=data['rows'][0]
    payload=json.dumps(dict(id=r['id'],annotation=annotation(r)))
    url='/api/annotation';headers={'Content-Type':'application/json'}
    assert request(server,'POST',url,payload,headers)[0]==403
    headers['X-Review-Token']=data['csrf']
    assert request(server,'POST',url,payload,dict(headers,Origin='https://evil.example'))[0]==403
    assert request(server,'POST',url,payload,headers)[0]==200
    assert request(server,'POST',url,payload,headers)[0]==409
    assert request(server,'GET','/api/runs',headers={'Host':'evil.example'})[0]==403
    for path in ('/api/run?id=../../secret','/controller/codex_workspace/rpc_out.jsonl','/video?id='+r['id']+'&video=../private','/../annotation_dashboard.py'):
        assert request(server,'GET',path)[0]==404
    assert json.loads(request(server,'GET','/api/revisions?id='+r['id'])[2])['revisions'][0]['revision']==1
    for path in ('/','/app.js','/style.css'):
        status,h,body=request(server,'GET',path);assert status==200 and body
        assert "frame-ancestors 'none'" in h['Content-Security-Policy']


def test_all_interfaces_destination_ip_and_same_origin_save(tmp_path):
    root=tmp_path/'results';make_run(root);catalog=Catalog(root)
    store=AnnotationStore(tmp_path/'annotations',root)
    server=ThreadingHTTPServer(('0.0.0.0',0),handler(catalog,store));server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        # This IP is not in the localhost Host allowlist: accept the actual
        # socket destination without admitting arbitrary domains or other IPs.
        address='127.0.0.2'
        status,_,body=request(server,'GET','/api/runs',address=address)
        assert status==200
        data=json.loads(body);row=data['rows'][0]
        headers={'Content-Type':'application/json','X-Review-Token':data['csrf'],
                 'Origin':f'http://{address}:{server.server_port}'}
        payload=json.dumps(dict(id=row['id'],annotation=annotation(row)))
        assert request(server,'POST','/api/annotation',payload,headers,address)[0]==200
        assert request(server,'POST','/api/annotation',payload,
                       dict(headers,Origin='http://evil.example'),address)[0]==403
        for bad in ('evil.example','10.20.30.40','0.0.0.0','user@127.0.0.2',
                    '127.0.0.2:bad','127.0.0.2/path','127.0.0.2#fragment'):
            assert request(server,'GET','/api/runs',headers={'Host':bad},address=address)[0]==403
        url='/video?id='+row['id']+'&video='+row['videos'][0]['key']
        status,_,body=request(server,'GET',url,headers={'Range':'bytes=0-3'},address=address)
        assert status==206 and body==b'0123'
        assert store.get(row['id'])['revision']==1
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


@pytest.mark.parametrize('byte_range,expected', [('bytes=2-5',b'2345'),('bytes=-3',b'789'),('bytes=8-',b'89'),('bytes=0-99',b'0123456789')])
def test_http_video_ranges(http_server,byte_range,expected):
    server,catalog,_=http_server;r=catalog.scan()[0];url='/video?id='+r['id']+'&video='+r['videos'][0]['key']
    status,h,body=request(server,'GET',url,headers={'Range':byte_range})
    assert status==206 and body==expected and int(h['Content-Length'])==len(body)
    assert request(server,'HEAD',url,headers={'Range':byte_range})[2]==b''
    for bad in ('bytes=100-','bytes=5-2','bytes=-0','bytes=','bytes=0-1,4-5'):
        status,h,body=request(server,'GET',url,headers={'Range':bad})
        assert status==416 and h['Content-Range']=='bytes */10'
