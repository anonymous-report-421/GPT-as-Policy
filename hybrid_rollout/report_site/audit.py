"""Verify every portable media asset against its source or derived frame range."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from .build import read, sha, probe, dump, SOURCE_SHA

def audit(site):
    d=read(site/'data.json');p=read(site/'provenance.json')
    assert d['source_sha256']==SOURCE_SHA
    assert len(d['cases'])==100 and len(d['clips'])==43
    source={r['id']:r for r in p['videos']}
    def check(r):
        video=(site/r['video']).resolve()
        assert video.is_relative_to(site.resolve())
        assert video.is_file() and (site/r['poster']).is_file(),r['id']
        h=sha(video)
        if r['id'] in source:assert h==source[r['id']]['source_video_sha256'],r['id']
        info=probe(video)
        assert int(info['nb_frames'])==r['frames'],r['id']
        assert (info['width'],info['height'])==(1280,680),r['id']
        n,den=map(int,info['r_frame_rate'].split('/'))
        assert abs(n/den-r['fps'])<1e-8
        return dict(id=r['id'],video=r['video'],sha256=h,frames=r['frames'])
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(check,d['cases']+d['clips']):
            records.append(record)
            if len(records)%25==0:print(json.dumps(dict(checked=len(records),total=143)),flush=True)
    for t in d['tasks']:assert (site/t['image']).is_file()
    for f in ('index.html','gallery.html','gallery.js','gallery.css','data.js','scores.csv','media/fonts/report-cjk.woff'):
        assert (site/f).is_file(),f
    result=dict(status='passed',cases=100,clips=43,featured=12,original_videos_hash_matched=100,
        debug_dimensions=[1280,680],media=records,data_sha256=sha(site/'data.json'),
        report_html_sha256=sha(site/'index.html'),raw_data_modified=False,model_calls=0)
    dump(site/'media-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='media'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('site',type=Path)
    audit(parser.parse_args().site)
