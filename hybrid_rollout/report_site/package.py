"""Package an already validated static site without original/private archives."""
import argparse
import json
from pathlib import Path
import zipfile
from .build import read, sha, dump

def package(site, target):
    media=read(site/'media-audit.json');browser=read(site/'browser-smoke.json')
    assert media['status']==browser['status']=='passed'
    assert media['data_sha256']==sha(site/'data.json')
    assert media['report_html_sha256']==sha(site/'index.html')
    assert browser['videoPlayback'] and browser['referenceHover']
    assert not target.exists(), 'Do not overwrite a previous delivery'
    records=[]
    for file in sorted(site.rglob('*')):
        if not file.is_file():continue
        assert not file.is_symlink(),file
        records.append((file,file.relative_to(site)))
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for file,relative in records:z.write(file,'gpt-embodied-report/'+str(relative))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.infolist())==len(records)
    manifest=dict(zip_file=target.name,bytes=target.stat().st_size,sha256=sha(target),
        files=len(records),crc_verified=True,cases=100,clips=43,
        debug_ui='existing_debug_placeholder',robolab='pending',published=False)
    dump(target.with_suffix('.manifest.json'),manifest)
    print(json.dumps(manifest))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('site',type=Path);p.add_argument('zip',type=Path)
    a=p.parse_args();package(a.site,a.zip)
