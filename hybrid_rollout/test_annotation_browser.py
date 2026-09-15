"""Optional Chromium interaction test. No downloads during pytest collection/run.

Set ROLLOUT_REVIEW_PLAYWRIGHT to an already installed playwright module directory,
and PLAYWRIGHT_BROWSERS_PATH to its browser cache. Otherwise this test is skipped.
"""
import os
from pathlib import Path
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import pytest

from .annotation_dashboard import AnnotationStore, Catalog, handler
from .test_annotation_dashboard import make_run


@pytest.mark.skipif(not os.environ.get('ROLLOUT_REVIEW_PLAYWRIGHT'), reason='Optional local Playwright runtime not configured')
def test_real_browser_video_and_segment_annotations(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('node'):
        pytest.skip('Synthetic video/browser tools unavailable')
    root=tmp_path/'results';run=make_run(root)
    video=run/'controller/debug_video/debug_rollout.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
        'testsrc2=size=640x360:rate=25','-t','4.04','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True,timeout=30)
    catalog=Catalog(root);store=AnnotationStore(tmp_path/'annotations',root)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(catalog,store));server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    env=dict(os.environ, ROLLOUT_REVIEW_SCREENSHOTS=str(tmp_path))
    try:
        script=Path(__file__).with_name('annotation_ui')/'browser_smoke.cjs'
        result=subprocess.run(['node',str(script),f'http://127.0.0.1:{server.server_port}/'],
                              env=env,capture_output=True,text=True,timeout=120)
        assert result.returncode==0, result.stdout+'\n'+result.stderr
        print(result.stdout)
        row=catalog.scan()[0];saved=store.get(row['id'])
        assert saved['revision']==3 and len(saved['clips'])==2
        assert len(store.history(row['id']))==3
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)
