"""Copy the public, official leaderboard's model marks for an offline report."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

BASE = 'https://media.luminis-sim.com/media/shared/teams/'
FILES = {'DM0.5':'dm05.png','GalaxeaVLA (G0.5)':'GalaxeaVLA.png',
         'Xiaomi-Robotics-1':'xiaomi.png','OpenWAM-α':'openwam-cube.png',
         'Meituan-Robotics-0':'meituan.png','Hy-Embodied-0.5-VLA':'hunyuan-color.svg',
         'Spatial Forcing':'spatial.png','Pi-05':'pi.png','InternVLA-A1.5':'ailab.png',
         'StarVLA-PI_v3':'spatial.png'}


def main():
    def fetch(filename):
        cache = Path('/tmp/report-model-logo-cache')
        cache.mkdir(exist_ok=True)
        target = cache/filename
        if not target.exists():
            subprocess.run(['curl','-fsSL','--retry','2','--max-time','25','-o',str(target)+'.part',BASE+filename],check=True)
            Path(str(target)+'.part').replace(target)
        data = target.read_bytes()
        if filename.endswith('.svg') and any(x in data.lower() for x in (b'<script', b'<foreignobject', b'javascript:')):
            raise ValueError('Unsafe SVG')
        mime = 'image/svg+xml' if filename.endswith('.svg') else 'image/png'
        return filename, dict(src='data:'+mime+';base64,'+base64.b64encode(data).decode(),
                             source=BASE+filename, sha256=hashlib.sha256(data).hexdigest())
    with ThreadPoolExecutor(max_workers=5) as pool:
        fetched = dict(pool.map(fetch, sorted(set(FILES.values()))))
    logos = {model:fetched[file] for model,file in FILES.items()}
    data = Path('/tmp/report-openai-favicon.ico').read_bytes()
    if not data.startswith(b'\x00\x00\x01\x00'):
        raise ValueError('ChatGPT official icon is not ICO')
    logos['gpt'] = dict(src='data:image/x-icon;base64,'+base64.b64encode(data).decode(),
                        source='https://cdn.oaistatic.com/assets/favicon-eex17e9e.ico',sha256=hashlib.sha256(data).hexdigest())
    target = Path(__file__).parent/'app/src/content/report/model-logos.json'
    target.write_text(json.dumps(logos,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(logos)} verified model logo mappings; {target.stat().st_size} bytes')


if __name__ == '__main__':
    main()
