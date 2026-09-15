"""Derive an offline Chinese webfont from the repository's licensed font."""
from pathlib import Path
import argparse
import shutil
from fontTools import subset

HERE = Path(__file__).resolve().parent

def build_font(site, app_only=False):
    sources = [HERE/'content.json', site/'data.json', HERE/'web/gallery.js', HERE/'web/gallery.html',
               HERE/'app/src/content/report/ReportContent.jsx']
    sources.extend((HERE/'app/src/content/report').glob('*.json'))
    sources.extend((HERE/'app/src/content/report').glob('*.jsx'))
    if (site/'hero_clip.json').exists():
        sources.append(site/'hero_clip.json')
    text = ''.join(p.read_text() for p in sources)
    options = subset.Options()
    options.flavor = 'woff'
    font = subset.load_font(str(HERE.parent/'assets/fonts/NotoSansCJKsc-Regular.otf'), options)
    worker = subset.Subsetter(options=options)
    worker.populate(text=text + ''.join(chr(i) for i in range(32, 256)))
    worker.subset(font)
    target = HERE/'app/src/content/assets/report-cjk.woff'
    subset.save_font(font, str(target), options)
    if not app_only:
        (site/'media/fonts').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, site/'media/fonts/report-cjk.woff')
        shutil.copyfile(HERE.parent/'assets/fonts/LICENSE', site/'media/fonts/LICENSE.txt')
    print('Offline CJK font', target.stat().st_size, 'bytes')

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('site',type=Path)
    parser.add_argument('--app-only', action='store_true')
    args=parser.parse_args()
    build_font(args.site, app_only=args.app_only)
