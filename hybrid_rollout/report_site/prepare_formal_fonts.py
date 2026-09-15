"""Subset licensed Noto Serif CJK SC and Source Serif 4 into offline report fonts."""
from pathlib import Path
import argparse
from fontTools import subset

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chinese', type=Path, required=True)
    parser.add_argument('--latin', type=Path, required=True)
    args = parser.parse_args()
    sources = list((HERE/'app/src/content/report').glob('*.jsx'))
    sources += list((HERE/'app/src/content/report').glob('*.json'))
    sources += [HERE/'app/src/data.json']
    text = ''.join(p.read_text() for p in sources)
    for source, filename, glyphs in [
        (args.chinese, 'report-song.woff', text),
        (args.latin, 'report-latin.woff', text),
    ]:
        options = subset.Options()
        options.flavor = 'woff'
        font = subset.load_font(str(source), options)
        worker = subset.Subsetter(options=options)
        worker.populate(text=glyphs + ''.join(chr(i) for i in range(32, 256)))
        worker.subset(font)
        target = HERE/'app/src/content/assets'/filename
        subset.save_font(font, str(target), options)
        print(filename, target.stat().st_size)

if __name__ == '__main__':
    main()
