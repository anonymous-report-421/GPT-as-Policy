"""Restore the selected-task opening chart; retain the withdrawn derivation for audit."""
from copy import deepcopy
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def add_overall_references(data, references):
    result = deepcopy(data)
    subset = result['queries']['robolab_models']
    assert len(subset['rows']) == 5 and all(r['episodes'] == 50 for r in subset['rows'])
    assert references['tasks'] == 120 and references['instruction_variant'] == 'default'
    rows = deepcopy(subset['rows'])
    for row in references['rows']:
        assert row['episodes'] == 1200 and 0 <= row['successes'] <= 1200
        assert abs(row['sr'] - 100 * row['successes'] / row['episodes']) <= .051
        rows.append({**row, 'source_kind': 'official_overall', 'reference_label': '†',
                     'benchmark': references['benchmark'], 'task_count': references['tasks'],
                     'instruction_variant': references['instruction_variant'],
                     'source_url': references['source_url'],
                     'source_data_url': references['source_data_url'],
                     'source_sha256': references['source_sha256'],
                     'source_generated_at': references['source_generated_at'],
                     'retrieved_on': references['retrieved_on'],
                     'selection': references['selection']})
    assert len(rows) == 9 and len({r['id'] for r in rows}) == 9
    source = deepcopy(subset['source'])
    source['label'] = 'RoboLab · ten-task evaluation and explicitly marked official overall references'
    source['selection'] = 'Unmarked: existing ten tasks × five trials. †: official RoboLab-120 / Default, 120 tasks × ten trials. Different evaluation scopes; not a matched ranking.'
    source['tables'].append({'name': 'RoboLab-120 official leaderboard · Default',
                             'href': references['source_url']})
    source['official_reference'] = {k: v for k, v in references.items() if k != 'rows'}
    result['queries']['robolab_opening_models'] = {
        'label': 'RoboLab success-rate overview (mixed evaluation scopes)',
        'rows': rows, 'source': source}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=HERE/'app/src/data.json')
    args = parser.parse_args()
    data = json.loads(args.snapshot.read_text())
    data['queries'].pop('robolab_opening_models', None)
    args.snapshot.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print('Removed withdrawn overall-reference query; selected-task results and report body preserved.')


if __name__ == '__main__':
    main()
