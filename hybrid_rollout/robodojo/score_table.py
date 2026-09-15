"""Readable final paired scores, explicitly exposing missing native scores."""
import argparse
import csv
import json
import math
from pathlib import Path

from .io import require, sha256, write_json
from .paired_evaluation import comparison


def score_summary(rows):
    values=[r['native_score'] for r in rows if r.get('native_score') is not None]
    require(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in values),
        'Invalid normalized native score')
    return dict(n=len(rows),score_n=len(values),
        native_complete=sum(r.get('native_complete',True) is True for r in rows),
        successes=sum(r.get('evaluation_success',r.get('native_success')) is True for r in rows),
        idle_failures=sum(r.get('evaluation_failure_reason')=='simulator_rpc_idle_timeout_900s' for r in rows),
        mean_score_100=100*sum(values)/len(rows) if values and len(values)==len(rows) else None,
        available_mean_score_100=100*sum(values)/len(values) if values else None)


def build(report):
    checked=comparison(report['hybrid'],report['direct'])
    rows=[]
    tasks=sorted({r['task'] for r in report['hybrid']['cases']})
    require(len(tasks)==10,'Need the full ten-task comparison')
    for task in tasks+['ALL_10_TASKS']:
        groups={key:[r for r in report[key]['cases'] if task=='ALL_10_TASKS' or r['task']==task]
            for key in ['hybrid','direct']}
        counts={k:score_summary(v) for k,v in groups.items()}
        require(counts['hybrid']['n']==counts['direct']['n']==(50 if task=='ALL_10_TASKS' else 5),
            'Missing task cases')
        rows.append(dict(task=task,**{f'{k}_{name}':value for k,stats in counts.items() for name,value in stats.items()}))
    return rows,checked


def export(input_path, output):
    input_path,output=Path(input_path),Path(output)
    require(not output.exists(),'Preserve old score tables; use a new output directory')
    report=json.loads(input_path.read_text());rows,checked=build(report)
    output.mkdir(parents=True)
    with (output/'scores.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def number(v):return 'N/A' if v is None else f'{v:.2f}'
    lines=['# Paired RoboDojo v3 scores','',
        'Same 10 tasks × 5 fixed cases per method. Scores below use a 0–100 display scale.','',
        '| Task | Hybrid success | Hybrid mean score | GPT-only success | GPT-only mean score | GPT scored cases | GPT available-score mean | Idle failures |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['task']} | {r['hybrid_successes']}/{r['hybrid_n']} | {number(r['hybrid_mean_score_100'])} | "
            f"{r['direct_successes']}/{r['direct_n']} | {number(r['direct_mean_score_100'])} | "
            f"{r['direct_score_n']}/{r['direct_n']} | {number(r['direct_available_mean_score_100'])} | {r['direct_idle_failures']} |")
    lines+=['','N/A means a full fixed-panel native-score mean cannot be computed without imputing a missing score. '
        'The available-score mean uses only the explicitly counted scored cases; it is not the full-panel mean. '
        'User-authorized 900s simulator idle failures count as failures for success rate, but retain null native scores and truncated recorded trajectories.',
        '',checked['caveat'],'']
    (output/'README.md').write_text('\n'.join(lines))
    write_json(output/'manifest.json',dict(input=str(input_path.resolve()),input_sha256=sha256(input_path),
        score_scale='0-100',missing_scores_imputed=False,rows=11,
        csv_sha256=sha256(output/'scores.csv'),markdown_sha256=sha256(output/'README.md')))
    return dict(output=str(output),tasks=10,rows=11)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.input,a.output)))
