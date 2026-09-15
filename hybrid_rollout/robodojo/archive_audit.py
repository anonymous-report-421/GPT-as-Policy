"""Read-only deep checks of saved complete trajectories, with reports outside them."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from .artifact_manifest import OBSERVATION_KEYS
from .campaign import optional, utc
from .io import sha256, write_json


def audit_saved_payload(root, step, manifest):
    """Check every actually recorded action/frame; caller proves outcome eligibility."""
    root = Path(root).resolve()
    reset = optional(root/'sim/reset.json')
    episode = root/'sim'/reset['episode_id']
    frames = sorted((episode/'observations').glob('*.npz'))
    actions = sorted(episode.glob('action_*.json'))
    assert [int(p.stem) for p in frames] == list(range(step+1))
    assert [int(p.stem.split('_')[-1]) for p in actions] == list(range(step))
    records = {r['path']:r['sha256'] for r in manifest['all_immutable_artifacts']}
    for path in frames:
        assert sha256(path) == records[str(path.relative_to(root))]
        with np.load(path, allow_pickle=False) as data:
            assert OBSERVATION_KEYS <= set(data.files)
            # Materialize EVERY member, which also verifies ZIP CRC and NPY
            # payload decoding. Merely inspecting data.files is insufficient.
            values = {k:data[k] for k in data.files}
        for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            assert values[key].dtype == np.uint8 and values[key].ndim == 3 and values[key].shape[-1] == 3
        assert values['states'].shape == (14,) and np.isfinite(values['states']).all()
        assert np.isfinite(values['eef_positions']).all() and np.isfinite(values['eef_quaternions_wxyz']).all()
    for i,path in enumerate(actions):
        assert sha256(path) == records[str(path.relative_to(root))]
        row = optional(path); value = np.asarray(row['executed_action'],dtype=float)
        assert row['step_id'] == i+1 and value.shape == (14,) and np.isfinite(value).all()
    video_records = []
    for relative in ('sim/sensors.mp4','controller/debug_video/debug_rollout.mp4'):
        path = root/relative
        assert sha256(path) == records[relative]
        process = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(path)],
            capture_output=True,text=True,check=True,timeout=180)
        stream = json.loads(process.stdout)['streams'][0]
        assert int(stream['nb_read_frames']) == step+1
        video_records.append(dict(path=relative,sha256=records[relative],stream=stream))
    return dict(steps=step, observations=len(frames), actions=len(actions), videos=video_records)


def audit(root):
    root = Path(root).resolve()
    result = optional(root/'controller/result.json')
    manifest = optional(root/'artifact_manifest.json')
    outcome = optional(root/'sim/evaluation_outcome.json')
    if not (result.get('complete') and outcome.get('valid_for_success_rate')
            and manifest.get('status') == 'verified'):
        raise ValueError('Only native-complete, archived trajectories can be deep-audited')
    payload = audit_saved_payload(root, result['step_id'], manifest)
    run = optional(root/'controller/run.json')
    return dict(archive=str(root),checked_utc=utc(),passed=True,**payload,
        context_version=run.get('context_version','v1'), native_complete=True,
        evaluation_method=run.get('evaluation_method','pi05_plus_gpt'),native_success=outcome['native_success'],
        artifact_manifest_sha256=sha256(root/'artifact_manifest.json'),
        checks='Every NPZ member decoded/CRC and numeric fields checked; action JSON/SHA; actual video frame decode count',
        original_data_modified=False,model_calls=0,simulation_runs=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); snapshot=optional(args.snapshot)
    roots=[Path(r['archive']).resolve() for r in snapshot['completed']]
    if any(args.output.resolve().is_relative_to(root) for root in roots):
        raise ValueError('Audit output must not alter original trajectory archives')
    if args.output.exists():
        raise ValueError('Use a new audit report path; never overwrite prior evidence')
    report=dict(snapshot=str(args.snapshot),snapshot_sha256=sha256(args.snapshot),started_utc=utc(),
                target=len(roots),rows=[],status='in_progress')
    write_json(args.output,report)
    for root in roots:
        try:
            row=audit(root)
        except Exception as error:
            row=dict(archive=str(root),passed=False,error_type=type(error).__name__,error=str(error),checked_utc=utc())
        report['rows'].append(row)
        write_json(args.output,report)
        print(json.dumps(dict(done=len(report['rows']),target=len(roots),passed=row['passed'],archive=str(root))),flush=True)
    report.update(status='passed' if all(r['passed'] for r in report['rows']) else 'failed',finished_utc=utc())
    write_json(args.output,report)


if __name__=='__main__':
    main()
