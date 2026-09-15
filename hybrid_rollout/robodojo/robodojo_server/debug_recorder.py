"""Render recorded observations and Codex decisions at simulation 1x speed.

Postprocessing only: no policy imports, model calls, simulator connections,
extra frames, or reading-time pauses. The original video/logs are untouched.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..io import sha256, write_json


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


LABELS = {
    'request_id': 'Request ID', 'mode': 'Control mode', 'steps': 'Requested steps',
    'reason': 'Decision rationale', 'edit': 'Local correction', 'target': 'Target pose',
    'assessment': 'Takeover assessment', 'delta_position': 'Position offset',
    'delta_rotation_vector': 'Rotation-vector offset', 'gripper': 'Gripper command',
    'position': 'Position', 'quaternion_wxyz': 'Quaternion (wxyz)',
    'gripper_closed': 'Gripper closed', 'task_progress': 'Task progress',
    'verified_completed': 'Verified complete', 'currently_attempting': 'Current attempt',
    'remaining': 'Remaining', 'current_subgoal': 'Current subgoal',
    'execution_status': 'Previous execution', 'execution_evidence': 'Execution evidence',
    'expected_next_intent': 'Expected next intent', 'predicted_next_intent': 'Predicted next intent',
    'intent_status': 'Intent status', 'intent_evidence': 'Intent evidence',
    'rotation_vector': 'Rotation vector', 'rpy_xyz': 'Euler angles (xyz)',
}
VALUES = {
    'student': 'Original student action', 'edit': 'Local trajectory correction',
    'eef': 'EEF pose correction', 'stop': 'Stop', 'not_started': 'Not started',
    'progressing': 'Progressing', 'failed': 'Failed', 'uncertain': 'Uncertain',
    'recovered': 'Recovered', 'aligned': 'Aligned', 'misaligned': 'Misaligned',
    'keep': 'Keep', 'open': 'Open', 'closed': 'Closed', 'none': 'None',
    'wrong_intent': 'Wrong intent', 'execution_failure': 'Execution failure',
    'both': 'Execution failure and wrong intent', 'terminal': 'Environment terminal',
    'decision_budget': 'Decision budget exhausted', 'model_stop': 'Model stop',
    'controller_error': 'Controller error or interruption',
}


def display_status(value):
    if value is None:
        return 'Not recorded'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    return VALUES.get(value, str(value))


def display_json(value, depth=0):
    """Keep vectors/lists inline so numeric output does not consume the panel."""
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    rows = [('  '*(depth+1) + json.dumps(LABELS.get(key, key), ensure_ascii=False) + ': ' +
             display_json(display_status(item) if key in ('mode', 'execution_status', 'intent_status', 'gripper', 'gripper_closed') else item, depth+1))
            for key, item in value.items()]
    return '{\n' + ',\n'.join(rows) + '\n' + '  '*depth + '}'


def wrap_lines(text, font, width):
    """Pixel-width wrapping, including CJK and long unbroken request IDs."""
    lines = []
    for paragraph in str(text).split('\n'):
        if not paragraph:
            lines.append('')
        while paragraph:
            lo, hi = 1, len(paragraph)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if font.getlength(paragraph[:mid]) <= width:
                    lo = mid
                else:
                    hi = mid - 1
            cut = lo
            if cut < len(paragraph):
                space = paragraph.rfind(' ', 0, cut + 1)
                if space > cut // 2:
                    cut = space + 1
            lines.append(paragraph[:cut].rstrip())
            paragraph = paragraph[cut:]
    return lines


def load_font(size, supplied=None):
    candidates = [supplied, os.environ.get('DEBUG_VIDEO_FONT'),
        Path(__file__).parents[2]/'assets/fonts/NotoSansCJKsc-Regular.otf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(str(path), size), str(path)
    raise FileNotFoundError('Set DEBUG_VIDEO_FONT or --font to an installed TrueType/OpenType font')


def check_glyph_coverage(font, text):
    """Check every non-ASCII glyph used by the English UI and public output."""
    absent = bytes(font.getmask('\uffff'))
    characters = {c for c in text if ord(c) > 127 and not c.isspace()}
    missing = sorted(c for c in characters if bytes(font.getmask(c)) == absent)
    if missing:
        raise ValueError('Video font lacks required glyphs: ' + ''.join(missing))
    return len(characters)


class DecisionTimeline:
    """A frame at tick t>0 shows the ACK for action t-1, never a future chunk."""
    def __init__(self, controller):
        self.controller = Path(controller)
        self.run = read_json(self.controller/'run.json', {})
        self.result = read_json(self.controller/'result.json', {})
        adjudication_path = self.controller.parent/'evaluation_adjudication.json'
        self.adjudication = None
        if adjudication_path.is_file():
            from ..idle_timeout_policy import verified_adjudication
            self.adjudication = verified_adjudication(self.controller.parent)
            # Display-only result: never synthesize a native controller/result.json.
            self.result = dict(self.result, complete=False, success=False,
                step_id=self.adjudication['control_steps'],
                evaluation_failure_reason=self.adjudication['reason'],
                native_score=None, native_complete=False)
        direct = self.run.get('evaluation_method') == 'gpt_only'
        self.records = read_json(self.controller/'history.json', [])
        self.segments = []
        previous_end = 0
        for record in self.records:
            n = record.get('executed_steps', 0)
            if not n:
                continue
            start, end = record['start_tick'], record['end_tick']
            if start < previous_end or end - start != n:
                raise ValueError('Overlapping history or control count mismatch')
            decision = record['decision']
            if direct:
                proposed = raw = np.empty((0, 14), dtype=np.float32)
                if (self.controller/f'proposal_{decision:03d}.npz').exists():
                    raise ValueError('GPT-only recording unexpectedly contains a student proposal')
            else:
                with np.load(self.controller/f'proposal_{decision:03d}.npz', allow_pickle=False) as data:
                    proposed = data['actions'].copy()
                    raw = data['raw_actions'].copy()
            with np.load(self.controller/f'execution_{decision:03d}.npz', allow_pickle=False) as data:
                executed, states = data['actions'].copy(), data['states'].copy()
            horizon, dim = self.run.get('action_horizon', 50), self.run.get('action_dim', 14)
            if (not direct and proposed.shape != (horizon, dim)) or executed.shape != (n, dim) or states.shape != (n, dim):
                raise ValueError('Unexpected proposal/execution array shape')
            if not all(np.isfinite(a).all() for a in (proposed, raw, executed, states)):
                raise ValueError('Nonfinite action or state in debug recording')
            mode = record['response']['mode']
            delta = np.empty((0, dim)) if direct else executed - proposed[:n]
            request = read_json(self.controller/f'request_{decision:03d}.json', {})
            segment = dict(record, request=request, evaluation_method='gpt_only' if direct else 'pi05_plus_gpt',
                raw_pi05_actions=raw.tolist(),
                pi05_actions=proposed.tolist(), executed_actions=executed.tolist(),
                measured_states=states.tolist(), delta_vs_pi05_prefix=delta.tolist(),
                eef_execution=read_json(self.controller/f'edit_{decision:03d}.json', []),
                codex_override=not direct and mode in ('edit', 'eef'),
                prefix_only=mode == 'student' and record['response']['steps'] < horizon,
                robot_profile='robodojo',
                numeric_action_changed=None if direct else bool(np.any(np.abs(delta) > 1e-6)))
            self.segments.append(segment)
            previous_end = end
        self.starts = [s['start_tick'] for s in self.segments]

    def at(self, tick):
        action_tick = max(0, tick - 1)
        index = bisect.bisect_right(self.starts, action_tick) - 1
        if index < 0 or action_tick >= self.segments[index]['end_tick']:
            return None, None
        return self.segments[index], None if tick == 0 else action_tick - self.starts[index]


def public_outputs(controller):
    """Only public agent messages and structured decisions; no reasoning stream."""
    messages = []
    rpc = controller/'codex_workspace/rpc_out.jsonl'
    if rpc.is_file():
        with rpc.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                params = event.get('params', {})
                item = params.get('item', {})
                if event.get('method') == 'item/completed' and item.get('type') == 'agentMessage':
                    messages.append(dict(turn_id=params.get('turnId'), text=item.get('text', '')))
    return dict(agent_messages=messages,
        decisions=[dict(path=str(p), response=read_json(p)) for p in sorted(controller.glob('response_*.json'))])


class DebugVideoRecorder:
    def __init__(self, controller, *, output=None, font=None, presentation='legacy',
                 robodojo_source=None, usd_python='/usr/bin/python3', encoder_threads=None):
        self.controller = Path(controller).resolve()
        self.output = Path(output).resolve() if output else self.controller/'debug_video'
        self.font = font
        self.presentation = presentation
        self.robodojo_source, self.usd_python = robodojo_source, usd_python
        if encoder_threads is not None and encoder_threads < 1:
            raise ValueError('Encoder thread limit must be positive')
        self.encoder_threads = encoder_threads

    def render(self):
        manifest_path = self.output/'manifest.json'
        if self.output.exists():
            manifest = read_json(manifest_path, {})
            if manifest.get('status') == 'completed':
                return manifest
            raise FileExistsError(f'Refusing to overwrite unfinished debug output: {self.output}')
        self.output.mkdir(parents=True)
        source = self.controller.parent/'sim/sensors.mp4'
        write_json(manifest_path, dict(status='rendering', source_video=str(source), pid=os.getpid()))
        try:
            if not source.is_file() or not (self.controller/'run.json').is_file():
                result = dict(status='skipped', reason='No recorded episode/video; no frames fabricated',
                              source_video=str(source))
                write_json(manifest_path, result)
                return result
            timeline = DecisionTimeline(self.controller)
            dt = float(timeline.run['control_dt'])
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError('A positive simulation control_dt is required')
            fps = 1 / dt
            font, font_path = load_font(22, self.font)
            from .video_panel import VideoPanel, PANEL_HEIGHT, TERMINAL_LABEL_VERSION
            checked_glyphs = check_glyph_coverage(font,
                json.dumps(timeline.records, ensure_ascii=False)
                + Path(__file__).with_name('video_panel.py').read_text())
            panel_ui = VideoPanel(timeline, font_path)
            arrows = None
            calibration = None
            layout = 'compact_cards_v3_en'
            if self.presentation == 'paper':
                from .video_panel_paper import PaperVideoPanel, LAYOUT_VERSION
                from .video_projection import CommandArrows, calibration_for_run
                if not self.robodojo_source:
                    raise ValueError('Paper preview requires an explicit --robodojo-source for verified camera reconstruction')
                calibration = calibration_for_run(self.controller.parent,self.robodojo_source,self.usd_python)
                arrows = CommandArrows(timeline,calibration)
                panel_ui = PaperVideoPanel(timeline,font_path,arrows)
                layout = LAYOUT_VERSION
                write_json(self.output/'camera_calibration.json',calibration)
            width = 1280
            reader = imageio.get_reader(str(source),**(
                {'input_params':['-threads',str(self.encoder_threads)]} if self.encoder_threads else {}))
            writer = None
            frame_count = unattributed = 0
            temporary = self.output/'debug_rollout.partial.mp4'
            output_video = self.output/'debug_rollout.mp4'
            try:
                metadata = reader.get_meta_data()
                if not math.isclose(float(metadata['fps']), fps, rel_tol=1e-3):
                    raise ValueError('Source video FPS differs from simulation clock; refusing time resampling')
                top_height = None
                for tick, frame in enumerate(reader):
                    rgb = Image.fromarray(frame[..., :3])
                    segment, row = timeline.at(tick)
                    canvas = panel_ui.render(rgb, segment, tick, dt)
                    if top_height is None:
                        top_height = round(rgb.height * width / rgb.width)
                        top_height += top_height % 2
                        panel_height = PANEL_HEIGHT
                        height = canvas.height
                        writer = imageio.get_writer(str(temporary), fps=fps, codec='libx264',
                            pixelformat='yuv420p', macro_block_size=2,
                            output_params=['-preset', 'fast', '-crf', '20', '-movflags', '+faststart'] + (
                                ['-threads',str(self.encoder_threads)] if self.encoder_threads else []))
                    if segment is None:
                        unattributed += 1
                    writer.append_data(np.asarray(canvas))
                    frame_count += 1
            finally:
                reader.close()
                if writer is not None:
                    writer.close()
            if frame_count == 0:
                raise ValueError('Recorded video contains no decodable frames')
            # The source records exactly one initial observation and one per control.
            expected = timeline.result.get('step_id')
            if timeline.result.get('complete') and frame_count != expected + 1:
                raise ValueError('Completed episode frame count is not controls + initial observation')
            os.replace(temporary, output_video)
            write_json(self.output/'decisions.json', timeline.segments)
            write_json(self.output/'codex_outputs.json', public_outputs(self.controller))
            if arrows:
                write_json(self.output/'projected_commands.json',arrows.records)
            inputs = [source, self.controller/'run.json', self.controller/'history.json',
                      self.controller.parent/'evaluation_adjudication.json',
                      *self.controller.glob('request_*.json'), *self.controller.glob('response_*.json'),
                      *self.controller.glob('proposal_*.npz'), *self.controller.glob('execution_*.npz')]
            result = dict(status='completed', display_language='en', layout=layout, video_path=str(output_video), source_video=str(source),
                frames=frame_count, fps=fps, duration_seconds=frame_count/fps,
                simulation_last_observation_seconds=(frame_count-1)*dt,
                playback_speed=1.0, reading_pause_frames=0, frame_mapping='initial frame 0; frame t>0 shows action t-1 ACK',
                width=width, height=height, observation_height=top_height, font=font_path, font_sha256=sha256(font_path),
                unattributed_frames=unattributed, episode_result=timeline.result,
                decisions_path=str(self.output/'decisions.json'), outputs_path=str(self.output/'codex_outputs.json'),
                input_sha256={str(p): sha256(p) for p in inputs if p.is_file()},
                checked_non_ascii_glyphs=checked_glyphs,
                recorder_sha256=sha256(Path(__file__)), panel_sha256=sha256(Path(__file__).with_name('video_panel.py')),
                terminal_label_version=TERMINAL_LABEL_VERSION,
                evaluation_adjudication=timeline.adjudication,
                video_sha256=sha256(output_video))
            if arrows:
                extra_files=[Path(__file__).with_name(name) for name in ('video_projection.py','video_panel_paper.py')]
                result.update(presentation_source_sha256={str(p):sha256(p) for p in extra_files},
                    encoder_threads=self.encoder_threads,
                    camera_calibration_path=str(self.output/'camera_calibration.json'),
                    camera_calibration_sha256=sha256(self.output/'camera_calibration.json'),
                    projected_commands_path=str(self.output/'projected_commands.json'),
                    projected_commands_sha256=sha256(self.output/'projected_commands.json'),
                    arrow_frames=len(arrows.drawn_frames),
                    arrow_semantics='Recorded EEF translation direction, hybrid corrections only; not measured arrival',
                    task_prompt=timeline.run.get('instruction'),original_video_unchanged=True)
                # Only poses were read from each NPZ; hash the original evidence
                # for reproducible re-rendering without any simulation restore.
                result['input_sha256'].update(calibration['input_sha256'])
                result['input_sha256'].update({str(p):sha256(p) for p in
                    (arrows.observations/f'{tick:06d}.npz' for tick in sorted({r['tick'] for r in arrows.records}))})
            write_json(manifest_path, result)
            return result
        except Exception:
            write_json(manifest_path, dict(status='failed', error=traceback.format_exc(), source_video=str(source)))
            raise

    def finalize(self):
        """A rendering failure must never invalidate or retry physical execution."""
        try:
            result = self.render()
        except Exception:
            result = dict(status='failed', error=traceback.format_exc(), output_dir=str(self.output))
        print(json.dumps(dict(event='debug_video', status=result['status'],
            video_path=result.get('video_path'), manifest_path=str(self.output/'manifest.json'),
            frames=result.get('frames'), fps=result.get('fps'), error=result.get('error')),
            ensure_ascii=False), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--presentation',choices=('legacy','paper'),default='legacy')
    parser.add_argument('--robodojo-source',type=Path)
    parser.add_argument('--usd-python',default='/usr/bin/python3')
    parser.add_argument('--encoder-threads',type=int)
    parser.add_argument('--wait', action='store_true', help='Wait for launcher exit; never connect to simulation')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_dir():
        parser.error('Run root must already exist')
    if args.wait:
        print(json.dumps(dict(event='waiting_for_rollout_exit', run_root=str(root), pid=os.getpid())), flush=True)
        while not (root/'job_exit_status.txt').is_file():
            time.sleep(5)
    result = DebugVideoRecorder(root/'controller', output=args.output, font=args.font,
        presentation=args.presentation,robodojo_source=args.robodojo_source,usd_python=args.usd_python,
        encoder_threads=args.encoder_threads).finalize()
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
