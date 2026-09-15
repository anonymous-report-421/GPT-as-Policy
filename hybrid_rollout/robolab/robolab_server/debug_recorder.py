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


LABELS_ZH = {
    'request_id': '请求编号', 'mode': '控制模式', 'steps': '请求步数', 'reason': '决策说明',
    'edit': '局部修正', 'target': '目标位姿', 'assessment': '接管评估',
    'delta_position': '位置偏移', 'delta_rotation_vector': '旋转向量偏移', 'gripper': '夹爪指令',
    'position': '位置', 'quaternion_wxyz': '四元数(wxyz)', 'gripper_closed': '夹爪闭合',
    'task_progress': '任务进度', 'verified_completed': '已确认完成',
    'currently_attempting': '正在尝试', 'remaining': '尚待完成', 'current_subgoal': '当前子目标',
    'execution_status': '上段执行状态', 'execution_evidence': '执行证据',
    'expected_next_intent': '期望的下一步意图', 'predicted_next_intent': '预测的下一步意图',
    'intent_status': '意图匹配状态', 'intent_evidence': '意图证据',
    'rotation_vector': '旋转向量', 'rpy_xyz': '欧拉角(xyz)',
}
VALUES_ZH = {
    'student': '学生原动作', 'edit': '局部轨迹修正', 'eef': '末端位姿纠正', 'stop': '停止',
    'not_started': '尚未开始', 'progressing': '正在推进', 'failed': '失败',
    'uncertain': '不确定', 'recovered': '已恢复', 'aligned': '一致', 'misaligned': '偏离',
    'keep': '保持', 'open': '打开', 'closed': '闭合', 'none': '无',
    'wrong_intent': '意图错误', 'execution_failure': '执行失败', 'both': '执行失败且意图错误',
    'terminal': '环境终止', 'decision_budget': '决策预算耗尽', 'model_stop': '模型主动停止',
    'controller_error': '控制器异常或中断',
}


def status_zh(value):
    if value is None:
        return '未记录'
    if isinstance(value, bool):
        return '是' if value else '否'
    return VALUES_ZH.get(value, str(value))


def display_json(value, depth=0):
    """Keep vectors/lists inline so numeric output does not consume the panel."""
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    rows = [('  '*(depth+1) + json.dumps(LABELS_ZH.get(key, key), ensure_ascii=False) + ': ' +
             display_json(status_zh(item) if key in ('mode', 'execution_status', 'intent_status', 'gripper', 'gripper_closed') else item, depth+1))
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


def check_chinese_coverage(font, text):
    """Check the actual episode vocabulary, not a handful of sample glyphs."""
    absent = bytes(font.getmask('\uffff'))
    characters = {c for c in text if '\u3400' <= c <= '\u9fff' or '\U00020000' <= c <= '\U000323af'}
    missing = sorted(c for c in characters if bytes(font.getmask(c)) == absent)
    if missing:
        raise ValueError('Video font lacks Chinese glyphs: ' + ''.join(missing))
    return len(characters)


class DecisionTimeline:
    """A frame at tick t>0 shows the ACK for action t-1, never a future chunk."""
    def __init__(self, controller):
        self.controller = Path(controller)
        self.run = read_json(self.controller/'run.json', {})
        self.result = read_json(self.controller/'result.json', {})
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
            with np.load(self.controller/f'proposal_{decision:03d}.npz', allow_pickle=False) as data:
                proposed = data['actions'].copy()
                raw = data['raw_actions'].copy()
            with np.load(self.controller/f'execution_{decision:03d}.npz', allow_pickle=False) as data:
                executed, states = data['actions'].copy(), data['states'].copy()
            if proposed.shape != (15, 8) or executed.shape != (n, 8) or states.shape != (n, 8):
                raise ValueError('Unexpected proposal/execution array shape')
            if not all(np.isfinite(a).all() for a in (proposed, raw, executed, states)):
                raise ValueError('Nonfinite action or state in debug recording')
            mode = record['response']['mode']
            delta = executed - proposed[:n]
            request = read_json(self.controller/f'request_{decision:03d}.json', {})
            segment = dict(record, request=request, raw_pi05_actions=raw.tolist(),
                pi05_actions=proposed.tolist(), executed_actions=executed.tolist(),
                measured_states=states.tolist(), delta_vs_pi05_prefix=delta.tolist(),
                eef_execution=read_json(self.controller/f'edit_{decision:03d}.json', []),
                codex_override=mode in ('edit', 'eef'),
                prefix_only=mode == 'student' and record['response']['steps'] < 15,
                numeric_action_changed=bool(np.any(np.abs(delta) > 1e-6)))
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
    def __init__(self, controller, *, output=None, font=None):
        self.controller = Path(controller).resolve()
        self.output = Path(output).resolve() if output else self.controller/'debug_video'
        self.font = font

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
            from .video_panel import VideoPanel, PANEL_HEIGHT
            checked_glyphs = check_chinese_coverage(font,
                json.dumps(timeline.records, ensure_ascii=False)
                + Path(__file__).with_name('video_panel.py').read_text())
            panel_ui = VideoPanel(timeline, font_path)
            width = 1280
            reader = imageio.get_reader(str(source))
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
                    if top_height is None:
                        top_height = round(rgb.height * width / rgb.width)
                        top_height += top_height % 2
                        panel_height = PANEL_HEIGHT
                        height = top_height + panel_height
                        height += height % 2
                        writer = imageio.get_writer(str(temporary), fps=fps, codec='libx264',
                            pixelformat='yuv420p', macro_block_size=2,
                            output_params=['-preset', 'fast', '-crf', '20', '-movflags', '+faststart'])
                    segment, row = timeline.at(tick)
                    if segment is None:
                        unattributed += 1
                    canvas = panel_ui.render(rgb, segment, tick, dt)
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
            inputs = [source, self.controller/'run.json', self.controller/'history.json',
                      *self.controller.glob('request_*.json'), *self.controller.glob('response_*.json'),
                      *self.controller.glob('proposal_*.npz'), *self.controller.glob('execution_*.npz')]
            result = dict(status='completed', display_language='zh-CN', layout='compact_cards_v2', video_path=str(output_video), source_video=str(source),
                frames=frame_count, fps=fps, duration_seconds=frame_count/fps,
                simulation_last_observation_seconds=(frame_count-1)*dt,
                playback_speed=1.0, reading_pause_frames=0, frame_mapping='initial frame 0; frame t>0 shows action t-1 ACK',
                width=width, height=height, observation_height=top_height, font=font_path, font_sha256=sha256(font_path),
                unattributed_frames=unattributed, episode_result=timeline.result,
                decisions_path=str(self.output/'decisions.json'), outputs_path=str(self.output/'codex_outputs.json'),
                input_sha256={str(p): sha256(p) for p in inputs if p.is_file()},
                checked_chinese_glyphs=checked_glyphs,
                recorder_sha256=sha256(Path(__file__)), panel_sha256=sha256(Path(__file__).with_name('video_panel.py')),
                video_sha256=sha256(output_video))
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
    parser.add_argument('--wait', action='store_true', help='Wait for launcher exit; never connect to simulation')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_dir():
        parser.error('Run root must already exist')
    if args.wait:
        print(json.dumps(dict(event='waiting_for_rollout_exit', run_root=str(root), pid=os.getpid())), flush=True)
        while not (root/'job_exit_status.txt').is_file():
            time.sleep(5)
    result = DebugVideoRecorder(root/'controller', output=args.output, font=args.font).finalize()
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
