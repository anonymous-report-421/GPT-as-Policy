"""Synthetic video checks only; no real observations or model calls."""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from .io import write_json
from .robolab_server.debug_recorder import DebugVideoRecorder, DecisionTimeline, public_outputs, display_json


def fixture(root):
    controller, sim = root/'controller', root/'sim'
    controller.mkdir(parents=True)
    sim.mkdir()
    write_json(controller/'run.json', dict(control_dt=1/15, instruction='Stack four test blocks.'))
    write_json(controller/'result.json', dict(step_id=5, complete=True, truncated=True, success=False, reason='terminal'))
    history = []
    for decision, mode, start, n in [(0, 'student', 0, 2), (1, 'edit', 2, 2), (2, 'eef', 4, 1)]:
        proposed = np.zeros((15,8), np.float32)
        executed = proposed[:n].copy()
        # Zero delta override is still a Codex override, not a student chunk.
        if mode == 'edit':
            executed[:, 2] = .01
        np.savez_compressed(controller/f'proposal_{decision:03d}.npz', actions=proposed, raw_actions=proposed)
        np.savez_compressed(controller/f'execution_{decision:03d}.npz', actions=executed, states=executed*.5)
        response = dict(request_id=f'request-{decision}', mode=mode, steps=n,
            edit=dict(delta_position=[.01,0,0], delta_rotation_vector=[0,0,0], gripper='keep'),
            target=dict(position=[.01,0,0], quaternion_wxyz=[1,0,0,0], gripper_closed=False),
            reason='合成测试：根据记录的观测纠正末端位置，并非真实任务决策。',
            assessment=dict(current_subgoal='stack', execution_status='progressing',
                intent_status='aligned' if mode == 'student' else 'misaligned',
                intent_evidence='Synthetic FK evidence.', task_progress=dict(remaining=['stack'])) )
        write_json(controller/f'response_{decision:03d}.json', response)
        write_json(controller/f'request_{decision:03d}.json', dict(inference_seconds=.1,
            current_eef=dict(position=[0,0,0], quaternion_wxyz=[1,0,0,0])))
        history.append(dict(decision=decision, start_tick=start, end_tick=start+n, executed_steps=n,
            discarded_student_steps=15-n if mode == 'student' else 15,
            response=response, takeover_trigger='none' if mode == 'student' else 'wrong_intent'))
    write_json(controller/'history.json', history)
    with imageio.get_writer(str(sim/'sensors.mp4'), fps=15, codec='libx264', macro_block_size=2) as video:
        for tick in range(6):
            frame = np.zeros((90,320,3), np.uint8)
            frame[..., 0] = 25 + tick*20
            frame[:,160:,1] = 120
            video.append_data(frame)
    return controller


def test_tick_alignment_and_modified_vs_prefix(tmp_path):
    timeline = DecisionTimeline(fixture(tmp_path))
    assert [timeline.at(t)[0]['decision'] for t in range(6)] == [0,0,0,1,1,2]
    assert [timeline.at(t)[1] for t in range(6)] == [None,0,1,0,1,0]
    first, second, third = timeline.segments
    assert first['prefix_only'] and not first['codex_override'] and not first['numeric_action_changed']
    assert second['codex_override'] and second['numeric_action_changed']
    assert third['codex_override'] and not third['numeric_action_changed']
    assert timeline.at(6) == (None, None)


def test_video_preserves_all_frames_and_exact_1x_speed(tmp_path, monkeypatch):
    from .robolab_server import debug_recorder
    drawn = []
    original = debug_recorder.ImageDraw.ImageDraw.text
    def capture(draw, xy, text, *args, **kwargs):
        drawn.append(text)
        return original(draw, xy, text, *args, **kwargs)
    monkeypatch.setattr(debug_recorder.ImageDraw.ImageDraw, 'text', capture)
    controller = fixture(tmp_path)
    source = (controller.parent/'sim/sensors.mp4').read_bytes()
    recorder = DebugVideoRecorder(controller)
    report = recorder.render()
    assert report['frames'] == 6 and report['fps'] == 15
    assert report['display_language'] == 'zh-CN'
    assert any('GPT 正在修正' in line for line in drawn)
    assert any('合成测试：' in line for line in drawn)
    assert any('任务超时' in line for line in drawn)
    assert report['layout'] == 'compact_cards_v2'
    assert report['height'] == report['observation_height'] + 440
    assert report['duration_seconds'] == pytest.approx(6/15)
    assert report['simulation_last_observation_seconds'] == pytest.approx(5/15)
    assert report['reading_pause_frames'] == 0 and report['unattributed_frames'] == 0
    assert report['height'] > report['observation_height'] and report['width'] == 1280
    with imageio.get_reader(report['video_path']) as reader:
        frames = list(reader.iter_data())
    assert len(frames) == 6
    assert all(frame.shape == (report['height'],1280,3) for frame in frames)
    # Monotonically changing synthetic observations survive in the upper area.
    assert [int(f[50,50,0]) for f in frames] == sorted(int(f[50,50,0]) for f in frames)
    assert (controller.parent/'sim/sensors.mp4').read_bytes() == source
    assert json.loads(Path(report['decisions_path']).read_text())[1]['response']['mode'] == 'edit'
    assert recorder.render() == report  # Finalizing twice does not append/pause/overwrite.


def test_localization_preserves_original_protocol_and_free_text():
    value = dict(mode='eef', reason='failed', target=dict(position=[0.1, 0.2, 0.3], gripper_closed=False),
                 assessment=dict(execution_status='failed', intent_evidence='根据观测判断。'))
    before = json.dumps(value)
    rendered = json.loads(display_json(value))
    assert rendered['控制模式'] == '末端位姿纠正'
    assert rendered['决策说明'] == 'failed'  # Do not translate or reinterpret arbitrary evidence.
    assert rendered['目标位姿']['位置'] == [0.1, 0.2, 0.3]
    assert rendered['目标位姿']['夹爪闭合'] == '否'
    assert rendered['接管评估']['上段执行状态'] == '失败'
    assert json.dumps(value) == before


def test_bundled_chinese_font_and_compact_corrections():
    from .robolab_server.debug_recorder import load_font, check_chinese_coverage
    from .robolab_server.video_panel import correction_lines, fitted_lines
    font, path = load_font(22)
    assert path.endswith('assets/fonts/NotoSansCJKsc-Regular.otf')
    assert check_chinese_coverage(font, '方块纠正红绿蓝观察状态执行') == len(set('方块纠正红绿蓝观察状态执行'))
    lines = fitted_lines('很长的决策原因，需要在有限空间内显示。'*20, font, 180, 3)
    assert len(lines) == 3 and lines[-1].endswith('…')
    assert all(font.getlength(line) <= 180 for line in lines)
    segment = dict(response=dict(mode='eef', steps=5, target=dict(position=[.12, .17, .3],
        quaternion_wxyz=[1,0,0,0], gripper_closed=False)), executed_steps=5,
        request=dict(current_eef=dict(position=[.1,.2,.3], quaternion_wxyz=[1,0,0,0])))
    lines = correction_lines(segment)
    assert 'X +2.0   Y -3.0   Z +0.0 cm' in lines
    assert '姿态目标旋转 0.0°' in lines
    assert '夹爪：缺少原计划 / 执行指令记录' in lines


def test_gripper_before_after_uses_each_aligned_proposal_step():
    from .robolab_server.video_panel import gripper_change
    action = lambda grip: [0.0]*7+[grip]
    # Proposal changes midway through chunk; do not compare every step to step 0.
    segment = dict(pi05_actions=[action(1), action(0), action(0), action(1)],
                   executed_actions=[action(0), action(0), action(1), action(1)])
    assert gripper_change(segment, 0) == '夹爪 π0.5→GPT：闭合 → 打开'
    assert gripper_change(segment, 1) == '夹爪：保持打开（与 π0.5 一致）'
    assert gripper_change(segment, 2) == '夹爪 π0.5→GPT：打开 → 闭合'
    assert gripper_change(segment, 3) == '夹爪：保持闭合（与 π0.5 一致）'
    assert '缺少' in gripper_change(segment, 4)


def test_no_episode_does_not_fabricate_video_and_filters_private_reasoning(tmp_path):
    controller = tmp_path/'controller'
    controller.mkdir()
    assert DebugVideoRecorder(controller).render()['status'] == 'skipped'
    assert not (controller/'debug_video/debug_rollout.mp4').exists()
    workspace = controller/'codex_workspace'
    workspace.mkdir()
    events = [dict(method='item/completed', params=dict(item=dict(type=kind, text=text)))
              for kind, text in [('reasoning','private reasoning'), ('agentMessage','Public final result')]]
    (workspace/'rpc_out.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    outputs = public_outputs(controller)
    assert outputs['agent_messages'] == [dict(turn_id=None, text='Public final result')]


def test_client_finalizes_recorder_after_closing_simulator_connection(tmp_path):
    from .test_contract import make, invoke_next, response
    tools = make(tmp_path)
    invoke_next(tools)
    invoke_next(tools)
    final = invoke_next(tools, response=response(tools.request))
    assert final['result']['debug_video_path'].endswith('/debug_video/debug_rollout.mp4')
    order = []
    tools.sim.close = lambda: order.append('close_sim')
    tools.recorder.finalize = lambda: order.append('finalize_video')
    tools.close()
    assert order == ['close_sim', 'finalize_video']
