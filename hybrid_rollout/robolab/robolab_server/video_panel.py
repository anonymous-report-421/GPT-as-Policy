"""Compact offline video UI. Display recorded evidence; never make action decisions."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_HEIGHT = 440
BG = '#0c1422'
CARD = '#152136'
INK = '#edf3fb'
MUTED = '#94a7c0'
AMBER = '#ffbd59'
CYAN = '#5bdacc'


def fitted_lines(text, font, width, max_lines):
    """Clip to a bounded card with explicit ellipsis, not off-screen text."""
    lines, current = [], ''
    for char in str(text):
        if char == '\n' or (current and font.getlength(current+char) > width):
            lines.append(current)
            current = ''
        if char != '\n':
            current += char
    if current:
        lines.append(current)
    overflow = len(lines) > max_lines
    lines = lines[:max_lines]
    if overflow:
        last = lines[-1]
        while last and font.getlength(last+'…') > width:
            last = last[:-1]
        lines[-1] = last+'…'
    return lines


def gripper_change(segment, action_index=0):
    """Compare aligned proposal/executed commands, not measured gripper state."""
    proposed = segment.get('pi05_actions', [])
    executed = segment.get('executed_actions', [])
    if action_index < 0 or action_index >= min(len(proposed), len(executed)):
        return '夹爪：缺少原计划 / 执行指令记录'
    before = bool(proposed[action_index][7] > .5)
    after = bool(executed[action_index][7] > .5)
    label = lambda closed: '闭合' if closed else '打开'
    if before == after:
        return f'夹爪：保持{label(after)}（与 π0.5 一致）'
    return f'夹爪 π0.5→GPT：{label(before)} → {label(after)}'


def correction_lines(segment, action_index=0):
    response = segment['response']
    mode = response['mode']
    if mode == 'student':
        return ['保留 π0.5 原动作目标',
                f'执行前 {segment["executed_steps"]} / 15 步',
                '仅缩短前缀，不修改目标' if segment['prefix_only'] else '整段执行，无 GPT 覆盖']
    if mode not in ('eef', 'edit'):
        return ['没有执行新的动作']
    if mode == 'eef':
        target = response['target']
        current = segment.get('request', {}).get('current_eef', {})
        if 'position' in current:
            delta = np.array(target['position'])-np.array(current['position'])
            position = '相对决策时实测位置'
        else:
            delta = np.array(target['position'])
            position = '绝对目标位置（基座坐标）'
        q1 = np.asarray(target['quaternion_wxyz'], dtype=float)
        if 'quaternion_wxyz' in current:
            q0 = np.asarray(current['quaternion_wxyz'], dtype=float)
            cosine = abs(float(np.dot(q0, q1)/(np.linalg.norm(q0)*np.linalg.norm(q1))))
            angle = math.degrees(2*math.acos(min(1., cosine)))
            rotation = f'姿态目标旋转 {angle:.1f}°'
        else:
            rotation = '姿态变化：缺少参考记录'
    else:
        edit = response['edit']
        delta = np.array(edit['delta_position'])
        position = '相对 π0.5 轨迹的位置偏移'
        rotation = f'姿态目标旋转 {math.degrees(np.linalg.norm(edit["delta_rotation_vector"])):.1f}°'
    return [position,
            f'X {delta[0]*100:+.1f}   Y {delta[1]*100:+.1f}   Z {delta[2]*100:+.1f} cm',
            rotation,
            gripper_change(segment, action_index),
            f'执行 {segment["executed_steps"]} / 请求 {response["steps"]} 步',
            '以上为目标指令，不代表已到达']


class VideoPanel:
    def __init__(self, timeline, font_path, width=1280):
        self.timeline, self.width = timeline, width
        self.fonts = {size: ImageFont.truetype(str(font_path), size) for size in (18, 20, 22, 24, 28, 30)}
        self.cache = {}

    def text(self, draw, xy, text, size=22, color=INK, width=None, lines=1):
        font = self.fonts[size]
        for i, line in enumerate(fitted_lines(text, font, width or self.width-xy[0]-24, lines)):
            draw.text((xy[0], xy[1]+i*(size+9)), line, font=font, fill=color)

    def background(self, segment):
        key = segment['decision'] if segment else None
        if key in self.cache:
            return self.cache[key].copy()
        panel = Image.new('RGB', (self.width, PANEL_HEIGHT), BG)
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((24, 104, 750, 358), radius=16, fill=CARD)
        draw.rounded_rectangle((766, 104, 1256, 358), radius=16, fill=CARD)
        if segment:
            response = segment['response']
            assessment = response.get('assessment', {})
            progress = assessment.get('task_progress', {})
            current = assessment.get('current_subgoal') or progress.get('currently_attempting') or '未记录'
            self.text(draw, (46, 117), '当前状态 · Codex 判断', 18, MUTED)
            self.text(draw, (46, 145), current, 24, width=678, lines=2)
            self.text(draw, (46, 209), '决策原因', 18, MUTED)
            self.text(draw, (46, 240), response.get('reason', '未记录'), 22, width=678, lines=3)
            override = segment['codex_override']
            self.text(draw, (790, 117), 'GPT 修正了什么' if override else '本段执行方式', 18, AMBER if override else CYAN)
            for i, line in enumerate(correction_lines(segment)):
                if override and i == 3:
                    continue  # Draw the aligned gripper comparison for this frame below.
                self.text(draw, (790, 151+i*35), line, 20 if i != 1 else 22,
                          MUTED if i == 5 else INK, width=442)
        else:
            self.text(draw, (46, 145), '没有匹配的已完成执行回执', 24, width=678)
            self.text(draw, (46, 209), '此帧的动作来源无法确认', 22, MUTED, width=678)
        self.cache[key] = panel
        return panel.copy()

    def render(self, rgb, segment, tick, dt):
        panel = self.background(segment)
        draw = ImageDraw.Draw(panel)
        override = bool(segment and segment['codex_override'])
        color = AMBER if override else CYAN
        if override:
            action_index = max(0, tick-1-segment['start_tick'])
            self.text(draw, (790, 256), gripper_change(segment, action_index), 20, AMBER, width=442)
        badge = ('GPT 正在修正' if override else 'π0.5 自主执行') if segment else '动作来源未确认'
        if tick == 0:
            badge = '下一段 · GPT 修正' if override else '下一段 · π0.5 执行'
        final = tick == self.timeline.result.get('step_id')
        if final:
            result = self.timeline.result
            badge = '任务成功' if result.get('success') else '任务超时' if result.get('truncated') else '运行结束 · 未完成'
            color = CYAN if result.get('success') else AMBER
        draw.rounded_rectangle((24, 20, 370, 84), radius=14, fill=color)
        self.text(draw, (43, 31), badge, 28, '#101a28', width=310)
        index = segment['decision']+1 if segment else '—'
        self.text(draw, (394, 22), f'决策 {index}    ·    仿真 {tick*dt:.2f} 秒', 24)
        source = 'GPT 覆盖学生动作' if override else '学生动作目标未修改'
        self.text(draw, (394, 58), source + ('    ·    初始观测，尚未执行' if tick == 0 else ''), 18, MUTED)
        self.text(draw, (1038, 23), '1× 仿真速度', 20, MUTED, width=216)
        end = max(1, self.timeline.result.get('step_id', 0), tick)
        x0, span = 24, self.width-48
        draw.rounded_rectangle((x0, 380, x0+span, 388), radius=4, fill=CARD)
        for part in self.timeline.segments:
            left = x0+span*part['start_tick']/end
            right = x0+span*part['end_tick']/end
            draw.rectangle((left, 380, right, 388), fill=AMBER if part['codex_override'] else CYAN)
        cursor = x0+span*tick/end
        draw.ellipse((cursor-5, 376, cursor+5, 392), fill=INK)
        self.text(draw, (24, 401), f'控制步 {tick} / {end}', 18, MUTED)
        self.text(draw, (275, 401), '青色：学生执行', 18, CYAN)
        self.text(draw, (480, 401), '橙色：GPT 修正', 18, AMBER)
        model = self.timeline.run.get('teacher_model', '模型未记录')
        effort = self.timeline.run.get('teacher_reasoning_effort', '未记录')
        self.text(draw, (860, 401), f'{model} / {effort}', 18, MUTED, width=395)
        top_height = round(rgb.height*self.width/rgb.width)
        top_height += top_height % 2
        canvas = Image.new('RGB', (self.width, top_height+PANEL_HEIGHT), BG)
        canvas.paste(rgb.resize((self.width, top_height)), (0, 0))
        canvas.paste(panel, (0, top_height))
        return canvas
