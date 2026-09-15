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
TERMINAL_LABEL_VERSION = 'native_terminal_labels_v2'


def terminal_badge(result):
    """Distinguish recorded native termination from a controller interruption."""
    if (not result.get('complete') and result.get('evaluation_failure_reason') ==
            'simulator_rpc_idle_timeout_900s'):
        return 'Failed · RPC idle 900s'
    if not result.get('complete'):
        return 'Run ended · incomplete'
    if result.get('success'):
        return 'Task success'
    if result.get('truncated'):
        return 'Task timeout'
    if result.get('terminated'):
        return 'Task failed · native'
    return 'Run ended · incomplete'


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
    if segment.get('evaluation_method') == 'gpt_only' and 0 <= action_index < len(executed):
        return 'GPT opening ' + '  '.join(f'{arm} {executed[action_index][i]:.2f}'
                                         for arm, i in (('L', 6), ('R', 13)))
    if action_index < 0 or action_index >= min(len(proposed), len(executed)):
        return 'Gripper: missing proposal/execution record'
    return 'Gripper opening ' + '  '.join(
        f'{arm} {proposed[action_index][i]:.2f}→{executed[action_index][i]:.2f}'
        for arm, i in (('L', 6), ('R', 13)))


def correction_lines(segment, action_index=0):
    response = segment['response']
    mode = response['mode']
    if mode == 'joint':
        return ['GPT-authored absolute joint actions', gripper_change(segment, action_index),
                f'Executed {segment["executed_steps"]} / requested {response["steps"]} steps',
                'No pi05 inference or proposal']
    if mode == 'student':
        return ['Keep the original π0.5 action target',
                f'Execute {segment["executed_steps"]} / {len(segment.get("pi05_actions", [])) or 15} steps',
                'Shortened prefix; target unchanged' if segment['prefix_only'] else 'Full chunk; no GPT override']
    if mode not in ('eef', 'edit'):
        return ['No new action was executed']
    rows = []
    for arm, label in (('left', 'Left'), ('right', 'Right')):
        delta = (np.array(response['target'][arm]['position'])-
                 np.array(segment['request']['current_eef'][arm]['position'])) if mode == 'eef' else np.array(response['edit'][arm]['delta_position'])
        rows.append(f'{label} ΔXYZ {delta[0]*100:+.1f} {delta[1]*100:+.1f} {delta[2]*100:+.1f} cm')
    return [*rows, 'Gripper opening: 0 closed · 1 open', gripper_change(segment, action_index),
            f'Executed {segment["executed_steps"]} / requested {response["steps"]} steps',
            'Targets are commands, not proof of arrival']


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
            current = assessment.get('current_subgoal') or progress.get('currently_attempting') or (
                'Direct observation-to-action' if segment.get('evaluation_method') == 'gpt_only' else 'Not recorded')
            self.text(draw, (46, 117), 'Current state · Codex assessment', 18, MUTED)
            self.text(draw, (46, 145), current, 24, width=678, lines=2)
            self.text(draw, (46, 209), 'Decision rationale', 18, MUTED)
            self.text(draw, (46, 240), response.get('reason', 'Not recorded'), 22, width=678, lines=3)
            override = segment['codex_override']
            self.text(draw, (790, 117), 'GPT correction' if override else 'Execution mode', 18, AMBER if override else CYAN)
            for i, line in enumerate(correction_lines(segment)):
                if override and i == 3:
                    continue  # Draw the aligned gripper comparison for this frame below.
                self.text(draw, (790, 151+i*35), line, 20 if i != 1 else 22,
                          MUTED if i == 5 else INK, width=442)
        else:
            self.text(draw, (46, 145), 'No matching completed execution receipt', 24, width=678)
            self.text(draw, (46, 209), 'Action source cannot be confirmed for this frame', 22, MUTED, width=678)
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
        badge = ('GPT correction' if override else 'π0.5 autonomous') if segment else 'Unconfirmed action source'
        if tick == 0:
            badge = 'Next chunk · GPT correction' if override else 'Next chunk · π0.5'
        if segment and segment.get('evaluation_method') == 'gpt_only':
            badge = 'GPT-only · direct ' + segment['response']['mode'] + ' action'
        final = tick == self.timeline.result.get('step_id')
        if final:
            result = self.timeline.result
            badge = terminal_badge(result)
            color = CYAN if result.get('success') else AMBER
        draw.rounded_rectangle((24, 20, 370, 84), radius=14, fill=color)
        self.text(draw, (43, 31), badge, 28, '#101a28', width=310)
        index = segment['decision']+1 if segment else '—'
        self.text(draw, (394, 22), f'Decision {index}    ·    Simulation {tick*dt:.2f} s', 24)
        source = 'GPT overrides student action' if override else 'Student target unchanged'
        if getattr(self.timeline, 'adjudication', None):
            source = 'Evaluation failure: RPC idle 900s · no retry · native score N/A'
        self.text(draw, (394, 58), source + ('    ·    Initial observation' if tick == 0 else ''), 18, MUTED)
        self.text(draw, (1038, 23), '1× simulation', 20, MUTED, width=216)
        end = max(1, self.timeline.result.get('step_id', 0), tick)
        x0, span = 24, self.width-48
        draw.rounded_rectangle((x0, 380, x0+span, 388), radius=4, fill=CARD)
        for part in self.timeline.segments:
            left = x0+span*part['start_tick']/end
            right = x0+span*part['end_tick']/end
            draw.rectangle((left, 380, right, 388), fill=AMBER if part['codex_override'] else CYAN)
        cursor = x0+span*tick/end
        draw.ellipse((cursor-5, 376, cursor+5, 392), fill=INK)
        self.text(draw, (24, 401), f'Control {tick} / {end}', 18, MUTED)
        direct = self.timeline.run.get('evaluation_method') == 'gpt_only'
        self.text(draw, (275, 401), 'Cyan: GPT-only' if direct else 'Cyan: student', 18, CYAN)
        self.text(draw, (480, 401), 'No pi05' if direct else 'Amber: GPT correction', 18, AMBER)
        model = self.timeline.run.get('teacher_model', 'Model not recorded')
        effort = self.timeline.run.get('teacher_reasoning_effort', 'Not recorded')
        self.text(draw, (860, 401), f'{model} / {effort}', 18, MUTED, width=395)
        top_height = round(rgb.height*self.width/rgb.width)
        top_height += top_height % 2
        canvas = Image.new('RGB', (self.width, top_height+PANEL_HEIGHT), BG)
        canvas.paste(rgb.resize((self.width, top_height)), (0, 0))
        canvas.paste(panel, (0, top_height))
        return canvas
