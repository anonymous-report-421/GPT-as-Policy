"""Opt-in, compact report-video layout. Replays only recorded evidence."""
from PIL import Image, ImageDraw, ImageFont

from .video_panel import BG, CARD, INK, MUTED, AMBER, CYAN, VideoPanel, correction_lines, terminal_badge, fitted_lines


LAYOUT_VERSION = 'paper_prompt_arrows_sample_v1'
PROMPT_HEIGHT = 76
PANEL_HEIGHT = 320


class PaperVideoPanel(VideoPanel):
    def __init__(self, timeline, font_path, arrows=None, width=1280):
        super().__init__(timeline, font_path, width)
        self.fonts.update({size: ImageFont.truetype(str(font_path),size) for size in (16,17)})
        self.arrows = arrows
        self.prompt = str(timeline.run.get('instruction') or timeline.run.get('task') or 'Task instruction not recorded')
        self.line_cache = {}

    def text(self, draw, xy, text, size=22, color=INK, width=None, lines=1):
        """Cache only deterministic wrapping; leave approved pixels unchanged."""
        available=width or self.width-xy[0]-24
        key=(str(text),size,available,lines)
        if key not in self.line_cache:
            self.line_cache[key]=fitted_lines(text,self.fonts[size],available,lines)
        for i,line in enumerate(self.line_cache[key]):
            draw.text((xy[0],xy[1]+i*(size+9)),line,font=self.fonts[size],fill=color)

    def render(self, rgb, segment, tick, dt):
        observation_height = round(rgb.height*self.width/rgb.width)
        observation_height += observation_height%2
        canvas = Image.new('RGB',(self.width,PROMPT_HEIGHT+observation_height+PANEL_HEIGHT),BG)
        draw=ImageDraw.Draw(canvas)
        self.text(draw,(24,8),'TASK',16,CYAN,width=70)
        self.text(draw,(95,8),self.prompt,20,INK,width=self.width-119,lines=2)
        observation=rgb.resize((self.width,observation_height),Image.Resampling.LANCZOS)
        if self.arrows:
            observation=self.arrows.draw(observation,segment,tick)
        overlay=ImageDraw.Draw(observation)
        for i,name in enumerate(('HEAD VIEW','LEFT WRIST','RIGHT WRIST')):
            x=round(self.width*i/3)
            overlay.rounded_rectangle((x+8,7,x+147,32),radius=5,fill=BG)
            self.text(overlay,(x+17,8),name,16,INK,width=132)
        canvas.paste(observation,(0,PROMPT_HEIGHT))
        base=PROMPT_HEIGHT+observation_height
        override=bool(segment and segment['codex_override'])
        direct=self.timeline.run.get('evaluation_method')=='gpt_only'
        badge='GPT 6 Astra correction' if override else 'π0.5 action'
        if direct: badge='GPT 6 Astra Direct'
        if not segment: badge='Unconfirmed action source'
        if tick==self.timeline.result.get('step_id'): badge=terminal_badge(self.timeline.result)
        color=AMBER if override or (tick==self.timeline.result.get('step_id') and not self.timeline.result.get('success')) else CYAN
        draw.rounded_rectangle((24,base+12,350,base+50),radius=9,fill=color)
        self.text(draw,(37,base+16),badge,20,BG,width=300)
        decision=segment['decision']+1 if segment else '—'
        limit=self.timeline.run.get('max_episode_steps',self.timeline.result.get('step_id','—'))
        self.text(draw,(374,base+16),f'Decision {decision}   ·   Control {tick} / {limit}   ·   {tick*dt:.2f} s',20,INK,width=700)
        self.text(draw,(1110,base+19),'1× simulation',16,MUTED,width=150)
        draw.rounded_rectangle((24,base+62,766,base+252),radius=12,fill=CARD)
        draw.rounded_rectangle((782,base+62,1256,base+252),radius=12,fill=CARD)
        if segment:
            response=segment['response']; assessment=response.get('assessment',{})
            current=assessment.get('current_subgoal') or assessment.get('task_progress',{}).get('currently_attempting') or 'Not recorded'
            self.text(draw,(42,base+72),'CURRENT STATE · MODEL ASSESSMENT',16,MUTED,width=700)
            self.text(draw,(42,base+99),current,20,INK,width=700,lines=2)
            self.text(draw,(42,base+150),'DECISION RATIONALE',16,MUTED,width=700)
            self.text(draw,(42,base+175),response.get('reason','Not recorded'),18,INK,width=700,lines=3)
            self.text(draw,(800,base+72),'COMMAND DETAILS',16,AMBER if override else CYAN,width=440)
            action_index=max(0,tick-1-segment['start_tick'])
            rows=correction_lines(segment,action_index)
            # Same recorded numeric information, shorter compact layout.
            for i,line in enumerate(rows[:6]):
                line=line.replace('GPT','GPT 6 Astra')
                self.text(draw,(800,base+101+24*i),line,17,MUTED if i==5 else INK,width=434)
        else:
            self.text(draw,(42,base+99),'No matching execution receipt',20,INK,width=700)
        end=max(1,self.timeline.result.get('step_id',0),tick)
        x0,span=24,self.width-48
        draw.rounded_rectangle((x0,base+272,x0+span,base+279),radius=3,fill=CARD)
        for part in self.timeline.segments:
            draw.rectangle((x0+span*part['start_tick']/end,base+272,x0+span*part['end_tick']/end,base+279),fill=AMBER if part['codex_override'] else CYAN)
        cursor=x0+span*tick/end
        draw.ellipse((cursor-4,base+269,cursor+4,base+282),fill=INK)
        legend='Arrows: recorded EEF translation direction · L cyan / R amber · not proof of arrival'
        if direct: legend='GPT 6 Astra Direct · no policy proposal'
        self.text(draw,(24,base+290),legend,16,MUTED,width=1000)
        self.text(draw,(1080,base+290),'GPT 6 Astra',16,MUTED,width=175)
        return canvas
