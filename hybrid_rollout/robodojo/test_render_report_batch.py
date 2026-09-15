"""Model-free production handoff and isolation checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from . import render_report_batch as batch
from .io import write_json,sha256
from .robodojo_server.debug_recorder import load_font
from .robodojo_server.video_panel import VideoPanel
from .robodojo_server.video_panel_paper import PaperVideoPanel


class ProductionVideoTests(unittest.TestCase):
    def test_frozen_global_selection_and_annotations_are_rechecked(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);selection=root/'selection.json';annotation=root/'00000002.json'
            write_json(selection,{'selected':['clip0']});write_json(annotation,{'start':0,'end':10})
            plan=dict(input_sha256={str(p):sha256(p) for p in (selection,annotation)})
            self.assertEqual(batch.verify_global_inputs(plan),2)
            write_json(annotation,{'start':1,'end':10})
            with self.assertRaisesRegex(ValueError,'global input changed'):batch.verify_global_inputs(plan)
        with self.assertRaisesRegex(ValueError,'no global source hashes'):batch.verify_global_inputs({})

    def test_public_media_paths_cannot_escape(self):
        self.assertEqual(batch.relative_media('media/clips/example.mp4'),'media/clips/example.mp4')
        for value in ('/mnt/private/key','../secret','media/../../secret','not-media/a.mp4'):
            with self.assertRaises(ValueError):batch.relative_media(value)

    def test_wrapping_cache_preserves_approved_pixels(self):
        _,font=load_font(22)
        timeline=types.SimpleNamespace(run=dict(instruction='Place the blocks.',max_episode_steps=700),
            result=dict(step_id=2,complete=False),segments=[])
        cached=PaperVideoPanel(timeline,font);original=PaperVideoPanel(timeline,font)
        original.text=types.MethodType(VideoPanel.text,original)
        rgb=Image.new('RGB',(1920,360))
        for tick in (0,1,2):
            np.testing.assert_array_equal(np.asarray(cached.render(rgb,None,tick,.04)),
                np.asarray(original.render(rgb,None,tick,.04)))

    def test_export_refuses_unreceipted_existing_media(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'media/x.mp4';target.parent.mkdir();target.write_bytes(b'preserve')
            row=dict(id='x',video='media/x.mp4',poster='media/x.jpg')
            with self.assertRaises(FileExistsError):batch.export_one(root,row,{'video':'unused','sha256':'unused'})
            self.assertEqual(target.read_bytes(),b'preserve')

    def test_export_receipt_mismatch_is_not_silently_reused(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'media-receipts').mkdir()
            write_json(root/'media-receipts/x.json',dict(source_render_sha256='old',files=[]))
            with self.assertRaises(ValueError):batch.export_one(root,dict(id='x',video='media/x.mp4',poster='media/x.jpg'),
                dict(video='unused',sha256='new'))

    def test_direct_cannot_receive_hybrid_arrows(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'debug_rollout.mp4').write_bytes(b'fake')
            outcome=dict(step_id=1,complete=True,success=False,terminated=False,truncated=True)
            manifest=dict(status='completed',layout='paper_prompt_arrows_sample_v1',task_prompt='task',
                video_sha256=sha256(root/'debug_rollout.mp4'),episode_result=outcome,arrow_frames=1)
            write_json(root/'manifest.json',manifest)
            write_json(root/'decisions.json',[dict(decision=0,codex_override=True,start_tick=0,end_tick=1)])
            write_json(root/'projected_commands.json',[dict(tick=1,decision=0,drawn=True,commanded_translation_m=[.03,0,0])])
            run=dict(frames=2,fps=25,instruction='task',original_result=outcome,adjudicated_idle=False,
                method='gpt',original_videos=[],inputs={})
            with mock.patch.object(batch,'probe',return_value=dict(frames=2,fps=25,width=1280,height=636,duration=.08)):
                with self.assertRaisesRegex(ValueError,'executed hybrid correction'):batch.validate_render(root,run)

    def test_idle_failure_badge_is_preserved(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'debug_rollout.mp4').write_bytes(b'fake')
            outcome=dict(step_id=1,complete=False,success=False,terminated=False,truncated=False,
                evaluation_failure_reason='simulator_rpc_idle_timeout_900s')
            write_json(root/'manifest.json',dict(status='completed',layout='paper_prompt_arrows_sample_v1',task_prompt='task',
                video_sha256=sha256(root/'debug_rollout.mp4'),episode_result=outcome,arrow_frames=0))
            write_json(root/'decisions.json',[]);write_json(root/'projected_commands.json',[])
            run=dict(frames=2,fps=25,instruction='task',original_result=outcome,adjudicated_idle=True,
                method='gpt',original_videos=[],inputs={})
            with mock.patch.object(batch,'probe',return_value=dict(frames=2,fps=25,width=1280,height=636,duration=.08)):
                result=batch.validate_render(root,run)
            self.assertEqual(result['terminal_badge'],'Failed · RPC idle 900s')
            self.assertEqual(result['arrow_frames'],0)


if __name__=='__main__':unittest.main()
