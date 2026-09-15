"""CPU-only projection / timeline / compact-layout checks; no rollout calls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from .robodojo_server.video_projection import CommandArrows, local_pose, pinhole_project
from .robodojo_server.video_panel_paper import PaperVideoPanel
from .robodojo_server.debug_recorder import load_font


class ProjectionTests(unittest.TestCase):
    def test_usd_axes_and_nonsquare_intrinsics(self):
        k=[[100,0,320],[0,200,240],[0,0,1]]
        np.testing.assert_allclose(pinhole_project([.1,.1,-1],np.eye(4),k),[330,220])
        self.assertIsNone(pinhole_project([0,0,1],np.eye(4),k))
        self.assertIsNone(pinhole_project([0,0,-.001],np.eye(4),k))

    def test_configuration_uses_intrinsic_xyz(self):
        a=local_pose(dict(pos=[1,2,3],ori=[0,-80,-90]))
        np.testing.assert_allclose(a[:3,3],[1,2,3])
        # Composition uses Rx @ Ry @ Rz, not the reverse/extrinsic order.
        from scipy.spatial.transform import Rotation
        want=Rotation.from_euler('Y',-80,degrees=True).as_matrix() @ Rotation.from_euler('Z',-90,degrees=True).as_matrix()
        np.testing.assert_allclose(a[:3,:3],want,atol=1e-12)

    def test_initial_student_and_direct_never_draw(self):
        arrows=object.__new__(CommandArrows)
        frame=Image.new('RGB',(1280,240))
        self.assertIs(arrows.draw(frame,dict(codex_override=True),0),frame)
        self.assertIs(arrows.draw(frame,dict(codex_override=False),1),frame)
        self.assertIs(arrows.draw(frame,None,1),frame)

    def test_translation_and_zero_translation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'sim/e/observations').mkdir(parents=True)
            (root/'sim/reset.json').write_text(json.dumps(dict(episode_id='e')))
            np.savez(root/'sim/e/observations/000001.npz',eef_positions=[[0,0,-1],[.2,0,-1]],
                eef_quaternions_wxyz=[[1,0,0,0],[1,0,0,0]])
            timeline=SimpleNamespace(controller=root/'controller')
            camera=dict(local_pose=np.eye(4).tolist(),intrinsic=[[400,0,320],[0,400,240],[0,0,1]],
                resolution=[640,480],near=.005,mount_arm=None)
            calibration=dict(cameras={n:camera for n in ('cam_head','cam_left_wrist','cam_right_wrist')},
                tool_anchor_link6=[0,0,0])
            arrows=CommandArrows(timeline,calibration)
            seg=dict(codex_override=True,decision=2,request=dict(current_eef={
                'left':dict(position=[0,0,-1]),'right':dict(position=[.2,0,-1])}),
                response=dict(mode='eef',target={'left':dict(position=[.1,0,-1]),'right':dict(position=[.2,0,-1])}))
            before=Image.new('RGB',(1280,240))
            after=arrows.draw(before,seg,1)
            self.assertTrue(np.asarray(after).any())
            drawn=[r for r in arrows.records if r['drawn']]
            self.assertEqual(len(drawn),3)
            self.assertTrue(all(r['arm']=='left' for r in drawn))
            self.assertTrue(all(r['arrow_end_canvas'][0]>r['arrow_start_canvas'][0] for r in drawn))

    def test_paper_frame_size_and_legacy_separation(self):
        _,font=load_font(22)
        timeline=SimpleNamespace(run=dict(instruction='Place the blocks.',max_episode_steps=700),
            result=dict(step_id=0,complete=False),segments=[])
        panel=PaperVideoPanel(timeline,font)
        frame=panel.render(Image.new('RGB',(1920,360)),None,0,.04)
        self.assertEqual(frame.size,(1280,636))
        from .robodojo_server.video_panel import PANEL_HEIGHT
        self.assertEqual(PANEL_HEIGHT,440)


if __name__=='__main__': unittest.main()
