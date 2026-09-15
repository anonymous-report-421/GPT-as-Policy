"""Offline projection of recorded EEF commands; never connects to a simulator.

The old recordings contain measured link6 poses, not camera matrices. Reconstruct
the latter from each run's resolved camera config and the robot's fixed camera
mount, and verify that mount against the actual USD asset before drawing arrows.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import struct
import xml.etree.ElementTree as ET

import numpy as np
from PIL import ImageDraw
from scipy.spatial.transform import Rotation

from ..io import sha256
from .action_edit_kinematics import transform
from .kinematics import ArmFK


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
USD_TO_CV = np.diag([1., -1., -1., 1.])
ARROW_COLORS = {'left': '#66e8fa', 'right': '#ffcc70'}
PROJECTION_VERSION = 'recorded_command_projection_v1'


def local_pose(camera):
    result = np.eye(4)
    result[:3, 3] = camera['pos']
    orientation = camera['ori']
    # RoboDojo utils/rotations.py uses intrinsic XYZ for camera configuration.
    result[:3, :3] = (Rotation.from_euler('XYZ', orientation, degrees=True).as_matrix()
        if len(orientation) == 3 else
        Rotation.from_quat(np.asarray(orientation)[[1, 2, 3, 0]]).as_matrix())
    return result


def pinhole_project(point, camera_to_environment, intrinsic, near=0.005):
    value = USD_TO_CV @ np.linalg.inv(camera_to_environment) @ np.r_[point, 1.]
    if not np.isfinite(value).all() or value[2] <= near:
        return None
    pixel = np.asarray(intrinsic) @ value[:3]
    return pixel[:2] / pixel[2]


def finger_visual_anchor(urdf):
    """Midpoint of distal finger collision bounds, not an assumed TCP.

    The actual X5A finger meshes extend along local +X. Opposed +/-Y
    prismatic joints leave the midpoint invariant as the gripper opens. Do not
    infer an axis from the unrelated scalar gripper_bias setting.
    """
    root=ET.parse(urdf).getroot()
    tips=[]; evidence=[]
    for name,sign in (('joint7',1),('joint8',-1)):
        joint=root.find(f"joint[@name='{name}']")
        if joint.get('type')!='prismatic' or joint.find('parent').get('link')!='link6':
            raise ValueError('Unsupported finger attachment')
        axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
        if not np.allclose(axis,[0,sign,0]):
            raise ValueError('Cannot prove gripper midpoint invariance')
        origin=joint.find('origin'); offset=np.fromstring(origin.get('xyz'),sep=' ')
        if not np.allclose(np.fromstring(origin.get('rpy'),sep=' '),0):
            raise ValueError('Unsupported rotated finger geometry')
        link=root.find(f"link[@name='{joint.find('child').get('link')}']")
        collision=link.find('collision')
        for key in ('xyz','rpy'):
            if not np.allclose(np.fromstring(collision.find('origin').get(key),sep=' '),0):
                raise ValueError('Unsupported transformed finger collision')
        path=Path(urdf).parent/collision.find('geometry/mesh').get('filename')
        content=path.read_bytes();count=struct.unpack_from('<I',content,80)[0]
        if len(content)!=84+50*count: raise ValueError('Expected binary STL')
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
        vertices=np.frombuffer(content,offset=84,count=count,dtype=dtype)['vertices'].reshape(-1,3)
        lo,hi=vertices.min(0),vertices.max(0)
        if np.argmax(hi-lo)!=0 or hi[0]<=abs(lo[0]):
            raise ValueError('Finger mesh does not establish a distal +X extension')
        tip=(lo+hi)/2;tip[0]=hi[0]
        tips.append(offset+tip)
        evidence.append(dict(mesh=str(path),sha256=sha256(path),bounds_min=lo.tolist(),
            bounds_max=hi.tolist(),joint_origin=offset.tolist(),distal_bound_center=(offset+tip).tolist()))
    return np.mean(tips,axis=0).tolist(),evidence


def calibration_for_run(run_root, source_root, usd_python):
    """Return an inspectable reconstruction, refusing unknown mounts/noise.

    ``usd_python`` only opens the existing USD with pxr. No SimulationApp,
    physics, policy, network connection, or environment reset is involved.
    """
    run_root, source_root = Path(run_root), Path(source_root)
    config_path = run_root/'sim/resolved_config.json'
    config = json.loads(config_path.read_text())
    camera_config = config['camera']
    if camera_config.get('random'):
        raise ValueError('Camera randomization was not recorded per frame; refuse inferred arrows')
    robot_path = source_root/'Assets/Robots/x5'
    urdf, usd = robot_path/'X5A.urdf', robot_path/'ARX.usd'
    fixed = ArmFK(urdf, [], base='link6', tip='camera').matrix([])
    # Validate against the actual asset used by robot_config/x5.py, not just a
    # planning URDF that might disagree with the rendered wrist-camera frame.
    check_script = '''import json,sys
from pxr import Usd,UsdGeom
s=Usd.Stage.Open(sys.argv[1],Usd.Stage.LoadNone)
c=UsdGeom.XformCache()
tip=s.GetPrimAtPath('/X5A/link6'); camera=s.GetPrimAtPath('/X5A/camera')
if not tip or not camera: raise ValueError('Expected actual X5A link6/camera prims')
# Gf matrices are row-vector matrices: transpose the relative transform.
m=c.GetLocalToWorldTransform(camera)*c.GetLocalToWorldTransform(tip).GetInverse()
print(json.dumps([[float(m[j][i]) for j in range(4)] for i in range(4)]))
'''
    verified = subprocess.run([str(usd_python), '-c', check_script, str(usd)],
        check=True, text=True, capture_output=True)
    usd_fixed = np.asarray(json.loads(verified.stdout), float)
    mount_error = float(np.max(np.abs(fixed-usd_fixed)))
    if mount_error > 2e-6:
        raise ValueError(f'USD/URDF camera mount mismatch: {mount_error}')
    template = source_root/'env_cfg/camera/template.py'
    constants = {}
    for node in ast.parse(template.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    if constants['PINHOLE'] != {'position': (0., 0., 0.), 'orientation': (1., 0., 0., 0.)}:
        raise ValueError('Unsupported nonidentity optical camera transform')
    import yaml
    robot_config_path = robot_path/'robot_config.yml'
    robot_config = yaml.safe_load(robot_config_path.read_text())
    if robot_config['ee_link'] != 'link6' or robot_config['delta_matrix'] != [[1,0,0],[0,1,0],[0,0,1]]:
        raise ValueError('Unrecognized EEF frame')
    anchor,anchor_evidence=finger_visual_anchor(urdf)
    cameras = {}
    for index, name in enumerate(CAMERAS):
        camera = camera_config[name]['camera']
        expected = None if index == 0 else f'robot{index-1}/camera'
        if camera.get('mount_link') != expected or camera['mesh'] != 'pinhole':
            raise ValueError(f'Unsupported camera mount: {name}')
        kind = {'Gemini_345Lg': 'GEMINI_345LG', 'd435': 'D435'}[camera['type']]
        args = constants[kind]
        width, height = args['resolution']
        intrinsic = [[width*args['focal_length']/args['horizontal_aperture'], 0, width/2],
                     [0, height*args['focal_length']/args['vertical_aperture'], height/2], [0,0,1]]
        cameras[name] = dict(intrinsic=intrinsic, resolution=[width,height],
            local_pose=local_pose(camera).tolist(), mount_arm=None if index == 0 else ('left','right')[index-1],
            near=args['clipping_range'][0])
    provenance = [config_path, template, urdf, usd, robot_config_path,
        source_root/'env/camera_manager/camera_manager.py', source_root/'utils/rotations.py',
        source_root/'env/robot_manager/robot_config/x5.py']
    # Prove the config/pose code is the recorded RoboDojo revision, rather than
    # silently using a newer implementation to reconstruct old recordings.
    recorded_head = (run_root/'robodojo_head.txt').read_text().strip()
    for path in provenance[-3:] + [template]:
        tracked = subprocess.run(['git', '-C', str(source_root), 'show',
            f'{recorded_head}:{path.relative_to(source_root)}'], check=True, capture_output=True).stdout
        if tracked != path.read_bytes():
            raise ValueError(f'Calibration code differs from recorded RoboDojo revision: {path}')
    return dict(schema=PROJECTION_VERSION, cameras=cameras, camera_mount=fixed.tolist(),
        tool_anchor_link6=anchor,anchor_geometry=anchor_evidence,
        anchor_description='Midpoint of distal finger collision-mesh bounds in link6; visual marker, not exact TCP/contact point or object ground truth',
        usd_urdf_mount_max_abs_error=mount_error, recorded_robodojo_revision=recorded_head,
        camera_matrix_source='Reconstructed from resolved config and per-frame measured EEF; not directly recorded extrinsics',
        optical_convention='USD +Y up/-Z forward -> CV +Y down/+Z forward; fx/fy retain separate apertures',
        input_sha256={**{str(path): sha256(path) for path in provenance},
            **{item['mesh']:item['sha256'] for item in anchor_evidence}})


class CommandArrows:
    """Draw direction glyphs only during actually executed hybrid corrections.

    The direction is the EEF translation commanded at the chunk boundary (eef),
    or the recorded offset to the student waypoint (edit). The anchor follows
    the measured gripper work-point. Glyph length is capped for legibility and
    never encodes exact distance or proof of physical arrival.
    """
    def __init__(self, timeline, calibration):
        self.timeline, self.calibration = timeline, calibration
        reset = json.loads((timeline.controller.parent/'sim/reset.json').read_text())
        self.observations = timeline.controller.parent/'sim'/reset['episode_id']/'observations'
        self.records, self.drawn_frames = [], set()

    def draw(self, image, segment, tick):
        if tick == 0 or not segment or not segment['codex_override']:
            return image
        path = self.observations/f'{tick:06d}.npz'
        with np.load(path, allow_pickle=False) as data:
            positions = data['eef_positions'].copy()
            rotations = data['eef_quaternions_wxyz'].copy()
        poses = {arm: transform(positions[i], rotations[i]) for i,arm in enumerate(('left','right'))}
        frame = image.copy()
        draw = ImageDraw.Draw(frame)
        response = segment['response']
        for camera_index, name in enumerate(CAMERAS):
            camera = self.calibration['cameras'][name]
            matrix = np.asarray(camera['local_pose'])
            arm_mount = camera['mount_arm']
            if arm_mount:
                matrix = poses[arm_mount] @ np.asarray(self.calibration['camera_mount']) @ matrix
            tile_width = image.width/3
            for arm in ('left', 'right'):
                # Each wrist view shows only that arm's command to avoid an
                # off-camera arm's projected glyph obscuring the local action.
                if arm_mount and arm != arm_mount:
                    continue
                if response['mode'] == 'eef':
                    delta = np.asarray(response['target'][arm]['position']) - np.asarray(segment['request']['current_eef'][arm]['position'])
                else:
                    delta = np.asarray(response['edit'][arm]['delta_position'])
                record = dict(tick=tick, decision=segment['decision'], camera=name, arm=arm,
                    mode=response['mode'], commanded_translation_m=delta.tolist(), drawn=False,
                    vector_definition=('requested EEF target minus measured current EEF at chunk boundary; not residual versus pi05'
                        if response['mode']=='eef' else 'recorded additive translation offset to student FK waypoint'),
                    anchor_definition='Measured pose of collision-mesh fingertip midpoint; not exact TCP')
                self.records.append(record)
                if np.linalg.norm(delta) < .001:
                    record['reason'] = 'No >=1mm translation command (rotation/grip-only does not imply translation)'
                    continue
                start_world = (poses[arm] @ np.r_[self.calibration['tool_anchor_link6'],1])[:3]
                start = pinhole_project(start_world, matrix, camera['intrinsic'], camera['near'])
                end = pinhole_project(start_world+delta, matrix, camera['intrinsic'], camera['near'])
                if start is None or end is None:
                    record['reason'] = 'Point behind camera or within near clip'
                    continue
                original_width, original_height = camera['resolution']
                scale = np.array([tile_width/original_width,image.height/original_height])
                start, end = start*scale, end*scale
                record.update(start_pixel=start.tolist(), target_pixel=end.tolist())
                if not (8 < start[0] < tile_width-8 and 8 < start[1] < image.height-8):
                    record['reason'] = 'Projected anchor outside this camera image'
                    continue
                vector = end-start
                distance = np.linalg.norm(vector)
                if distance < 2:
                    record['reason'] = 'Projected translation <2px; no enlarged fictitious direction'
                    continue
                end = start + vector*min(1,52/distance)
                # Clip a long arrow to the actual tile, preserving its direction.
                ratio = 1.
                for axis,maximum in ((0,tile_width-8),(1,image.height-8)):
                    if end[axis] > maximum: ratio = min(ratio,(maximum-start[axis])/(end[axis]-start[axis]))
                    if end[axis] < 8: ratio = min(ratio,(8-start[axis])/(end[axis]-start[axis]))
                end = start + (end-start)*ratio
                if np.linalg.norm(end-start) < 2:
                    record['reason'] = 'Arrow falls outside camera image'
                    continue
                start += [camera_index*tile_width,0]
                end += [camera_index*tile_width,0]
                color = ARROW_COLORS[arm]
                draw.line([tuple(start),tuple(end)],fill='#0c1422',width=6)
                draw.line([tuple(start),tuple(end)],fill=color,width=3)
                direction=(end-start)/np.linalg.norm(end-start)
                perpendicular=np.array([-direction[1],direction[0]])
                tip_length=min(8,np.linalg.norm(end-start)*.6)
                vertices=[end,end-direction*tip_length+perpendicular*tip_length*.5,
                          end-direction*tip_length-perpendicular*tip_length*.5]
                draw.polygon([tuple(point) for point in vertices],fill=color)
                draw.ellipse((start[0]-2,start[1]-2,start[0]+2,start[1]+2),fill=color)
                record.update(drawn=True, arrow_start_canvas=start.tolist(),arrow_end_canvas=end.tolist(),
                    reason='Recorded command direction; bounded glyph, not distance or measured motion')
                self.drawn_frames.add(tick)
        return frame
