"""Decode actual teacher payloads and verify isolation from policy/recording."""
import base64
import copy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from .artifact_manifest import is_render_input
from .pi05_server.client import Pi05Client
from .robodojo_server.client import RoboDojoTools
from .skill.image_preview import image_max_edge, prepare_image
from .skill.run import CodexPolicy, content_items
from .test_contract import Sim, Student, invoke


@pytest.mark.parametrize('size,edge,expected', [
    ((640, 480), 480, (480, 360)), ((480, 640), 480, (360, 480)),
    ((1280, 720), 480, (480, 270)), ((320, 240), 480, (320, 240)),
    ((640, 480), 0, (640, 480)), ((640, 480), 640, (640, 480)),
])
def test_preview_dimensions_no_upscale_no_original_overwrite(tmp_path, size, edge, expected):
    path = tmp_path/'cam.png'
    Image.new('RGB', size, (17, 43, 201)).save(path)
    original = path.read_bytes()
    item = dict(camera='cam_high', path=str(path))
    descriptor, payload = prepare_image(item, edge)
    with Image.open(BytesIO(payload)) as decoded:
        assert decoded.size == expected
    assert path.read_bytes() == original
    assert item == dict(camera='cam_high', path=str(path))
    assert descriptor['path'] == str(path)
    if size == expected:
        assert descriptor == item and payload == original
        assert not (tmp_path/'codex_previews').exists()
    else:
        preview = descriptor['codex_preview']
        assert preview['size'] == list(expected) and preview['original_size'] == list(size)
        assert Path(preview['path']).read_bytes() == payload
        assert prepare_image(item, edge) == (descriptor, payload)


def test_mismatched_preview_is_not_overwritten(tmp_path):
    source = tmp_path/'cam.png'
    Image.new('RGB', (640, 480), 'red').save(source)
    item = dict(path=str(source))
    descriptor, _ = prepare_image(item, 480)
    preview = Path(descriptor['codex_preview']['path'])
    preview.write_bytes(b'preserve existing evidence')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        prepare_image(item, 480)
    assert preview.read_bytes() == b'preserve existing evidence'


@pytest.mark.parametrize('value', ['-1', '1.5', 'bad', '', True])
def test_invalid_max_edge(value):
    with pytest.raises(ValueError, match='CODEX_IMAGE_MAX_EDGE'):
        image_max_edge(value)


def test_invalid_environment_fails_before_codex_or_workspace_start(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_IMAGE_MAX_EDGE', '-1')
    with pytest.raises(ValueError, match='CODEX_IMAGE_MAX_EDGE'):
        CodexPolicy(tmp_path/'audit', '/does-not-exist/codex')
    assert not (tmp_path/'audit').exists()


def test_non_image_tool_result_does_not_read_or_resize_paths():
    packet = dict(images=[dict(path='/does-not-exist/image.png')], value=17)
    items = content_items(packet, images=False)
    assert items == [dict(type='inputText', text=json.dumps(packet, separators=(',', ':')))]


def test_codex_attachment_is_smaller_but_policy_and_archives_are_pixel_exact(tmp_path):
    rng = np.random.default_rng(3)
    rgb = {key: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
           for key in RoboDojoTools.camera_keys}
    captured = {}

    class CameraSim(Sim):
        def request(self, op, **arguments):
            result = super().request(op, **arguments)
            if op == 'teacher_observation':
                result.update({key: value.copy() for key, value in rgb.items()})
            return result

    class Policy:
        def __init__(self, *args):
            self._ws = SimpleNamespace(ping_timeout=20, close=lambda: None)

        def get_server_metadata(self):
            return dict(Student.metadata, backend='OpenPI/JAX', inferences=0)

        def infer(self, observation):
            captured.update(observation)
            return dict(actions=np.zeros((50, 14), np.float32),
                policy_identity=dict(checkpoint_sha256='a'*64, inference_index=0))

    student = Pi05Client(1, '/test/checkpoint', client_cls=Policy)
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', student, rpc_factory=CameraSim)
    observation = invoke(tools)
    untouched_packet = copy.deepcopy(observation)
    originals = {image['path']: Path(image['path']).read_bytes() for image in observation['images']}
    items = content_items(observation, images=True)
    assert observation == untouched_packet
    visible = json.loads(items[0]['text'])
    assert len(items) == 4
    for descriptor, attachment in zip(visible['images'], items[1:]):
        payload = base64.b64decode(attachment['imageUrl'].split(',', 1)[1])
        with Image.open(BytesIO(payload)) as image:
            assert image.size == (480, 360)
        preview = descriptor['codex_preview']
        assert Path(preview['path']).read_bytes() == payload
        relative = Path(preview['path']).relative_to(tmp_path).as_posix()
        assert is_render_input('controller/'+relative)
        assert Path(descriptor['path']).read_bytes() == originals[descriptor['path']]
        np.testing.assert_array_equal(tools.obs[descriptor['camera']], rgb[descriptor['camera']])
    invoke(tools)  # Use the real Pi05Client transformation with a fake policy transport.
    for key, original in rgb.items():
        assert captured['images'][key].shape == (3, 480, 640)
        np.testing.assert_array_equal(captured['images'][key], np.transpose(original, (2, 0, 1)))
        with np.load(observation['arrays_path']) as data:
            np.testing.assert_array_equal(data[key], original)
        with np.load(tmp_path/'proposals/000/actions.npz') as data:
            np.testing.assert_array_equal(data[key], original)
    student.close()
