"""Blocking OpenPI inference. Arrays stay in files, not in model tool arguments."""
from pathlib import Path
import os
import time
import uuid
import numpy as np

from ..io import require


class Pi05Client:
    def __init__(self, port, checkpoint, client_cls=None):
        if client_cls is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            client_cls = WebsocketClientPolicy
        # Only this loopback connection bypasses proxies. Restore the parent's
        # network configuration before starting Codex's OpenAI connection.
        proxy_keys = ('ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'http_proxy', 'https_proxy')
        saved = {key: os.environ.pop(key) for key in proxy_keys if key in os.environ}
        try:
            self.client = client_cls('127.0.0.1', port)
        finally:
            os.environ.update(saved)
        # First JAX compilation can exceed the pinned client's ping timeout.
        self.client._ws.ping_timeout = None
        self.metadata = self.client.get_server_metadata()
        self.index = 0
        require(self.metadata.get('checkpoint') == str(Path(checkpoint).resolve()) and
                self.metadata.get('checkpoint_sha256') and
                self.metadata.get('config') == 'pi05_droid_jointpos_polaris' and
                self.metadata.get('backend') == 'OpenPI/JAX' and
                self.metadata.get('inferences') == 0, 'Fresh identified pi05 server required')

    def infer(self, observation, output):
        before = time.monotonic()
        predicted = self.client.infer({
            'observation/exterior_image_1_left': observation['main_images'],
            'observation/wrist_image_left': observation['wrist_images'],
            'observation/joint_position': observation['states'][:7],
            'observation/gripper_position': observation['states'][7:8],
            'prompt': observation['instruction'],
        })
        identity = predicted.get('policy_identity', {})
        require(identity.get('checkpoint_sha256') == self.metadata['checkpoint_sha256'] and
                identity.get('inference_index') == self.index, 'pi05 identity/sequence mismatch')
        raw = np.asarray(predicted['actions'], np.float32)
        require(raw.shape == (15, 8) and np.isfinite(raw).all(), 'Expected finite pi05 [15,8] chunk')
        actions = raw.copy()
        actions[:, 7] = (actions[:, 7] > .5).astype(np.float32)
        np.savez_compressed(output, raw_actions=raw, actions=actions,
                            **{k: observation[k] for k in ('main_images', 'wrist_images', 'states')})
        self.index += 1
        return actions, dict(prediction_id=uuid.uuid4().hex,
                             inference_index=self.index - 1,
                             inference_seconds=time.monotonic() - before)

    def close(self):
        self.client._ws.close()
