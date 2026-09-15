"""Blocking OpenPI inference. Arrays stay in files, not in model tool arguments."""
from pathlib import Path
import os
import time
import uuid
import numpy as np

from ..io import require
from .checkpoint import ROBODOJO_CONFIG


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
                self.metadata.get('config') == ROBODOJO_CONFIG and
                self.metadata.get('action_horizon') == 50 and self.metadata.get('action_dim') == 14 and
                self.metadata.get('backend') == 'OpenPI/JAX' and
                self.metadata.get('inferences') == 0, 'Fresh identified pi05 server required')

    def infer(self, observation, output):
        before = time.monotonic()
        inputs = dict(
            state=observation['states'], prompt=observation['instruction'],
            images={key: np.transpose(observation[key], (2, 0, 1))
                    for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')})
        predicted = self.client.infer(inputs)
        identity = predicted.get('policy_identity', {})
        require(identity.get('checkpoint_sha256') == self.metadata['checkpoint_sha256'] and
                identity.get('inference_index') == self.index, 'pi05 identity/sequence mismatch')
        raw = np.asarray(predicted['actions'], np.float32)
        shape = (50, 14)
        require(raw.shape == shape and np.isfinite(raw).all(), f'Expected finite pi05 {shape} chunk')
        actions = raw.copy()
        # Match native RoboDojo execution: continuous opening, NOT binary closure.
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0., 1.)
        keys = ('cam_high', 'cam_left_wrist', 'cam_right_wrist', 'states')
        np.savez_compressed(output, raw_actions=raw, actions=actions,
                            **{k: observation[k] for k in keys})
        self.index += 1
        return actions, dict(prediction_id=uuid.uuid4().hex,
                             inference_index=self.index - 1,
                             inference_seconds=time.monotonic() - before)

    def close(self):
        self.client._ws.close()
