"""Serve an identified base or DAgger JAX checkpoint through OpenPI's protocol."""
import argparse
import json
from pathlib import Path

from ..io import require, sha256, write_json
from .checkpoint import checkpoint_identity, data_contract


def main():
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--port', type=int, default=18830)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--identity-output', type=Path, required=True)
    args = p.parse_args()
    require(not args.identity_output.exists(), 'Identity output already exists')
    import openpi
    native_root = Path(openpi.__file__).resolve().parent
    source_files = [*native_root.rglob('*.py'), *Path(__file__).parent.glob('*.py')]
    source_sha256 = {str(p): sha256(p) for p in source_files}
    identity = checkpoint_identity(args.checkpoint)
    cfg, _ = data_contract(args.checkpoint)
    policy = create_trained_policy(cfg, args.checkpoint)
    require(checkpoint_identity(args.checkpoint) == identity, 'Checkpoint changed while loading')
    require(all(sha256(Path(p)) == digest for p, digest in source_sha256.items()), 'Source changed while loading')
    metadata = dict(identity, policy_rng='JAX seed0; fresh server per evaluation episode', policy_rng_seed=0, inferences=0,
                    implementation_sha256=source_sha256)
    class CountedPolicy:
        def infer(self, obs):
            result = policy.infer(obs)
            result['policy_identity'] = dict(checkpoint_sha256=identity['checkpoint_sha256'],
                                             inference_index=metadata['inferences'])
            metadata['inferences'] += 1
            return result
    write_json(args.identity_output, metadata)
    print(json.dumps(dict(event='loaded', port=args.port, checkpoint_sha256=identity['checkpoint_sha256'])), flush=True)
    WebsocketPolicyServer(CountedPolicy(), host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
