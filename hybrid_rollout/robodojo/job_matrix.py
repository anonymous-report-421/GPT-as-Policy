"""Emit ten cluster replicas, each running six fresh sequential episodes."""
from __future__ import annotations

import argparse
import json

from .evaluation import TASKS, read_panel, replica_cases

def jobs(experiment_id: str, manifest_path, attempt: int = 0, count: int = 6):
    panel = read_panel(manifest_path)
    for replica, task in enumerate(TASKS):
        replica_cases(panel, task, replica)
        yield {
            'ROLLOUT_EXPERIMENT_ID': experiment_id,
            'ROBODOJO_TASK': task,
            'ROLLOUT_REPLICA_ID': str(replica),
            'ROLLOUT_EVAL_MANIFEST': str(manifest_path),
            'ROLLOUT_EVAL_MANIFEST_SHA256': panel['panel_sha256'],
            'ROLLOUT_COUNT': str(count),
            'ROLLOUT_ATTEMPT': str(attempt),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--eval-manifest', required=True)
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--count', type=int, default=6)
    args = parser.parse_args()
    if args.attempt < 0 or not 1 <= args.count <= 6:
        parser.error('attempt must be non-negative and count must be in 1..6')
    for job in jobs(args.experiment_id, args.eval_manifest, args.attempt, args.count):
        print(json.dumps(job, sort_keys=True))


if __name__ == '__main__':
    main()
