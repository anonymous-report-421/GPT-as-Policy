"""Bounded-page ACP inventory; 500 large jobs can exceed the gateway's 4 MiB limit."""
import json
import os
import subprocess


def user_jobs(sco, workspace, *, runner=subprocess.run):
    env = dict(os.environ)
    for name in ('NO_PROXY', 'no_proxy'):
        env[name] = 'management.sensecoreapi.cn,aec2.cn-sh-01.sensecoreapi.cn'
    jobs = {}
    for page in range(1, 201):
        result = runner([str(sco), 'acp', 'jobs', 'list', '--workspace-name='+workspace,
            '--user-name=sujiayi', '--page-size=100', '--page-token='+str(page), '-o', 'json'],
            env=env, capture_output=True, text=True, check=True, timeout=90)
        # sco emits this exact text (even with -o json) for an empty page.
        # Other malformed/empty responses must still fail closed.
        rows = [] if result.stdout.strip() == 'No jobs found' else json.loads(result.stdout)
        if not isinstance(rows, list):
            raise ValueError('Unexpected ACP page schema')
        if rows and all(row['name'] in jobs for row in rows):
            raise ValueError('ACP pagination repeated a page; inventory incomplete')
        for row in rows:
            if row.get('ownership', {}).get('user_name') != 'sujiayi':
                raise ValueError('ACP user filter returned another owner')
            jobs[row['name']] = row
        if len(rows) < 100:
            return list(jobs.values())
    raise ValueError('ACP pagination bound reached; do not submit with incomplete inventory')
