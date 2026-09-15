"""Full-agent setup and native tool progress, with no model/GPU calls."""
import json
import tomllib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from .skill.run import CodexPolicy
from .skill.transport import StdioAppServer
from .test_contract import Transport, make


def test_native_tools_are_additive_and_workspace_is_usable(tmp_path, monkeypatch):
    from .skill import run
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='codex-fixture'))
    audit = tmp_path/'controller/codex_workspace'
    policy = CodexPolicy(audit, 'codex', transport_factory=Transport)
    agent = policy.agent_workspace
    launch = json.loads((audit/'launch.json').read_text())['config']
    assert launch['model'] == 'gpt-6-astra' and launch['model_reasoning_effort'] == 'xhigh'
    assert json.loads((audit/'worker.json').read_text())['reasoning_effort'] == 'xhigh'
    assert launch['features.shell_tool'] and launch['features.view_image']
    assert not any(k.startswith('features.') and v is False for k, v in launch.items())
    assert 'project_doc_max_bytes' not in launch
    assert 'skills.include_instructions' not in launch
    assert 'orchestrator.mcp.enabled' not in launch
    assert launch['permissions.rollout_agent.extends'] == ':workspace'
    assert launch['permissions.rollout_agent.filesystem'][str(audit.parent)] == 'read'
    assert launch['permissions.rollout_agent.filesystem'][str(agent)] == 'write'
    argv = json.loads((audit/'launch.json').read_text())['argv']
    override = next(v for v in argv if v.startswith('permissions.rollout_agent.filesystem='))
    assert tomllib.loads(override)['permissions']['rollout_agent']['filesystem'] == launch['permissions.rollout_agent.filesystem']
    params = policy.transport.thread_params
    assert params['config']['model_reasoning_effort'] == 'xhigh'
    assert params['cwd'] == str(agent) and params['runtimeWorkspaceRoots'] == [str(agent)]
    assert params['ephemeral'] is False and 'environments' not in params
    assert 'baseInstructions' not in params  # Preserve normal native Codex instructions.
    metadata = json.loads((agent/'workspace.json').read_text())
    assert Path(metadata['python_executable']).is_file()
    assert metadata['controller_output'] == str(audit.parent)
    source_root = Path(metadata['source_root'])
    assert source_root == Path(__file__).resolve().parents[2]
    assert (source_root/'hybrid_rollout/robolab/robolab_server/client.py').is_file()
    assert metadata['baseline_full_conversation_available'] is False
    skill = agent/'.agents/skills/robolab-hybrid-rollout'
    assert (skill/'context/eef_control.md').is_file()
    assert (skill/'gate_prompt.md').read_bytes() == (audit/'gate_prompt.md').read_bytes()
    assert (agent/'AGENTS.md').is_file() and (agent/'NOTES.md').is_file()
    assert (agent/'scratch').is_dir()


def test_native_activity_is_logged_without_host_executing_or_rejecting_it(tmp_path, monkeypatch):
    from .skill import run
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='codex-fixture'))
    class NativeTransport(Transport):
        def __init__(self, *args):
            super().__init__(*args)
            self.native = iter([
                ('item/started', dict(id='cmd', type='commandExecution', command='python calculation.py')),
                ('item/completed', dict(id='cmd', type='commandExecution', exitCode=0, aggregatedOutput='2.58 cm')),
                ('item/completed', dict(id='image', type='imageView', path='scratch/comparison.png')),
                ('item/completed', dict(id='reason', type='reasoning', text='private')),
                ('item/completed', dict(id='message', type='agentMessage', text='Checking the tracking error.')),
            ])
        def next_message(self, timeout):
            entry = next(self.native, None)
            if entry:
                method, item = entry
                return dict(method=method, params=dict(threadId='thread', turnId='turn', item=item))
            return super().next_message(timeout)
    policy = CodexPolicy(tmp_path/'workspace', 'codex', transport_factory=NativeTransport)
    tools = make(tmp_path)
    policy.run(tools)
    events = [json.loads(line) for line in (policy.workspace/'agent_events.jsonl').read_text().splitlines()]
    assert len(events) == 4
    assert all(row['item']['type'] != 'reasoning' for row in events)
    assert len(policy.transport.replies) == 3  # Native tools do not route through the rollout dispatcher.
    assert tools.student.calls == 1 and tools.sim.calls.count('chunk_step') == 1
    assert json.loads((tools.output/'run.json').read_text())['teacher_reasoning_effort'] == 'xhigh'


def test_high_downgrade_is_rejected_before_rollout(tmp_path, monkeypatch):
    from .skill import run
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='codex-fixture'))
    closed = []
    class DowngradedTransport(Transport):
        def request(self, method, params, timeout):
            result = super().request(method, params, timeout)
            if method == 'thread/start':
                result['reasoningEffort'] = 'high'
            return result
        def close(self):
            closed.append(True)
    with pytest.raises(RuntimeError, match='required gpt-6-astra/xhigh'):
        CodexPolicy(tmp_path/'workspace', 'codex', transport_factory=DowngradedTransport)
    assert closed == [True]
    assert not (tmp_path/'workspace/worker.json').exists()


def test_native_progress_refreshes_inactivity_deadline(tmp_path, monkeypatch):
    from .skill import run
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='codex-fixture'))
    clock = [0.0]
    monkeypatch.setattr(run.time, 'monotonic', lambda: clock[0])
    class BusyTransport(Transport):
        completed = 0
        def next_message(self, timeout):
            assert timeout > 0
            if self.completed < 3:
                clock[0] += 800
                self.completed += 1
                return dict(method='item/completed', params=dict(threadId='thread', turnId='turn',
                    item=dict(id=str(self.completed), type='commandExecution', exitCode=0)))
            return super().next_message(timeout)
    policy = CodexPolicy(tmp_path/'workspace', 'codex', transport_factory=BusyTransport)
    tools = make(tmp_path)
    policy.run(tools)
    assert clock[0] == 2400 and tools.phase == 'done'


def test_failed_process_start_wakes_pending_rpc(tmp_path):
    server = StdioAppServer([sys.executable, '-c',
        'import sys; sys.stdin.readline(); sys.stderr.write("invalid test config\\n"); sys.exit(2)'], tmp_path)
    try:
        with pytest.raises(RuntimeError, match='stdout closed'):
            server.request('initialize', {}, timeout=5)
    finally:
        server.close()
