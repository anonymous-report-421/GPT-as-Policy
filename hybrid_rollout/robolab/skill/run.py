"""Add three rollout services to a full, persistent Codex policy agent."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import traceback

from ..io import InputError, write_json
from ..settings import MODEL, EFFORT
from ..pi05_server.client import Pi05Client
from ..robolab_server.client import RolloutTools
from .schema import _obj, response_schema
from .transport import StdioAppServer
from .workspace import prepare_workspace

SKILL_ROOT = Path(__file__).parent
NATIVE_WORK_ITEMS = frozenset(('commandExecution', 'fileChange', 'imageView',
    'webSearch', 'mcpToolCall', 'collabAgentToolCall'))


def toml_value(value):
    """Codex -c values are TOML; JSON objects/quoted dotted keys are not equivalent."""
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml_value(v) for k, v in value.items()) + '}'
    return json.dumps(value)


def agent_config(audit, agent):
    # Keep Codex's normal tool/skill configuration. These two capabilities are
    # essential, even if an inherited profile had disabled them previously.
    return dict(model=MODEL, model_reasoning_effort=EFFORT,
        sqlite_home=str(audit/'runtime_db'), log_dir=str(audit/'runtime_logs'),
        default_permissions='rollout_agent', **{
            'features.shell_tool': True, 'features.view_image': True,
            'permissions.rollout_agent.extends': ':workspace',
            # The parent contains host-owned RPC logs and experiment artifacts.
            # Only the agent directory is a writable runtime workspace root.
            'permissions.rollout_agent.filesystem': {
                str(audit.parent): 'read', str(agent): 'write'},
        })


def tool_specs():
    string = {'type': 'string'}
    def spec(name, description, **properties):
        return dict(type='function', name=name, description=description,
                    inputSchema=_obj(properties))
    return [
        spec('robolab_start', 'Blocking: start the requested RoboLab episode; save and return current RGB/proprio.',
             task=string, output_dir=string),
        spec('pi05_infer', 'Blocking: infer pi05 once from the latest observation; save chunk and return robot-only FK.',
             observation_path=string, output_dir=string),
        spec('robolab_execute', 'Blocking: validate and execute a reviewed fresh chunk or short EEF correction; save and return new observation and terminal result.',
             proposal_path=string, response=response_schema(), output_dir=string),
    ]


def content_items(packet, *, images):
    items = [dict(type='inputText', text=json.dumps(packet, separators=(',', ':')))]
    if images:
        for item in packet.get('images', []):
            data = base64.b64encode(Path(item['path']).read_bytes()).decode('ascii')
            items.append(dict(type='inputImage', imageUrl='data:image/png;base64,' + data))
    return items


class CodexPolicy:
    def __init__(self, workspace, codex, *, timeout=900, transport_factory=StdioAppServer):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=False)
        self.agent_workspace = prepare_workspace(self.workspace, SKILL_ROOT)
        self.timeout = timeout
        skill = (SKILL_ROOT/'SKILL.md').read_text()
        gate = (SKILL_ROOT/'gate_prompt.md').read_text()
        context = (SKILL_ROOT/'context/teacher_context.md').read_text()
        self.prompt = (skill + '\n\n# Unchanged baseline gate prompt\n\n' + gate
                       + '\n\n' + context + '\n\nAgent working directory: '
                       + str(self.agent_workspace)
                       + '\nResolve context/ and workspace.json relative to that directory.\n')
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace/'SKILL.md').write_text(skill)
        (self.workspace/'gate_prompt.md').write_text(gate)
        (self.workspace/'PROMPT.md').write_text(self.prompt)
        specs = tool_specs()
        write_json(self.workspace/'tools.json', specs)
        version = subprocess.run([codex, '--version'], capture_output=True, text=True, check=True).stdout.strip()
        config = agent_config(self.workspace, self.agent_workspace)
        argv = [codex, 'app-server', '--stdio', '--strict-config']
        for key, value in config.items():
            argv += ['-c', key + '=' + toml_value(value)]
        write_json(self.workspace/'launch.json', dict(argv=argv, config=config))
        self.transport = transport_factory(argv, self.workspace)
        try:
            self.transport.request('initialize', dict(
                clientInfo=dict(name='robolab-hybrid-policy', version='1'),
                capabilities=dict(experimentalApi=True)), timeout)
            self.transport.notify('initialized', {})
            response = self.transport.request('thread/start', dict(
                cwd=str(self.agent_workspace), model=MODEL, modelProvider='openai',
                config=dict(model_reasoning_effort=EFFORT), developerInstructions=self.prompt,
                dynamicTools=specs, ephemeral=False,
                allowProviderModelFallback=False, approvalPolicy='never',
                approvalsReviewer='auto_review', permissions='rollout_agent',
                runtimeWorkspaceRoots=[str(self.agent_workspace)]), timeout)
            if response.get('model') != MODEL or response.get('reasoningEffort') != EFFORT:
                raise RuntimeError(f'Codex model/effort differs from the required {MODEL}/{EFFORT}: '
                                   f'{response.get("model")}/{response.get("reasoningEffort")}')
            self.thread_id = response['thread']['id']
            write_json(self.workspace/'worker.json', dict(model=MODEL, reasoning_effort=EFFORT,
                codex_version=version, thread_id=self.thread_id, pid=self.transport.process.pid,
                prompt_sha256=self.prompt_sha256, no_rollback=True, tools=[s['name'] for s in specs],
                service_tools_are_additive=True, agent_workspace=str(self.agent_workspace),
                native_tools='Codex defaults; shell and image viewing explicitly enabled',
                baseline_full_conversation_available=False))
        except BaseException:
            self.close()
            raise

    def _turn(self, text):
        result = self.transport.request('turn/start', dict(threadId=self.thread_id,
            model=MODEL, effort=EFFORT, approvalPolicy='never', approvalsReviewer='auto_review',
            permissions='rollout_agent', cwd=str(self.agent_workspace),
            runtimeWorkspaceRoots=[str(self.agent_workspace)], input=[dict(type='text', text=text)]), self.timeout)
        return result['turn']['id']

    def run(self, rollout):
        turn_id = self._turn('Act as the autonomous policy agent for this single simulation rollout. '
            'Use your normal file, image, shell/code and planning tools as useful, alongside '
            'the three rollout service tools. Read workspace.json for paths and interpreter; '
            'keep working notes and analysis in your agent workspace. '
            'The user authorizes sending this episode\'s two RGB images, proprio, robot-only FK, '
            'task text and same-episode history to OpenAI Codex for online decisions. '
            'First call: ' + json.dumps(rollout.next_call()))
        call_index = 0
        errors = 0
        continuations = 0
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Codex made no completed service or native tool call within the configured timeout')
            event = self.transport.next_message(remaining)
            method, params = event.get('method'), event.get('params', {})
            if method == 'item/tool/call':
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    raise RuntimeError('Tool call belongs to another thread or turn')
                name, arguments = params['tool'], params['arguments']
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                print(json.dumps(dict(event='tool_start', tool=name, call=call_index,
                                      phase=rollout.phase, step_id=rollout.tick)), flush=True)
                audit = dict(call_id=params['callId'], tool=name, arguments=arguments)
                write_json(self.workspace/f'call_{call_index:04d}_request.json', audit)
                handlers = dict(robolab_start=rollout.start, pi05_infer=rollout.infer,
                                robolab_execute=rollout.execute)
                try:
                    if name not in handlers or not isinstance(arguments, dict):
                        raise InputError('Unknown host service; native Codex tools are executed by app-server')
                    packet = handlers[name](**arguments)
                except InputError as error:
                    errors += 1
                    packet = dict(error=str(error), no_execution=True, next_call=rollout.next_call())
                    self.transport.reply(event['id'], dict(success=False, contentItems=content_items(packet, images=False)))
                    if errors >= 5:
                        raise RuntimeError('Five rejected tool calls; stopping without further execution') from error
                else:
                    self.transport.reply(event['id'], dict(success=True,
                        contentItems=content_items(packet, images=name != 'pi05_infer')))
                write_json(self.workspace/f'call_{call_index:04d}_result.json', packet)
                call_index += 1
                print(json.dumps(dict(event='tool_done', tool=name, step_id=rollout.tick,
                    phase=rollout.phase, counters=rollout.counters)), flush=True)
                deadline = time.monotonic() + self.timeout
            elif method in ('item/started', 'item/completed'):
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    continue
                item = params.get('item', {})
                kind = item.get('type')
                if kind in NATIVE_WORK_ITEMS or (kind == 'agentMessage' and method == 'item/completed'):
                    # Public outputs only: never copy private reasoning items.
                    with (self.workspace/'agent_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(method=method, **params), ensure_ascii=False)+'\n')
                    event_record = dict(event='agent_activity', phase=method.split('/')[-1],
                        item_type=kind, item_id=item.get('id'), step_id=rollout.tick,
                        status=item.get('status'))
                    if kind == 'agentMessage':
                        event_record['text'] = item.get('text', '')
                    elif kind == 'commandExecution':
                        event_record['command'] = item.get('command')
                        event_record['exit_code'] = item.get('exitCode')
                    print(json.dumps(event_record, ensure_ascii=False), flush=True)
                    if kind in NATIVE_WORK_ITEMS:
                        # Real analysis is progress, not a missing rollout decision.
                        deadline = time.monotonic() + self.timeout
            elif method == 'turn/completed' and params.get('threadId') == self.thread_id:
                turn = params.get('turn', {})
                if turn.get('id') != turn_id:
                    continue
                write_json(self.workspace/f'turn_{continuations:02d}_completed.json', turn)
                if turn.get('status') != 'completed':
                    raise RuntimeError(f'Codex turn ended: {turn.get("status")}')
                if rollout.phase == 'done':
                    return
                if continuations >= 2:
                    raise RuntimeError('Codex ended repeatedly before completing the episode')
                # Context-only continuation, never a human action judgement.
                continuations += 1
                turn_id = self._turn('The same episode is unfinished. Continue the skill; next call: ' +
                                     json.dumps(rollout.next_call()))
                deadline = time.monotonic() + self.timeout
            elif method in ('error', 'turn/failed'):
                if method == 'error' and params.get('willRetry') is True:
                    # App-server owns model transport retry; this does not
                    # repeat inference, a tool handler, or physical execution.
                    print(json.dumps(dict(event='codex_transport_retry',
                        error=params.get('error'), step_id=rollout.tick)), flush=True)
                    continue
                raise RuntimeError(f'Codex error: {params}')
            elif 'id' in event and 'method' in event:
                raise RuntimeError(f'Unexpected Codex capability request: {method}')

    def close(self):
        self.transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--codex', required=True)
    parser.add_argument('--sim-port', type=int, default=19093)
    parser.add_argument('--student-port', type=int, default=18810)
    parser.add_argument('--max-decisions', type=int, default=180)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    student = worker = rollout = None
    def terminate(signum, frame):
        raise KeyboardInterrupt('Stopping only this owned rollout and Codex process')
    signal.signal(signal.SIGTERM, terminate)
    try:
        student = Pi05Client(args.student_port, args.checkpoint)
        worker = CodexPolicy(args.output/'codex_workspace', args.codex)
        rollout = RolloutTools(args.output, args.task, student, sim_port=args.sim_port,
            seed=args.seed, max_decisions=args.max_decisions, prompt_sha256=worker.prompt_sha256)
        worker.run(rollout)
    except BaseException:
        write_json(args.output/'failure.json', dict(error=traceback.format_exc(),
            episode_id=rollout.episode if rollout else None, step_id=rollout.tick if rollout else 0,
            counters=rollout.counters if rollout else {}, completed=False))
        if rollout is not None and rollout.episode is not None and rollout.phase != 'done':
            try:
                rollout.finish('controller_error')
            except Exception:
                pass  # Never reconnect or retry an action with an uncertain ACK.
        raise
    finally:
        if rollout is not None:
            rollout.close()
        if student is not None:
            student.close()
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
