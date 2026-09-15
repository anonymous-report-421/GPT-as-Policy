"""Add three rollout services to a full, persistent Codex policy agent."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from ..io import InputError, write_json
from ..prompt_context import CONTEXT_VERSION
from ..settings import MODEL, EFFORT, PROVIDER, DEFAULT_CODEX_IMAGE_MAX_EDGE
from ..robodojo_server.client import RoboDojoTools
from ..method import evaluation_method
from .schema import _obj, response_schema, direct_response_schema
from .transport import StdioAppServer
from .workspace import prepare_workspace
from .image_preview import image_max_edge, prepare_image
from .network_recovery import NETWORK_CONTINUE_DELAYS, closed_network_turn, is_network_error

SKILL_ROOT = Path(__file__).parent
CONTROLLER_VERSION = 'recoverable_tool_input_v1'
NATIVE_WORK_ITEMS = frozenset(('commandExecution', 'fileChange', 'imageView',
    'webSearch', 'mcpToolCall', 'collabAgentToolCall'))


def rejected_input(error, rollout, arguments, count):
    """Feedback only for InputError's pre-execution rejection contract.

    Never change arguments, clip a target, retry a handler or reset simulation.
    Other exceptions (including uncertain RPC outcomes) remain fatal.
    """
    packet = dict(error=str(error), error_type='recoverable_tool_input',
        no_execution=True, retryable=True, controller_version=CONTROLLER_VERSION,
        rejected_calls=count, step_id=rollout.tick, next_call=rollout.next_call(),
        correction='Correct the arguments and retry next_call in this same episode. '
                   'Do not reset or replay already executed actions.')
    if getattr(rollout, 'observation_path', None):
        packet['observation_path'] = str(rollout.observation_path)
    request = getattr(rollout, 'request', None)
    if isinstance(request, dict) and rollout.phase in ('act', 'execute'):
        packet['request_id'] = request.get('request_id')
        packet['current_eef'] = request.get('current_eef')
        # Explain the existing Euclidean bound without changing its validator.
        response = arguments.get('response') if isinstance(arguments, dict) else None
        if ('exceeds 5 cm' in str(error) and isinstance(response, dict)
                and response.get('mode') == 'eef'):
            distances = {}
            for arm in ('left', 'right'):
                try:
                    target = response['target'][arm]['position']
                    current = request['current_eef'][arm]['position']
                    if len(target) != 3 or len(current) != 3:
                        continue
                    distance = math.dist(target, current)
                    if math.isfinite(distance):
                        distances[arm] = distance
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue  # Diagnostic only; the original error is authoritative.
            packet['eef_translation'] = dict(distance_m=distances, max_distance_m=0.05,
                metric='Euclidean distance from the current EEF, not a per-axis limit')
    return packet


def toml_value(value):
    """Codex -c values are TOML; JSON objects/quoted dotted keys are not equivalent."""
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml_value(v) for k, v in value.items()) + '}'
    return json.dumps(value)


def agent_config(audit, agent):
    # Keep Codex's normal tool/skill configuration. These two capabilities are
    # essential, even if an inherited profile had disabled them previously.
    return dict(model=MODEL, model_provider=PROVIDER, model_reasoning_effort=EFFORT,
        sqlite_home=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_db'),
        log_dir=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_logs'),
        default_permissions='rollout_agent', **{
            'features.shell_tool': True, 'features.view_image': True,
            'shell_environment_policy.inherit': 'all',
            'permissions.rollout_agent.extends': ':workspace',
            # The parent contains host-owned RPC logs and experiment artifacts.
            # Only the agent directory is a writable runtime workspace root.
            'permissions.rollout_agent.filesystem': {
                str(audit.parent): 'read', str(agent): 'write'},
        })


def tool_specs(method='pi05_plus_gpt'):
    string = {'type': 'string'}
    def spec(name, description, **properties):
        return dict(type='function', name=name, description=description,
                    inputSchema=_obj(properties))
    if method == 'gpt_only':
        return [spec('robodojo_start', 'Blocking: start the requested episode and return RGB/proprio.',
                     task=string, output_dir=string),
                spec('robodojo_act', 'Blocking: execute GPT-authored bounded dual-arm EEF targets '
                     '(1-5 steps), then return RGB/proprio and native outcome. '
                     'No pi05 inference, proposals or direct joint-action commands exist.',
                     observation_path=string, response=direct_response_schema(), output_dir=string)]
    return [
        spec('robodojo_start', 'Blocking: start the requested RoboDojo episode; save and return current RGB/proprio.',
             task=string, output_dir=string),
        spec('pi05_infer', 'Blocking: infer pi05 once from the latest observation; save chunk and return full robot-only FK. '
             'action_diagnostics provides shape, finiteness, opening ranges and maximum successive joint step. '
             'Use these for routine arithmetic; native tools remain available for independent or additional checks. '
             'Diagnostics are not a safety or gate verdict: perform the unchanged visual and intent assessment.',
             observation_path=string, output_dir=string),
        spec('robodojo_execute', 'Blocking: validate and execute a reviewed fresh chunk or short EEF correction; save and return new observation and terminal result.',
             proposal_path=string, response=response_schema(), output_dir=string),
    ]


def content_items(packet, *, images, max_image_edge=DEFAULT_CODEX_IMAGE_MAX_EDGE):
    # This boundary is downstream of observation capture and policy inference.
    # Copy metadata only: never change packet, source PNG/NPZ or policy arrays.
    visible_packet = packet
    attachments = []
    if images:
        visible_packet = dict(packet, images=[])
        for item in packet.get('images', []):
            descriptor, payload = prepare_image(item, max_image_edge)
            visible_packet['images'].append(descriptor)
            data = base64.b64encode(payload).decode('ascii')
            attachments.append(dict(type='inputImage', imageUrl='data:image/png;base64,' + data))
    return [dict(type='inputText', text=json.dumps(visible_packet, separators=(',', ':'))), *attachments]


class CodexPolicy:
    def __init__(self, workspace, codex, *, timeout=900, transport_factory=StdioAppServer,
                 method='pi05_plus_gpt'):
        self.method = evaluation_method(method)
        direct = self.method == 'gpt_only'
        skill_root = SKILL_ROOT.parent/'robodojo-gpt-only-rollout' if direct else SKILL_ROOT
        self.image_max_edge = image_max_edge(os.environ.get(
            'CODEX_IMAGE_MAX_EDGE', str(DEFAULT_CODEX_IMAGE_MAX_EDGE)))
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=False)
        write_json(self.workspace.parent/'controller_protocol.json', dict(
            controller_version=CONTROLLER_VERSION, context_version=CONTEXT_VERSION,
            recoverable_input_error_limit=None, validation_unchanged=True,
            same_thread=True, simulator_reset_on_input_error=False,
            action_replay_on_input_error=False))
        self.agent_workspace = prepare_workspace(self.workspace, skill_root, method=self.method)
        self.timeout = timeout
        skill = (skill_root/'SKILL.md').read_text()
        gate = '' if direct else (skill_root/'gate_prompt.md').read_text()
        context = (skill_root/'context/teacher_context.md').read_text()
        self.prompt = (skill + ('' if direct else '\n\n# Unchanged baseline gate prompt\n\n' + gate)
                       + '\n\n' + context + '\n\nAgent working directory: '
                       + str(self.agent_workspace)
                       + '\nResolve context/ and workspace.json relative to that directory.\n')
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace/'SKILL.md').write_text(skill)
        if not direct:
            (self.workspace/'gate_prompt.md').write_text(gate)
        (self.workspace/'PROMPT.md').write_text(self.prompt)
        specs = tool_specs(self.method)
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
                clientInfo=dict(name='robodojo-hybrid-policy', version='1'),
                capabilities=dict(experimentalApi=True)), timeout)
            self.transport.notify('initialized', {})
            response = self.transport.request('thread/start', dict(
                cwd=str(self.agent_workspace), model=MODEL, modelProvider=PROVIDER,
                config=dict(model_reasoning_effort=EFFORT), developerInstructions=self.prompt,
                dynamicTools=specs, ephemeral=False,
                allowProviderModelFallback=False, approvalPolicy='never',
                approvalsReviewer='auto_review', permissions='rollout_agent',
                runtimeWorkspaceRoots=[str(self.agent_workspace)]), timeout)
            if response.get('model') != MODEL or response.get('reasoningEffort') != EFFORT:
                raise RuntimeError(f'Codex model/effort differs from the required {MODEL}/{EFFORT}: '
                                   f'{response.get("model")}/{response.get("reasoningEffort")}')
            self.thread_id = response['thread']['id']
            write_json(self.workspace/'worker.json', dict(model=MODEL, model_provider=PROVIDER,
                controller_version=CONTROLLER_VERSION,
                evaluation_method=self.method,
                reasoning_effort=EFFORT, context_version=CONTEXT_VERSION,
                codex_version=version, thread_id=self.thread_id, pid=self.transport.process.pid,
                prompt_sha256=self.prompt_sha256, no_rollback=True, tools=[s['name'] for s in specs],
                service_tools_are_additive=True, agent_workspace=str(self.agent_workspace),
                native_tools='Codex defaults; shell and image viewing explicitly enabled',
                network_recovery=dict(revision='same_thread_continue_v1',
                    backoff_seconds=list(NETWORK_CONTINUE_DELAYS), restart_simulator=False,
                    new_thread=False, replay_actions=False),
                codex_image_max_edge=self.image_max_edge, policy_images_resized=False,
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

    def _network_continue(self, rollout, old_turn, error, consecutive, serial):
        if not is_network_error(error) or consecutive >= len(NETWORK_CONTINUE_DELAYS):
            raise RuntimeError(f'Codex network continue exhausted or ineligible: {error}')
        delay = NETWORK_CONTINUE_DELAYS[consecutive]
        record = dict(revision='same_thread_continue_v1', thread_id=self.thread_id,
            previous_turn_id=old_turn, error=error, step_id=rollout.tick,
            phase=rollout.phase, next_call=rollout.next_call(), delay_seconds=delay,
            consecutive_attempt=consecutive+1, status='waiting', physical_actions_replayed=0)
        path = self.workspace/f'network_continue_{serial:04d}.json'
        write_json(path, record)
        print(json.dumps(dict(event='codex_network_continue_wait', **record)), flush=True)
        # Keep Python/the simulator/observation alive. SIGTERM still exits normally.
        for _ in range(delay):
            time.sleep(1)
        text = ('Continue the same rollout. The previous turn ended because of a network error. '
                'The simulator and recorded actions are unchanged; do not reset or repeat executed actions. '
                f'Current step_id={rollout.tick}. Use the latest recorded observation and next_call: '
                + json.dumps(rollout.next_call()))
        if getattr(rollout, 'observation_path', None):
            text += '\nLatest observation: ' + str(rollout.observation_path)
        record['status'] = 'turn_start_requested'
        write_json(path, record)  # Ambiguous RPC must never be reissued blindly.
        new_turn = self._turn(text)
        record.update(status='continue_sent', new_turn_id=new_turn)
        write_json(path, record)
        print(json.dumps(dict(event='codex_network_continue_sent', thread_id=self.thread_id,
            old_turn_id=old_turn, turn_id=new_turn, step_id=rollout.tick)), flush=True)
        return new_turn

    def run(self, rollout):
        turn_id = self._turn('Act as the autonomous policy agent for this single simulation rollout. '
            'Use your normal file, image, shell/code and planning tools as useful, alongside '
            'the rollout service tools. Read workspace.json for paths and interpreter; '
            'keep working notes and analysis in your agent workspace. Use English for every '
            'public explanation, assessment, progress update, note, and final report. '
            'The user authorizes sending this episode\'s three RGB images, proprio, robot-only FK, '
            'task text and same-episode history to OpenAI Codex for online decisions. '
            'First call: ' + json.dumps(rollout.next_call()))
        call_index = 0
        errors = 0
        continuations = 0
        network_error = None
        network_consecutive = network_serial = 0
        completed_turns = 0
        token_limit = int(os.environ.get('CODEX_MAX_TOTAL_TOKENS', '0'))
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError('Codex made no completed service or native tool call within the configured timeout')
                event = self.transport.next_message(remaining)
            except TimeoutError:
                if not network_error:
                    raise
                # A missing terminal notification does not authorize a second active turn.
                response = self.transport.request('thread/read', dict(
                    threadId=self.thread_id, includeTurns=True), min(30, self.timeout))
                turn = closed_network_turn(response, self.thread_id, turn_id)
                if turn is None:
                    raise
                event = dict(method='turn/completed', params=dict(threadId=self.thread_id, turn=turn))
            method, params = event.get('method'), event.get('params', {})
            if method == 'thread/tokenUsage/updated' and params.get('threadId') == self.thread_id:
                usage = params.get('tokenUsage', {})
                write_json(self.workspace.parent/'token_usage.json', dict(
                    model=MODEL, effort=EFFORT, provider=PROVIDER, usage=usage,
                    total_tokens_include_cached_input=True, configured_limit=token_limit))
                print(json.dumps(dict(event='token_usage', total=usage.get('total', {}))), flush=True)
                if token_limit and usage.get('total', {}).get('totalTokens', 0) >= token_limit:
                    raise RuntimeError('Configured total-token budget reached; stopping this episode')
            elif method == 'item/tool/call':
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    raise RuntimeError('Tool call belongs to another thread or turn')
                name, arguments = params['tool'], params['arguments']
                print(json.dumps(dict(event='tool_start', tool=name, call=call_index,
                                      phase=rollout.phase, step_id=rollout.tick)), flush=True)
                audit = dict(call_id=params['callId'], tool=name, arguments=arguments)
                write_json(self.workspace/f'call_{call_index:04d}_request.json', audit)
                handlers = (dict(robodojo_start=rollout.start, robodojo_act=rollout.act)
                    if self.method == 'gpt_only' else
                    dict(robodojo_start=rollout.start, pi05_infer=rollout.infer, robodojo_execute=rollout.execute))
                try:
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as error:
                            raise InputError(f'Tool arguments must be a JSON object: {error}') from error
                    if name not in handlers:
                        raise InputError('Unknown host service; native Codex tools are executed by app-server')
                    if not isinstance(arguments, dict):
                        raise InputError('Tool arguments must be a JSON object')
                    packet = handlers[name](**arguments)
                except InputError as error:
                    errors += 1
                    packet = rejected_input(error, rollout, arguments, errors)
                    success = False
                else:
                    success = True
                    # Reset only network wakeups; rejected-call counts remain audit-only.
                    network_consecutive = 0
                    network_error = None
                # Persist every rejection, including the fifth and later, before
                # replying. A failed transport reply must not repeat a handler.
                write_json(self.workspace/f'call_{call_index:04d}_result.json', packet)
                self.transport.reply(event['id'], dict(success=success,
                    contentItems=content_items(packet, images=success and name != 'pi05_infer',
                                               max_image_edge=self.image_max_edge)))
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
                write_json(self.workspace/f'turn_{completed_turns:02d}_completed.json', turn)
                completed_turns += 1
                failure = turn.get('error') or network_error
                if turn.get('status') == 'failed' and is_network_error(failure):
                    if rollout.phase == 'done':
                        return  # Native completion needs no further paid model response.
                    turn_id = self._network_continue(rollout, turn_id, failure,
                        network_consecutive, network_serial)
                    network_serial += 1
                    network_consecutive += 1
                    network_error = None
                    deadline = time.monotonic() + self.timeout
                    continue
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
                if (params.get('threadId') not in (None, self.thread_id)
                        or params.get('turnId') not in (None, turn_id)):
                    continue
                if is_network_error(params.get('error', params)):
                    network_error = params.get('error', params)
                if method == 'error' and params.get('willRetry') is True:
                    # App-server owns model transport retry; this does not
                    # repeat inference, a tool handler, or physical execution.
                    print(json.dumps(dict(event='codex_transport_retry',
                        error=params.get('error'), step_id=rollout.tick)), flush=True)
                    continue
                if network_error and is_network_error(params.get('error', params)):
                    if rollout.phase == 'done':
                        return
                    # Wait for turn/completed, or verify idle/failed via the timeout probe.
                    # Do not send continue while Codex still owns an active turn.
                    deadline = min(deadline, time.monotonic() + 30)
                    print(json.dumps(dict(event='codex_network_wait_terminal',
                        error=network_error, step_id=rollout.tick)), flush=True)
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
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--evaluation-method', default=evaluation_method(), choices=('pi05_plus_gpt','gpt_only'))
    parser.add_argument('--codex', required=True)
    parser.add_argument('--sim-port', type=int, default=19113)
    parser.add_argument('--student-port', type=int, default=18830)
    parser.add_argument('--max-decisions', type=int, default=180, help='0 disables the decision-count limit')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    student = worker = rollout = None
    def terminate(signum, frame):
        raise KeyboardInterrupt('Stopping only this owned rollout and Codex process')
    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.evaluation_method == 'pi05_plus_gpt':
            if args.checkpoint is None:
                raise ValueError('Hybrid evaluation requires an explicit checkpoint')
            from ..pi05_server.client import Pi05Client
            student = Pi05Client(args.student_port, args.checkpoint)
        worker = CodexPolicy(args.output/'codex_workspace', args.codex, method=args.evaluation_method)
        if args.evaluation_method == 'gpt_only':
            from ..robodojo_server.gpt_only_client import GPTOnlyTools
            rollout = GPTOnlyTools(args.output, args.task, sim_port=args.sim_port,
                seed=args.seed, max_decisions=args.max_decisions, prompt_sha256=worker.prompt_sha256)
        else:
            rollout = RoboDojoTools(args.output, args.task, student, sim_port=args.sim_port,
                seed=args.seed, max_decisions=args.max_decisions, prompt_sha256=worker.prompt_sha256)
        worker.run(rollout)
    except BaseException:
        write_json(args.output/'failure.json', dict(error=traceback.format_exc(),
            controller_version=CONTROLLER_VERSION,
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
