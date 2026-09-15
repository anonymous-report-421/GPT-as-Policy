"""Check native Codex command execution without a model turn or simulator."""
import argparse
import json
from pathlib import Path

from .io import write_json
from .skill.run import agent_config, toml_value
from .skill.transport import StdioAppServer


def check(codex: str, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    agent = output/'agent'
    agent.mkdir()
    argv = [codex, 'app-server', '--stdio', '--strict-config']
    for key, value in agent_config(output, agent).items():
        argv += ['-c', key+'='+toml_value(value)]
    transport = StdioAppServer(argv, output)
    try:
        transport.request('initialize', dict(clientInfo=dict(name='rollout-tool-preflight', version='1'),
            capabilities=dict(experimentalApi=True)), 180)
        transport.notify('initialized', {})
        result = transport.request('command/exec', dict(
            command=['/bin/bash', '--noprofile', '--norc', '-c',
                'set -e; command -v bash python git rg; '
                'test -w .; touch native_tool_probe; '
                'python -c "import json; print(json.dumps(dict(calculation=7*8)))"'],
            cwd=str(agent), timeoutMs=15000), 30)
        write_json(output/'result.json', dict(model_turns=0, simulator_connections=0, command=result))
        if result.get('exitCode') != 0 or not (agent/'native_tool_probe').is_file():
            raise RuntimeError(f'Native Codex command preflight failed: {result}')
        print(json.dumps(dict(event='native_tool_preflight_passed', model_turns=0, output=str(output))), flush=True)
    finally:
        transport.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.codex, args.output)
