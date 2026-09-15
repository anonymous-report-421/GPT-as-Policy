"""Give the policy agent a real working directory beside the host's audit files."""
import hashlib
from pathlib import Path
import shutil
import sys

from ..io import write_json


def prepare_workspace(audit, skill_root, *, method='pi05_plus_gpt'):
    audit, skill_root = Path(audit).resolve(), Path(skill_root).resolve()
    agent = audit/'agent'
    agent.mkdir()
    (agent/'scratch').mkdir()
    context = agent/'context'
    shutil.copytree(skill_root/'context', context)
    # A discoverable local skill, in addition to the host's explicit injection.
    direct = method == 'gpt_only'
    skill = agent/'.agents/skills'/('robodojo-gpt-only-rollout' if direct else 'robodojo-hybrid-rollout')
    skill.mkdir(parents=True)
    for name in (('SKILL.md',) if direct else ('SKILL.md', 'gate_prompt.md')):
        shutil.copy2(skill_root/name, skill/name)
    shutil.copytree(context, skill/'context')
    controller = audit.parent
    workspace = dict(
        controller_output=str(controller), source_root=str(skill_root.parent.parent), robot_profile='robodojo',
        python_executable=sys.executable, history_path=str(controller/'history.json'),
        observations_path=str(controller/'observations'), proposals_path=str(controller/'proposals'),
        execution_files=str(controller/'execution_NNN.npz'),
        eef_diagnostics_files=str(controller/'edit_NNN.json'),
        scratch=str(agent/'scratch'), notes_path=str(agent/'NOTES.md'),
        context_files={str(p.relative_to(agent)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in context.rglob('*') if p.is_file()},
        baseline_full_conversation_available=False)
    if direct:
        workspace.pop('proposals_path')
        workspace['evaluation_method'] = 'gpt_only'
    write_json(agent/'workspace.json', workspace)
    instructions = (
        '# RoboDojo policy agent workspace\n\n'
        'You are the autonomous policy agent, not a JSON-only reviewer. Use your normal '
        'Codex tools to inspect files and images, write/run code, calculate poses, and '
        'maintain notes. Three additional tools connect you to RoboDojo/pi05; they are '
        'not your entire toolset.\n\n'
        'Read context/teacher_context.md for project knowledge and context/eef_control.md '
        'for the robot contract. workspace.json supplies the Python interpreter and '
        'exact artifact/source paths. Keep working memory in NOTES.md and code, crops, '
        'plots, and comparisons in scratch/. Both are yours to edit.\n\n'
        'Use the injected rollout skill and unchanged gate for simulation control. '
        'Write every public explanation, assessment, progress update, note, and final '
        'report in English. Never expose private chain-of-thought. '
        'Preserve host-owned evidence and executing source; do not use shell scripts '
        'to open a second simulator connection, reset, or bypass the recorded control '
        'tools. Inspect recorded RGB/proprio/actions freely; private simulator scene '
        'snapshots/object truth are not planning inputs.\n')
    if direct:
        instructions = instructions.replace('Three additional tools connect you to RoboDojo/pi05',
            'Two additional tools connect you to RoboDojo (no pi05)')
        instructions = instructions.replace('injected rollout skill and unchanged gate', 'injected GPT-only rollout skill')
    (agent/'AGENTS.md').write_text(instructions)
    (agent/'NOTES.md').write_text(
        '# Episode working memory\n\n'
        'No observations yet. Maintain confirmed progress, current objective, geometry '
        'estimates, tracking errors, and corrections as useful. Historical runs are '
        'background knowledge, not observations of this episode.\n')
    return agent
