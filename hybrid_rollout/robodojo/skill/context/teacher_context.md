# Task and predecessor knowledge

The goal is one recorded RoboDojo rollout of the task requested at launch, controlled
by a persistent GPT-6/xhigh agent on the selected backend and the official RoboDojo-finetuned
OpenPI/JAX pi05 policy. Derive the exact objective from the native instruction
and visible observations, not a hard-coded task script. This rollout does not train the student.

The control method is copied from the successful RoboLab full-agent pipeline:
fresh student inference, outcome/intent review, optional short EEF correction,
then fresh observation. The colleague's gate text is preserved verbatim. Full
normal Codex tools, writable notes and persistent episode context are retained.
The colleague's complete conversation is unavailable; equivalent capabilities
do not establish identical knowledge or task performance.

RoboLab's four-block mixed rollout reached native success at 510 controls,
with 425 student and 85 EEF steps in 48 decisions. This is evidence for the
agent workflow, not a RoboDojo task demonstration or student-only improvement.
Prior successes are not evidence that the current episode is complete.

Inspect images, recorded proprio, per-step proposal FK and actual execution
history. A gripper command does not prove grasp, and an IK target does not
prove arrival. Keep uncertain results uncertain; retract completed subgoals
if later visual evidence contradicts them. Use all three camera views,
calculation, crops and tracking diagnostics when helpful.

See `eef_control.md` for the X5 contract. `workspace.json` identifies the
Python interpreter, source and evidence paths. `observations/NNN/` contains
three RGB PNGs, observation JSON/NPZ; `proposals/NNN/actions.npz` stores raw
and native-format actions. `execution_NNN.npz`, `edit_NNN.json`, requests,
responses and `history.json` bind decisions to real control ACKs. These files
are read-only; use `NOTES.md` and `scratch/` for your own work.

All public reasoning summaries, assessment text, progress, working notes, and
final reports must be in English. Private chain-of-thought is never requested
or exported.
