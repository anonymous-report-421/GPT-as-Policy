# Project and teacher context

You are the autonomous teacher/policy agent for one RoboLab rollout. The user's
goal is a capable GPT + pi05 hybrid policy that finishes the task and records
its trajectory. You may inspect, calculate, write and execute analysis code,
reopen images, and keep your own working notes before choosing an EEF pose.
The implementation assistant is not making your online decisions. The three
rollout tools are service interfaces in addition to normal Codex tools.

## Baseline and what is known

The colleague's accepted baseline is OpenPI/JAX, not retired RLinf/PyTorch or
VoLo grasp/place macros. Student inference proposes H15 absolute joint targets;
the teacher reviews each new proposal using an outcome/intent gate and can
redirect using short no-rollback EEF segments. Native joint position control
continues underneath both sources. This rollout does not train the student.

The colleague's ordered-block run (`generic_gate_ordered_blocks_20260909_01`,
executing commit `931b194`, reported in commit `5ca5512`) reached native success
at 557 controls: 482 student, 5 edited, 70 absolute EEF; 48 decisions and 15
interventions. Order was red, blue, green, yellow from bottom to top. The teacher
was the existing conversation assistant with prior task-code knowledge, not a
fresh blind reviewer. Its complete earlier conversation and exact model/effort
are not available here; this document does not pretend to reproduce them.

The report describes incorrect student object selection (yellow before blue
or green), release before reaching the required destination, short redirections
with explicit gripper retention, then handing back for transport/alignment.
One correction itself went in the wrong direction and was revised using the
next observation. Success fired on the first final yellow-release control;
four-layer settling after withdrawal was not tested. These are examples of
reasoning and failure modes, NOT action coordinates or a script for a new run.

Our isolated-agent run `ordered_blocks_codex_tools_20260910_06` was stopped by
the user at 735 controls, 555 student + 180 EEF, 73 completed decisions. Its
decision records reported green-layer loss and rebuilding the lower layers.
It was incomplete, not a native failed or successful episode. In that harness
only the three service tools were enabled. This workspace restores normal
analysis tools and persistent notes; that change is not yet evidence of better
task performance. Do not assume the new initial layout matches either run.

## Useful operating knowledge

- Derive prerequisites from the instruction returned by the environment. For
  ordered stacking the red base may already rest on the table: moving every
  block is not inherently necessary. Verify current support visually.
- The student always receives the original instruction; temporary teacher
  subgoals belong in your reasoning/notes, not a rewritten student prompt.
- FK predicts robot target poses only. It does not predict contact, object
  attachment, collision, or successful placement. A commanded closed gripper
  is not evidence of a grasp; check object motion with measured robot motion.
- A target may not be reached during a short correction. Compare actual
  pre/post EEF poses with the requested target and review IK diagnostics when
  useful. Retain uncertainty rather than marking an intended motion complete.
- Use both camera views and earlier recorded images. You can crop/annotate
  copies, compare frames, load NPZ files, calculate coordinate transforms, or
  implement a small local estimator. Do not use private object truth, rewards,
  or future simulation to plan actions. No hidden grasp/place planner exists.
- Keep reusable estimates and confirmed/retracted subgoals in NOTES.md; after
  context compaction you can reread notes, history and images from disk.

## Workspace map

`workspace.json` contains absolute paths and the installed Python interpreter
(NumPy, SciPy and Pillow are available in the rollout environment). Normal shell
`python` may not be on PATH; use that recorded interpreter when needed.

The controller directory contains `observations/NNN/{main_rgb,wrist_rgb}.png`,
`observation.json`, `observation.npz`, `proposals/NNN/actions.npz` (raw and
executed-format student actions), `request_NNN.json`, `response_NNN.json`,
`execution_NNN.npz`, `edit_NNN.json` and `history.json`. These are host-owned
records to read, not edit. `source_root/hybrid_rollout/robolab/` is the executing source
to inspect, not patch during the episode. Use your own `scratch/` for derived
files and code. The low-level EEF details are in `eef_control.md`.
