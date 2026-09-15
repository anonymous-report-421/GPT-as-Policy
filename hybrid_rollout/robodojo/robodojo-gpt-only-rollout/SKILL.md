---
name: robodojo-gpt-only-rollout
description: Control one authorized RoboDojo dual-ARX-X5 evaluation episode directly from RGB and proprioception, without pi05 proposals or inference. Not the hybrid method or RoboLab.
---

# RoboDojo GPT-only policy agent

Operate as the autonomous GPT policy with persistent context, normal file/image
inspection, code execution, calculation, planning and notes. Two blocking
services are additional tools, not your entire toolset. The implementation
assistant does not choose your online actions. No pi05 model is available.

Read [teacher context](context/teacher_context.md) and the
[dual-arm EEF contract](context/eef_control.md). `workspace.json` supplies exact
paths and the Python interpreter. Keep confirmed progress and geometry estimates
in `NOTES.md`, scripts/crops in `scratch/`; recorded evidence is read-only.

Track `step_id / max_episode_steps` (`remaining_steps` left): reaching the limit
without native success is failure. RoboDojo reports native success separately
from a partial-credit score; partial credit or apparent visual completion is not
success. While `rollout_finished=false`, continue checking unmet conditions and
acting; when it becomes true, read the native outcome, which may also be failure.

The first observation supplies only this task's `task_context`. When
`requires_arm_return=true`, completion also requires both arms near their
episode-start end-effector positions and orientations. After the object-level
goal, release, safely retract and return, reserving steps for this finish.
Arm return never means resetting the simulator. `make_kong` instead checks its
tile-handling sequence; do not assume missing arm return explains non-success.

For container placement, clear the rim with both the object and fingers before
moving laterally; align above the opening, lower and release, then withdraw
upward before moving sideways. Choose clearance from the observed geometry.

1. Call `robodojo_start` once for the requested task and exact fresh output path.
   It returns head/left-wrist/right-wrist RGB, 14D proprio, measured EEF poses
   and `next_call`.
2. Choose actions from this observation, original task instruction and same-episode
   history. Call `robodojo_act` with the current observation path, matching
   `request_id`, brief visible evidence/action purpose in `reason`, your direct
   action and the exact fresh output directory. There is no proposal to approve.
3. Inspect the returned images and measured state, then repeat step 2. Native
   analysis tools may run between services; maintain working memory as useful.

Action mode:

- `eef` only: 1–5 steps toward explicit left/right poses, each within 5 cm and
  0.35 rad of the current measured EEF. Positions are environment-origin meters,
  quaternion is unit wxyz, `gripper_closed` is boolean. The same bounded local
  IK as the hybrid evaluation converts your targets after each real control ACK.
  Direct joint commands are unavailable. To hold one arm, give its measured pose and
  intended gripper command.

Three camera attachments may be teacher-only downscaled previews;
`images[].path` remains the original image, available through native image tools
when detail is needed. Raw recordings retain original resolution. History is
incremental in the persistent conversation and saved in `history.json`.

No student inference, learned policy, proposal replay, outcome/intent gate,
rollback, mid-episode reset, hidden planner, object truth, reward queries or
hypothetical physics for planning. Do not open a second simulator connection.
Use the original task instruction, not a substituted teacher subgoal.

Use English for every public decision explanation, progress update, working note
and final report; do not expose private chain-of-thought. Preserve protocol keys,
enums, paths and the original task instruction. When `rollout_finished=true`,
briefly report the native outcome and saved paths, then end. Low expected success
is not permission to stop early. Validation rejections execute nothing and can
be corrected; transport/physics errors are fatal to this attempt, not permission
to retry an uncertain action or reset.
