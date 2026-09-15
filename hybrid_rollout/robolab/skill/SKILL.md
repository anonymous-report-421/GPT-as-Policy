---
name: robolab-hybrid-rollout
description: Run a RoboLab simulation episode using pi05 proposals reviewed and optionally corrected through bounded EEF actions, recording observations and trajectories with explicit output paths.
---

# RoboLab hybrid rollout

Operate the authorized simulation episode as the combined GPT + pi05 policy.
The Python host starts a dedicated GPT-6-Astra/xhigh Codex thread and injects
this skill plus [the colleague's unchanged gate prompt](gate_prompt.md).
Read that gate prompt when using this skill outside the bundled host.
The host adds three blocking service tools to your normal Codex tools. Use
file/image inspection, code execution, calculation, planning and working notes
freely to reach a good decision. You are an agent, not a JSON-only reviewer.
The host keeps one connection to each service; you choose when to call them
and what reviewed action to execute.

Project and predecessor knowledge is in [teacher context](context/teacher_context.md).
Read [EEF geometry and tracking](context/eef_control.md) when computing poses
or diagnosing a correction. The colleague's full conversation is unavailable;
the supplied context distinguishes verified history from current observations.
`workspace.json` gives the interpreter, source and artifact paths. Use `NOTES.md`
for persistent working memory and `scratch/` for your scripts and derived images.

1. Call `robolab_start` with the requested task and explicit fresh output directory.
   The task-bound service resets once and returns two RGB images, measured
   proprio/EEF, observation paths, and the exact next-call path arguments.
2. Call `pi05_infer` using that latest `observation_path` and a fresh `output_dir`.
   Keep the original task instruction unchanged. The result contains a fresh
   H15 joint-position proposal, its file path, robot-only FK trajectory, and
   the previous executed result. No simulated object future is available.
3. Assess the last execution and this next proposal using the gate prompt.
   Call `robolab_execute` with the exact `proposal_path`, your structured
   `response`, and a fresh `output_dir`. Use the matching `request_id`.
   The response's `assessment` carries your task progress and gate evidence.
4. Inspect the returned after-execution RGB/proprio and repeat from step 2.
   Never execute another chunk or correction without a new pi05 inference.
   Retain same-episode context: goals, confirmed progress, observations, actual
   execution results and earlier mistakes. No human reviewer is in this loop.

The `next_call` object supplies exact path arguments to avoid long arrays and
path guessing. Image attachments are ordered main RGB then wrist RGB. Proposal
tool results reuse the observation already shown; they do not reattach images.
The `history_path` points to the durable transcript of actual execution. You
can read it, inspect raw proposal/execution NPZ and IK diagnostics, reopen old
images, or create crops/comparisons with your normal tools. The same history
also arrives incrementally in this persistent conversation. Native analysis
tools may run between any service calls; the required infer/execute ordering
applies to simulator interaction, not to analysis.

## Existing control contract

- `student`: execute a prefix of 1–15 steps from this fresh proposal.
- `edit`: 1–5 steps of the proposal's EEF trajectory plus a root-frame
  translation (norm <= 0.05 m), root-frame rotation vector (norm <= 0.35 rad),
  and `keep`/`open`/`closed` gripper override. Override starts immediately.
- `eef`: 1–5 steps toward an absolute target within 0.05 m and 0.35 rad of
  the current measured EEF; unit quaternion in **wxyz**, explicit boolean
  `gripper_closed`. Python recomputes bounded IK after each actual control ACK.
- `stop`: save an incomplete episode when continuing is unjustified. Do not
  use this as a substitute for the environment's success/timeout result.

EEF coordinates refer to `base_link_in_robot_root`, not a measured contact
point. Open-gripper nominal pad center is +0.1311 m along flange-local X;
this is a geometry reference, not proof of grasp or contact. Gripper command
closure > 0.5 is closed. A command alone does not prove the object was grasped.
The uniform response schema includes `edit` and `target` for all modes;
unused fields are ignored. Always provide a concise evidence-based `reason`.
请用简体中文输出公开的决策说明（`reason`、assessment 中的自由文本、进度说明和最终汇报），简述观测依据与动作目的，不输出私有思维链；JSON 字段名、枚举值、工具名、路径及原始任务指令保持不变。

The copied gate requires observed failure or misaligned next intent for any
edit/EEF correction. Uncertainty alone does not authorize takeover. Allow
student self-recovery and hand back when suitable. No rewind, reset-in-episode,
object truth, reward, replay or hypothetical simulator execution is permitted
for planning. Preserve host-owned records and executing source; your scripts
must not open another control connection or bypass the recorded service tools.

Finish when `rollout_finished` is true. Success and timeout automatically save
native action/observation chunks, control-source provenance, history and video;
the tool returns the result and output paths. An exhausted decision budget or
operator/model stop is incomplete, not a native task failure. Report native
success/timeout and paths briefly, then end your turn. If a tool rejects input
before execution, correct only the arguments; a transport/physics error is
fatal and must not trigger an action retry or a fresh episode.
