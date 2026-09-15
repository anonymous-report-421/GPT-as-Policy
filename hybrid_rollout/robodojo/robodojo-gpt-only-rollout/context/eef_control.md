# RoboDojo / dual ARX X5 embodiment contract

You choose all actions. Normal shell, calculation, image viewing, notes and
analysis tools remain available. No pi05 calls, alternate controller connection,
object truth, reward query, rollback or future physics for planning.

- Robot: dual ARX X5, not Panda/DROID. Views are `cam_high`, `cam_left_wrist`,
  `cam_right_wrist`; reopen/crop recorded images with native tools as useful.
- Proprio/actions: `[left joints 1..6, left opening, right joints 1..6, right
  opening]`. Joints are radians. Native opening is continuous **0=closed, 1=open**.
  Gripper proprio is the last controller opening command, not a measured finger
  gap or proof of grasp. Arm joints and EEF poses are measured.
- `current_eef.left/right` refer to each arm's **link6 in environment-origin
  coordinates**, meters, quaternion **wxyz**. Both arms share these axes, not
  two rotated root frames. The reported Euler angles are degrees for inspection;
  commanded orientations remain unit wxyz.
- EEF `target` explicitly contains both arms with `position`,
  `quaternion_wxyz`, `gripper_closed`. True maps to opening=0. To hold an arm,
  give its current measured pose and intended gripper command. Per-arm targets
  remain within 0.05 m and 0.35 rad of the current measured pose.
- EEF mode executes 1–5 control steps. Local damped least-squares IK is recomputed
  after every real ACK: task increments ≤2 cm/0.1 rad, each joint increment
  ≤0.05 rad. No cuRobo planning or teleportation. A target is not proof of
  arrival/contact; inspect after-execution RGB and measured poses.
- Only EEF actions are available. No direct joint commands, student continuation,
  proposal suffix or learned action cache exists.
- Do not reuse DROID's +0.1311 m pad offset. Native X5 config reports
  `gripper_bias=0.145`, but that scalar alone does not establish an offset axis;
  derive grasp geometry from robot geometry and observations if needed.
- Native controls are 25 Hz (0.04 s). Physics uses upstream interpolation and
  control. Observations are returned at action-chunk boundaries, not each
  physics substep. All executed control observations are recorded.

Follow `next_call` paths exactly. Public outputs and notes use English; keys,
enums, paths and original task instruction are unchanged.
