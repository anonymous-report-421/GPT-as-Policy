# EEF control and geometry

This is a single DROID Panda arm with a Robotiq gripper. Public measurements
`current_eef.position` and `quaternion_wxyz` describe the Robotiq **base_link**
in robot-root coordinates. This is not the camera frame or the separate,
rotated `eef_frame`. Position units are meters; quaternions are **wxyz**.

The nominal open-gripper pad center is +0.1311 m along base_link-local X:

    pad_root = base_position + R(base_quaternion) @ [0.1311, 0, 0]

Conversely, for a desired nominal pad position and chosen orientation:

    base_target = desired_pad_root - R(target_quaternion) @ [0.1311, 0, 0]

This is a nominal geometry reference, not a measured contact/TCP at arbitrary
finger angles. A rotation also moves this offset point. Estimate object
locations from the allowed observations, not simulator object coordinates.
SciPy Rotation expects xyzw; reorder wxyz when using it. Root-frame rotation
deltas left-multiply the current rotation. You can write code to do these
calculations; you do not need to perform them mentally.

## Output modes

- `student`: select the first 1–15 rows of the latest student proposal. No
  targets change; the unused suffix is discarded.
- `eef`: give one fixed absolute base_link position, unit wxyz quaternion and
  boolean gripper command, for 1–5 controls. To express a relative move, compute
  the absolute target from the latest measured pose using your code. Omitted
  coordinates are not supported by the tool schema. Keep the measured
  quaternion if no rotation is intended.
- `edit`: modify the first 1–5 FK targets using a root-frame translation and
  rotation-vector delta. At row i=0..n-1 the offset scales by (i+1)/n. The
  gripper override (keep/open/closed) applies immediately, not gradually.

The full response also carries request_id, reason and outcome/intent assessment.
The schema includes edit and target fields for every mode; unused fields are
ignored. Final output is a call to robolab_execute, not a separately executed
shell script that sends actions.

## What happens after an EEF request

The requested absolute target must be within 0.05 m and 0.35 rad of the current
measured pose. The host then recomputes a local IK update after **each actual
control ACK**, using the fresh measured robot state and Jacobian:

    dq = J.T @ solve(J @ J.T + 0.05**2 * I, bounded_pose_error)

Per update, translation error is bounded to 0.02 m, rotation to 0.1 rad, each
joint increment to 0.05 rad and the native soft joint limits. The resulting
seven absolute joint targets and 0/1 gripper command go to the unchanged native
joint-position controller. There is no teleport, global path planner,
collision avoidance, or guaranteed arrival. One control is 1/15 s; five
controls are about 0.333 s, not five independent 5 cm displacements.

You receive RGB after the whole short segment, while low-level IK uses fresh
proprio inside it. Your target stays fixed during an `eef` segment. Between
segments another fresh pi05 inference is required even if you intend another
correction; you may do any useful file/image/code analysis between service calls.

Read `edit_NNN.json` for the executed targets, clipping and IK diagnostics.
Compare the post-execution measured pose with the target to compute remaining
translation/rotation errors. Linearized IK residual is not measured tracking
error. For example, run06 decision4 requested a 4.85 cm displacement but moved
only 2.58 cm in five controls. This illustrates why a valid ACK or small
kinematic residual does not prove arrival, contact or task progress.

Inspect the executing `hybrid_rollout/robolab/robolab_server/{eef_controller,
action_edit_kinematics,validation,client}.py` for exact implementation details.
They are readable from the source_root listed in workspace.json. No runtime
import or call to the retired rlinf_robolab pipeline is needed.
