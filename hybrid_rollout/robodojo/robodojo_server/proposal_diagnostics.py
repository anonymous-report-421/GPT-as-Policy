"""Descriptive arithmetic over a proposal, never a gate or safety verdict.

These replace the repeated shape/finite/range/difference calculations observed
in pilot 05. They describe the native ``actions`` array AFTER gripper clipping,
not ``raw_actions`` and not measured execution. The full proposal remains
available unchanged. No simulator, policy, model, filesystem or network access.
"""
import numpy as np


JOINT_COLUMNS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
OPENING_COLUMNS = (6, 13)  # left, right; continuous opening commands


def action_diagnostics(actions):
    """Report full-precision statistics without mutating or approving actions.

    The joint difference excludes measured current_state -> proposal row 0.
    Shape/finiteness validation for execution remains in the existing client;
    invalid arrays produce an unavailable diagnostic, not invented zero ranges.
    """
    array = np.asarray(actions)
    result = dict(
        schema='robodojo.action_diagnostics.v1',
        source='actions.npz:actions (after native gripper clipping)',
        scope='All proposal rows; not a safety, collision, intent, or success verdict.',
        shape=list(array.shape),
        finite=None,
        opening_range=None,
        max_joint_step_rad=None,
        opening_range_arm_order=['left', 'right'],
        joint_step_scope='Successive proposal rows only; excludes measured state to first row.',
        status='unavailable',
    )
    # Native pi05 proposals are floating point. Do not accept integer arithmetic
    # here, where subtraction could wrap silently before the finite check.
    if array.dtype.kind != 'f':
        return result
    result['finite'] = bool(np.isfinite(array).all())
    if array.ndim != 2 or array.shape[1] != 14 or len(array) < 2 or not result['finite']:
        return result
    # Keep the proposal dtype and the same arithmetic as the pilot's script.
    # Do not round, quantize, clip, cast to float32, or alter any source value.
    with np.errstate(over='ignore', invalid='ignore'):
        differences = np.abs(np.diff(array[:, JOINT_COLUMNS], axis=0))
    if not np.isfinite(differences).all():
        return result
    result.update(
        status='computed',
        opening_range=[[float(array[:, column].min()), float(array[:, column].max())]
                       for column in OPENING_COLUMNS],
        max_joint_step_rad=float(differences.max()),
    )
    return result
