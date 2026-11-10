# SPDX-License-Identifier: MIT
"""The #44 oracle (lift_guard_oracle.py) on synthetic timelines: it finds each kind of guard fault it exists for, and stays quiet on a
correct run. No ROS graph. A property checker that cannot fail is a test of nothing, so every property has a case that must fail."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift_guard_oracle import (ERROR, OK, Timeline, arm_deviation, evaluate, no_motion_while_blocked,  # noqa: E402
                               no_motion_while_inactive, step_intervals)

CODES = ("ARM_NOT_STOWED_LEFT", "ARM_NOT_STOWED_RIGHT", "ARM_TRAJECTORY_ACTIVE_LEFT", "ARM_TRAJECTORY_ACTIVE_RIGHT",
         "ARM_STATE_MISSING", "ARM_STATE_STALE", "BASE_MOVING", "ODOM_MISSING", "ODOM_STALE", "SWITCH_FAILED")
ARGS = dict(tolerance=0.01, known_codes=CODES, settle=0.3, eps=0.0005, reaction=0.7, lag=0.5)


def correct_run():
    """Allowed and moving 0-2 s; the left arm leaves park at 2.0; the guard blocks at 2.2 and the lift stops; park again at 6.0; OK at 6.2."""
    tl = Timeline(end=10.0)
    t = 0.0
    while t <= 10.0 + 1e-9:
        tl.lift.append((t, 0.1 + 0.015 * min(t, 2.2)))
        left = 0.5 if 2.0 <= t < 6.0 else 0.0
        tl.arms.append((t, {"left": left, "right": 0.0}))
        blocked = 2.2 <= t < 6.2
        tl.status.append((t, ERROR if blocked else OK, "inactive" if blocked else "active",
                          ("ARM_NOT_STOWED_LEFT",) if blocked else ()))
        t = round(t + 0.1, 3)
    tl.controller = [(0.0, "active"), (2.25, "inactive"), (6.25, "active")]
    tl.marks = [(0.0, "case")]
    return tl


def total(result):
    return sum(len(v) for v in result.values())


def test_a_correct_run_has_no_violations():
    assert total(evaluate(correct_run(), **ARGS)) == 0, evaluate(correct_run(), **ARGS)


def test_p1_lift_moving_while_the_guard_reports_error_is_found():
    tl = correct_run()
    tl.lift = [(t, p + (0.01 if 4.0 <= t <= 5.0 else 0.0)) for t, p in tl.lift]   # a jump in the middle of the blocked interval
    assert evaluate(tl, **ARGS)["P1_no_motion_while_blocked"]


def test_p1_motion_inside_the_settle_window_is_allowed():
    tl = Timeline(end=5.0, lift=[(0.0, 0.0), (1.0, 0.0), (1.1, 0.002), (2.0, 0.002), (5.0, 0.002)],
                  status=[(0.0, OK, "active", ()), (1.0, ERROR, "inactive", ("BASE_MOVING",))])
    assert no_motion_while_blocked(tl, settle=0.3, eps=0.0005) == []


def test_p2_lift_moving_with_the_controller_inactive_is_found():
    tl = Timeline(end=5.0, lift=[(0.0, 0.0), (2.0, 0.0), (3.0, 0.01), (5.0, 0.01)], controller=[(0.0, "inactive")])
    assert no_motion_while_inactive(tl, settle=0.3, eps=0.0005)


def test_p3_a_momentary_allow_while_the_arm_is_out_is_found():
    tl = correct_run()
    tl.status = [(t, OK, "active", ()) if 4.0 <= t <= 4.1 else s for s in tl.status for t in [s[0]]]
    r = evaluate(tl, **ARGS)["P3_arm_out_is_blocked"]
    assert r and r[0]["side"] == "left"


def test_p3_the_wrong_side_code_is_found():
    tl = correct_run()
    tl.status = [(t, lvl, ctl, ("ARM_NOT_STOWED_RIGHT",) if codes else ()) for t, lvl, ctl, codes in tl.status]
    assert evaluate(tl, **ARGS)["P3_arm_out_is_blocked"]


def test_p3_a_silent_guard_with_the_controller_left_active_is_found():
    tl = correct_run()
    tl.status = []
    tl.controller = [(0.0, "active")]
    assert evaluate(tl, **ARGS)["P3_arm_out_is_blocked"]


def test_p3_a_reaction_inside_the_window_is_not_a_violation():
    tl = correct_run()                                   # blocks 0.2 s after the arm leaves, reaction 0.7
    assert evaluate(tl, **ARGS)["P3_arm_out_is_blocked"] == []


def test_p3_a_slow_reaction_is_found():
    tl = correct_run()
    tl.status = [(t, OK, "active", ()) if 2.0 <= t < 3.5 else s for s in tl.status for t in [s[0]]]
    tl.controller = [(0.0, "active"), (3.55, "inactive"), (6.25, "active")]
    assert evaluate(tl, **ARGS)["P3_arm_out_is_blocked"]


def test_p4_status_contract_breaches_are_found():
    tl = Timeline(end=1.0, status=[(0.0, OK, "active", ("BASE_MOVING",)), (0.1, ERROR, "inactive", ()),
                                   (0.2, 1, "inactive", ("BASE_MOVING",)), (0.3, ERROR, "inactive", ("SOMETHING_NEW",)),
                                   (0.4, ERROR, "sleeping", ("BASE_MOVING",))])
    assert len(evaluate(tl, **ARGS)["P4_status_contract"]) == 5


def test_p5_status_disagreeing_with_the_manager_beyond_the_lag_is_found():
    tl = Timeline(end=5.0, status=[(2.0, OK, "active", ())], controller=[(0.0, "inactive")])
    assert evaluate(tl, **ARGS)["P5_status_matches_controller"]


def test_p5_disagreement_inside_the_lag_is_allowed():
    tl = Timeline(end=5.0, status=[(2.0, ERROR, "inactive", ("BASE_MOVING",))], controller=[(0.0, "active"), (2.2, "inactive")])
    assert evaluate(tl, **ARGS)["P5_status_matches_controller"] == []


def test_p6_ok_with_the_controller_inactive_is_found():
    tl = Timeline(end=5.0, status=[(1.0, OK, "inactive", ())], controller=[(0.0, "inactive")])
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"]


def test_p6_error_with_the_controller_still_active_is_found():
    tl = Timeline(end=5.0, status=[(1.0, ERROR, "active", ("BASE_MOVING",))], controller=[(0.0, "active")])
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"]


def test_arm_deviation_never_reads_a_nan_or_a_missing_joint_as_stowed():
    park = {"a": 0.0, "b": 0.1}
    assert arm_deviation({"a": 0.0, "b": 0.1}, park, ["a", "b"]) == 0.0
    assert arm_deviation({"a": 0.02, "b": 0.1}, park, ["a", "b"]) == 0.02
    assert arm_deviation({"a": float("nan"), "b": 0.1}, park, ["a", "b"]) == math.inf
    assert arm_deviation({"a": float("inf"), "b": 0.1}, park, ["a", "b"]) == math.inf
    assert arm_deviation({"b": 0.1}, park, ["a", "b"]) == math.inf


def test_step_intervals():
    ev = [(0.0, 0), (1.0, 1), (2.0, 1), (3.0, 0), (4.0, 1)]
    assert step_intervals(ev, lambda e: e[1] == 1, 5.0) == [(1.0, 3.0), (4.0, 5.0)]


def test_p3_an_outside_activation_undone_within_the_reaction_is_not_a_violation():
    tl = correct_run()
    tl.controller = [(0.0, "active"), (2.25, "inactive"), (4.0, "active"), (4.3, "inactive"), (6.25, "active")]
    assert evaluate(tl, **ARGS)["P3_arm_out_is_blocked"] == []


def test_p3_an_outside_activation_left_standing_is_found():
    tl = correct_run()
    tl.controller = [(0.0, "active"), (2.25, "inactive"), (4.0, "active"), (5.5, "inactive"), (6.25, "active")]
    assert evaluate(tl, **ARGS)["P3_arm_out_is_blocked"]


def test_p3_a_guard_that_never_deactivates_is_found_even_while_reporting_error():
    tl = correct_run()
    tl.controller = [(0.0, "active")]
    assert evaluate(tl, **ARGS)["P3_arm_out_is_blocked"]


def test_p6_a_short_activation_inside_the_lag_counts_even_if_it_is_over_again_at_the_end_of_the_window():
    tl = Timeline(end=5.0, status=[(1.0, OK, "inactive", ())], controller=[(0.0, "inactive"), (1.1, "active"), (1.4, "inactive")])
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"] == []
