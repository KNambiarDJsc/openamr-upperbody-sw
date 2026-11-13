# SPDX-License-Identifier: MIT
"""The #44 oracle (lift_guard_oracle.py) on synthetic timelines: it finds each kind of guard fault it exists for, and stays quiet on a
correct run. No ROS graph. A property checker that cannot fail is a test of nothing, so every property has a case that must fail."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift_guard_oracle import (ERROR, OK, WARN, Timeline, arm_deviation, evaluate, first_not_active_after,  # noqa: E402
                               no_motion_while_blocked, no_motion_while_inactive, step_intervals, stowed_deviation, unobserved)

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


def sampled(*points, rate=0.1, end=5.0):
    """A lift_joint stream at `rate` s between samples, following the step function `points` [(t, position)], to `end`."""
    out, t = [], 0.0
    while t <= end + 1e-9:
        out.append((round(t, 3), [p for tp, p in points if tp <= t][-1]))
        t += rate
    return out


def total(result):
    return sum(len(v) for v in result.values())


def test_a_correct_run_has_no_violations():
    assert total(evaluate(correct_run(), **ARGS)) == 0, evaluate(correct_run(), **ARGS)


def test_p1_lift_moving_while_the_guard_reports_error_is_found():
    tl = correct_run()
    tl.lift = [(t, p + (0.01 if 4.0 <= t <= 5.0 else 0.0)) for t, p in tl.lift]   # a jump in the middle of the blocked interval
    assert evaluate(tl, **ARGS)["P1_no_motion_while_blocked"]


def test_p1_motion_inside_the_settle_window_is_allowed():
    tl = Timeline(end=5.0, lift=sampled((0.0, 0.0), (1.1, 0.002)),
                  status=[(0.0, OK, "active", ()), (1.0, ERROR, "inactive", ("BASE_MOVING",))])
    assert no_motion_while_blocked(tl, settle=0.3, eps=0.0005) == []


def test_p2_lift_moving_with_the_controller_inactive_is_found():
    tl = Timeline(end=5.0, lift=sampled((0.0, 0.0), (3.0, 0.01)), controller=[(0.0, "inactive")])
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
    tl = Timeline(end=1.0, status=[(0.0, OK, "active", ("BASE_MOVING",)), (0.1, ERROR, "active", ()),
                                   (0.2, 1, "inactive", ("BASE_MOVING",)), (0.3, ERROR, "inactive", ("SOMETHING_NEW",)),
                                   (0.4, ERROR, "sleeping", ("BASE_MOVING",))])
    assert len(evaluate(tl, **ARGS)["P4_status_contract"]) == 5


def test_p6_warn_asks_nothing_of_the_controller():
    """PR #53 run (L03 idle): the spawner activated lift_controller while the guard was still switching on (now WARN, #44 10 Nov)."""
    tl = Timeline(end=5.0, status=[(1.0, WARN, "activating", ()), (2.0, WARN, "activating", ()), (2.1, OK, "active", ())],
                  controller=[(0.0, "inactive"), (1.5, "active")])
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"] == []


def test_p6_error_while_active_beyond_the_lag_is_found():
    tl = Timeline(end=5.0, status=[(1.0, ERROR, "inactive", ("BASE_MOVING",))], controller=[(0.0, "active")])
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"]


def test_p1_motion_under_warn_is_not_blocked_motion_but_p5_sees_the_misreport():
    tl = Timeline(end=5.0, lift=[(t / 10, 0.1 + 0.001 * t) for t in range(0, 40)],
                  status=[(0.0, WARN, "activating", ())], controller=[(0.0, "active")])
    result = evaluate(tl, **ARGS)
    assert result["P1_no_motion_while_blocked"] == []
    assert result["P5_status_matches_controller"]


def test_p1_motion_under_error_is_found():
    tl = Timeline(end=5.0, lift=[(t / 10, 0.1 + 0.001 * t) for t in range(0, 40)],
                  status=[(0.0, ERROR, "inactive", ("BASE_MOVING",))], controller=[(0.0, "inactive")])
    assert evaluate(tl, **ARGS)["P1_no_motion_while_blocked"]


def test_p4_warn_activating_is_the_switching_on_state():
    """#44, 10 Nov: nothing blocks, the switch is asked for, lift_controller not active yet: WARN, empty blocking, 'activating'."""
    tl = Timeline(end=1.0, status=[(0.0, ERROR, "inactive", ("BASE_MOVING",)), (0.1, WARN, "activating", ()), (0.2, OK, "active", ())],
                  controller=[(0.0, "inactive"), (0.15, "active")])
    assert evaluate(tl, **ARGS)["P4_status_contract"] == []


def test_p4_the_old_switching_on_forms_are_breaches():
    """#53 at b09a54a sent ERROR with no code while switching on; the first double sent OK while the controller was inactive; WARN must
    name 'activating' and carry no code."""
    tl = Timeline(end=1.0, status=[(0.1, ERROR, "inactive", ()), (0.2, OK, "inactive", ()), (0.3, WARN, "inactive", ()),
                                   (0.4, WARN, "activating", ("BASE_MOVING",)), (0.5, ERROR, "activating", ("BASE_MOVING",))])
    assert len(evaluate(tl, **ARGS)["P4_status_contract"]) == 5


def test_p7_a_long_warn_spell_is_found():
    tl = Timeline(end=20.0, status=[(1.0, WARN, "activating", ()), (12.5, OK, "active", ())], controller=[(0.0, "inactive")])
    assert evaluate(tl, **{**ARGS, "warn_max": 3.0})["P7_switching_on_is_brief"]


def test_p7_a_short_warn_spell_and_one_before_the_controller_is_loaded_are_fine():
    tl = Timeline(end=20.0, status=[(0.0, WARN, "activating", ()), (8.0, OK, "active", ()), (10.0, ERROR, "active", ("BASE_MOVING",)),
                                    (11.0, WARN, "activating", ()), (11.2, OK, "active", ())],
                  controller=[(7.9, "inactive"), (7.95, "active"), (10.1, "inactive"), (11.15, "active")])
    assert evaluate(tl, **{**ARGS, "warn_max": 3.0})["P7_switching_on_is_brief"] == []


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
    # The status is a step function since 12 Nov, so the guard's next status (ERROR at 1.45) is part of the scenario: without it, the OK
    # would hold, with the controller inactive, until the end of the run.
    tl = Timeline(end=5.0, status=[(1.0, OK, "inactive", ()), (1.45, ERROR, "inactive", ("ARM_NOT_STOWED_LEFT",))],
                  controller=[(0.0, "inactive"), (1.1, "active"), (1.4, "inactive")])
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"] == []


def test_p6_an_ok_in_flight_across_a_deactivation_is_not_a_breach():
    """12 Nov, the first run on the integrated stack (L02-03, 07, 10): WARN, OK while active, the deactivation event, an OK published
    before it arriving 4 ms after it, then ERROR 7 ms later. The mismatch lasted milliseconds; the per-status check flagged it."""
    tl = Timeline(end=5.0, status=[(0.942, WARN, "activating", ()), (0.995, OK, "active", ()), (1.0, OK, "active", ()),
                                   (1.007, ERROR, "inactive", ("ARM_TRAJECTORY_ACTIVE_RIGHT",))],
                  controller=[(0.0, "inactive"), (0.942, "active"), (0.996, "inactive")])
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"] == []


def test_p6_a_guard_that_keeps_saying_ok_after_the_deactivation_is_found_with_how_long():
    oks = [(1.0 + 0.1 * i, OK, "active", ()) for i in range(9)]               # 1.0 .. 1.8
    tl = Timeline(end=5.0, status=oks + [(1.9, ERROR, "inactive", ("ARM_NOT_STOWED_LEFT",))],
                  controller=[(0.0, "active"), (1.05, "inactive")])
    found = evaluate(tl, **ARGS)["P6_allowed_means_active"]
    assert len(found) == 1 and found[0]["status"] == "OK" and math.isclose(found[0]["for_s"], 0.85, abs_tol=1e-6), found


def test_p6_an_error_the_controller_follows_within_the_lag_is_not_a_breach_and_one_it_does_not_is():
    follows = Timeline(end=5.0, status=[(1.0, OK, "active", ()), (2.0, ERROR, "active", ("BASE_MOVING",))],
                       controller=[(0.0, "active"), (2.3, "inactive")])
    assert evaluate(follows, **ARGS)["P6_allowed_means_active"] == []
    stuck = Timeline(end=5.0, status=follows.status, controller=[(0.0, "active"), (2.8, "inactive")])
    found = evaluate(stuck, **ARGS)["P6_allowed_means_active"]
    assert len(found) == 1 and found[0]["status"] == "ERROR" and math.isclose(found[0]["for_s"], 0.8, abs_tol=1e-6), found


def test_p6_p1_p7_a_dead_guard_says_nothing():
    """L03: once the guard's process has exited, its last status is not a statement any more (Timeline.status_end)."""
    tl = Timeline(end=10.0, status=[(1.0, OK, "active", ())], controller=[(0.0, "active"), (2.2, "inactive")], status_end=2.0)
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"] == []
    tl.status_end = None                                       # the same OK, still held: a breach
    assert evaluate(tl, **ARGS)["P6_allowed_means_active"]
    warn = Timeline(end=30.0, status=[(1.0, WARN, "activating", ())], controller=[(0.0, "inactive")], status_end=2.0)
    assert evaluate(warn, **ARGS)["P7_switching_on_is_brief"] == []
    warn.status_end = None
    assert evaluate(warn, **ARGS)["P7_switching_on_is_brief"]
    blocked = Timeline(end=10.0, status=[(1.0, ERROR, "inactive", ("BASE_MOVING",))], controller=[(0.0, "active")],
                       lift=sampled((0.0, 0.10), (6.0, 0.13), end=10.0), status_end=2.0)
    assert evaluate(blocked, **ARGS)["P1_no_motion_while_blocked"] == []
    blocked.status_end = None
    assert evaluate(blocked, **ARGS)["P1_no_motion_while_blocked"]


# ---- P1 / P2 must SEE the lift (final review, 12 Nov) --------------------------------------------------------------------------------------
BLOCKED = dict(status=[(0.0, ERROR, "inactive", ("BASE_MOVING",))], controller=[(0.0, "inactive")])


def test_p1_p2_an_interval_in_which_the_lift_was_never_sampled_is_not_a_pass():
    """lift_motion() of a stream with no samples is 0.0: 'did not move' must not be said of what was not seen."""
    result = evaluate(Timeline(end=10.0, lift=[], **BLOCKED), **ARGS)
    for key in ("P1_no_motion_while_blocked", "P2_no_motion_while_inactive"):
        assert result[key] and result[key][0].get("unobserved"), (key, result[key])


def test_p1_p2_a_long_gap_in_the_lift_samples_is_found_even_when_samples_exist_on_both_sides():
    lift = [(t / 10, 0.1) for t in range(0, 40)] + [(8.0 + t / 10, 0.1) for t in range(0, 20)]      # 3.9 .. 8.0: silent
    found = evaluate(Timeline(end=10.0, lift=lift, **BLOCKED), **ARGS)["P1_no_motion_while_blocked"]
    assert len(found) == 1 and found[0]["unobserved"] and found[0]["gap_s"] > 4.0, found


def test_p1_p2_a_stream_without_long_gaps_passes_and_a_nan_is_not_a_sample():
    assert total(evaluate(Timeline(end=10.0, lift=sampled((0.0, 0.1), end=10.0), **BLOCKED), **ARGS)) == 0
    nans = [(t, math.nan) for t, _ in sampled((0.0, 0.1), end=10.0)]
    assert evaluate(Timeline(end=10.0, lift=nans, **BLOCKED), **ARGS)["P1_no_motion_while_blocked"]


def test_p1_p2_a_declared_blackout_excuses_exactly_its_span():
    lift = [(t / 10, 0.1) for t in range(0, 40)] + [(8.0 + t / 10, 0.1) for t in range(0, 20)]
    declared = Timeline(end=10.0, lift=lift, blackouts=[(3.8, 8.0)], **BLOCKED)
    assert total(evaluate(declared, **ARGS)) == 0
    too_short = Timeline(end=10.0, lift=lift, blackouts=[(3.8, 5.0)], **BLOCKED)           # 5.0 .. 8.0 is still unseen
    found = evaluate(too_short, **ARGS)["P1_no_motion_while_blocked"]
    assert len(found) == 1 and found[0]["unobserved"] and found[0]["from"] == 5.0 and found[0]["gap_s"] == 3.0, found


def test_p1_p2_a_blackout_does_not_excuse_motion_in_the_samples_that_exist():
    lift = [(t / 10, 0.1) for t in range(0, 40)] + [(8.0 + t / 10, 0.13) for t in range(0, 20)]      # moved during the blackout
    tl = Timeline(end=10.0, lift=lift, blackouts=[(3.8, 8.0)], **BLOCKED)
    assert evaluate(tl, **ARGS)["P1_no_motion_while_blocked"][0]["moved_m"] > 0.02


def test_unobserved_reports_coverage_for_the_sidecar():
    tl = Timeline(end=10.0, lift=sampled((0.0, 0.1), end=10.0), blackouts=[(4.0, 6.0)])
    assert [(x, y) for x, y, _ in unobserved(tl, 0.0, 10.0, max_gap=0.0)] == [(0.0, 4.0), (6.0, 10.0)]


def test_p1_p2_start_up_before_the_first_lift_sample_is_not_a_gap_but_a_gap_after_it_is():
    """Final batch, L04: the guard said ERROR from t=0, and lift_joint was first sampled 2.1 s later (the stack was still coming up)."""
    late = [(2.1 + t / 10, 0.1) for t in range(0, 70)]                 # 2.1 .. 9.0
    assert total(evaluate(Timeline(end=9.0, lift=late, **BLOCKED), **ARGS)) == 0
    broken = [(2.1 + t / 10, 0.1) for t in range(0, 20)] + [(7.0 + t / 10, 0.1) for t in range(0, 20)]     # silent 4.0 .. 7.0
    found = evaluate(Timeline(end=9.0, lift=broken, **BLOCKED), **ARGS)["P1_no_motion_while_blocked"]
    assert len(found) == 1 and found[0]["unobserved"] and found[0]["gap_s"] > 2.9, found


# ---- stowed means the guard's stowed_states, not park alone (#44, 12 Nov: robot rule is stowed OR carry) ---------------------------------
J = ["a", "b"]
PARK, CARRY = {"a": 0.0, "b": 0.0}, {"a": 1.0, "b": -0.5}


def test_stowed_deviation_with_one_pose_is_the_old_distance_to_park():
    for pos in ({"a": 0.0, "b": 0.0}, {"a": 0.03, "b": -0.01}, {"a": 1.0, "b": -0.5}):
        assert stowed_deviation(pos, [PARK], J) == arm_deviation(pos, PARK, J)


def test_stowed_deviation_is_the_distance_to_the_closest_stowed_pose():
    at_carry = {"a": 1.0, "b": -0.5}
    assert stowed_deviation(at_carry, [PARK], J) == 1.0                    # park only: carry is not stowed
    assert stowed_deviation(at_carry, [PARK, CARRY], J) == 0.0             # park or carry: it is
    assert math.isclose(stowed_deviation({"a": 1.02, "b": -0.5}, [PARK, CARRY], J), 0.02)
    assert math.isclose(stowed_deviation({"a": 0.5, "b": -0.25}, [PARK, CARRY], J), 0.5)      # between the two is neither


def test_stowed_deviation_never_reads_a_missing_or_non_finite_joint_as_stowed():
    assert stowed_deviation({"a": 0.0}, [PARK, CARRY], J) == math.inf
    assert stowed_deviation({"a": 0.0, "b": math.nan}, [PARK, CARRY], J) == math.inf
    assert stowed_deviation({"a": 0.0, "b": 0.0}, [], J) == math.inf       # no stowed pose known: nothing is stowed


def test_p3_follows_stowed_states_an_arm_at_a_declared_carry_pose_is_not_a_breach():
    """The recorder's per-arm deviation is stowed_deviation. An arm that sits at carry while the guard says OK is a breach if only park
    is stowed, and none once carry is declared: the oracle must not encode park as the permanent rule."""
    ok = dict(status=[(0.0, OK, "active", ())], controller=[(0.0, "active")], end=10.0)
    park_only = Timeline(arms=[(0.0, {"left": 0.0, "right": 0.0}), (3.0, {"left": stowed_deviation(CARRY, [PARK], J), "right": 0.0})], **ok)
    either = Timeline(arms=[(0.0, {"left": 0.0, "right": 0.0}), (3.0, {"left": stowed_deviation(CARRY, [PARK, CARRY], J), "right": 0.0})], **ok)
    assert evaluate(park_only, **ARGS)["P3_arm_out_is_blocked"], "park only: an arm at carry with the guard allowing is out"
    assert evaluate(either, **ARGS)["P3_arm_out_is_blocked"] == []


def test_first_not_active_after_is_a_step_function_of_the_controller():
    ctl = [(0.0, "active"), (5.0, "inactive"), (9.0, "active")]
    assert first_not_active_after(ctl, 4.0) == 5.0                          # went inactive later
    assert first_not_active_after(ctl, 6.0) == 6.0                          # already inactive at the crossing: zero, not None
    assert first_not_active_after(ctl, 9.5) is None                         # active from then on
    assert first_not_active_after([], 1.0) is None
    assert first_not_active_after([(2.0, "inactive")], 1.0) == 2.0           # nothing known before; the first sample after counts
    assert first_not_active_after([(0.0, "active"), (3.0, "activating")], 1.0) == 3.0
