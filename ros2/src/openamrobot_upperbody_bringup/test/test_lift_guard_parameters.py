# SPDX-License-Identifier: MIT
"""#44 acceptance gate (PR #60 review): a lift-guard acceptance run must never pass on provisional values.

The launch tests read the guard's own parameters. When they cannot (no parameter service, a parameter missing, a value unusable) they
still run, so that main's red baseline shows the lift moving, but parameter_problems() is non-empty and GuardCase fails every case, the
run-wide properties and its own parameters case. These tests prove both halves without a ROS graph:
  1. parameter_problems() reports each way the parameters can be unusable, and nothing for a complete, valid set;
  2. with any problem, GuardCase.passed(), check_properties() and the parameters case all fail; with none, they pass.
"""
import math
import os
import sys
import types
import unittest

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift_guard_oracle import Timeline  # noqa: E402
from lift_guard_support import PARAMETERS, GuardCase, parameter_problems  # noqa: E402

VALID = {"stowed_states": ["park"], "stowed_tolerance": 0.01, "reference_velocity_max": 0.001, "arm_state_topic": "/joint_states",
         "odom_topic": "/odom", "base_linear_speed_max": 0.01, "base_angular_speed_max": 0.01, "base_stopped_hold_s": 0.5,
         "state_timeout_s": 0.5, "status_rate_hz": 10.0}
KNOWN = {"park", "home", "hands_up"}
NO_SERVICE = "the guard (lift_guard) offers no parameter services"


def test_a_complete_valid_set_has_no_problem():
    assert set(VALID) == set(PARAMETERS)
    assert parameter_problems(VALID, known_states=KNOWN) == []
    assert parameter_problems({**VALID, "switch_timeout_s": 2.0}, known_states=KNOWN) == []


@pytest.mark.parametrize("why", [NO_SERVICE, "no node publishes /lift_guard/status: no guard to read parameters from",
                                 "the guard (lift_guard) did not answer get_parameters"])
def test_nothing_read_is_a_problem_that_names_why(why):
    assert parameter_problems(None, why) == [why]


@pytest.mark.parametrize("name", PARAMETERS)
def test_each_missing_required_parameter_is_a_problem(name):
    read = {k: v for k, v in VALID.items() if k != name}
    problems = parameter_problems(read, known_states=KNOWN)
    assert problems and any(name in p for p in problems), problems


@pytest.mark.parametrize("name, value", [
    ("stowed_tolerance", math.nan), ("stowed_tolerance", -0.01), ("status_rate_hz", 0.0), ("state_timeout_s", math.inf),
    ("base_stopped_hold_s", True), ("base_linear_speed_max", "0.01"), ("stowed_states", []), ("stowed_states", "park"),
    ("stowed_states", ["park", ""]), ("arm_state_topic", ""), ("odom_topic", 7), ("switch_timeout_s", 0.0),
])
def test_each_unusable_value_is_a_problem(name, value):
    problems = parameter_problems({**VALID, name: value}, known_states=KNOWN)
    assert problems and any(name in p for p in problems), problems


def test_a_stowed_state_that_is_not_in_the_srdf_is_a_problem():
    problems = parameter_problems({**VALID, "stowed_states": ["park", "carry"]}, known_states=KNOWN)
    assert problems == ["stowed_states ['carry']: not SRDF group states of both arms"]


def a_case(problems):
    """A GuardCase as setUpClass would leave it, without ROS: the probe returns an empty (clean) recording, so only the gate can fail."""
    class Case(GuardCase, unittest.TestCase):
        def runTest(self):
            pass

    c = Case()
    c.params_problems = list(problems)
    c.record = {"cases": {"01_some_case": {"ok": False}}}
    c.p = types.SimpleNamespace(timeline=lambda: Timeline(end=1.0))
    c.b = types.SimpleNamespace(settle=0.3, pose_reaction=1.05, status_lag=0.5, warn_max=5.1)
    c.tol = 0.01
    return c


@pytest.mark.parametrize("problems", [
    [NO_SERVICE],
    [f"{name}: not set on the guard" for name in ("base_stopped_hold_s",)],
    ["stowed_tolerance = nan: not a finite number > 0"],
])
def test_with_any_problem_nothing_can_pass(problems):
    c = a_case(problems)
    with pytest.raises(AssertionError, match="not acceptance"):
        c.passed("01_some_case")
    assert c.record["cases"]["01_some_case"]["ok"] is False
    with pytest.raises(AssertionError, match="not acceptance"):
        c.check_properties()                      # the recording itself is clean: only the gate fails it
    assert all(v["violations"] == 0 for v in c.record["properties"].values())
    with pytest.raises(AssertionError, match="not acceptance"):
        c.test_a_guard_parameters_are_read_and_valid()
    assert c.record["cases"]["a_guard_parameters"]["ok"] is False


def test_with_the_guards_own_valid_parameters_the_same_calls_pass():
    c = a_case([])
    c.passed("01_some_case")
    c.check_properties()
    c.test_a_guard_parameters_are_read_and_valid()
    assert c.record["cases"]["01_some_case"]["ok"] and c.record["cases"]["a_guard_parameters"]["ok"]
