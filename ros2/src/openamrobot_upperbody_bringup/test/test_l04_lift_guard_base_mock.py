# SPDX-License-Identifier: MIT
"""L04 (#44): the lift guard on `mock`: the base condition (odometry).

mock has no base; the design starts a zero-odom stub with the guard. Decided on #44 (10 Nov), names confirmed 11 Nov: the stub takes
std_srvs/SetBool on /lift_guard_odom_stub/enable and velocity parameters linear_x / angular_z, mock only. Base motion goes through those
parameters (one source of /odom, as the guard sees on a real base), and a missing stub, parameter or switch FAILS the case: until 12 Nov
the test published /odom itself or skipped, so these cases could pass with no stub at all (PR #60 review). A NaN twist is injected: it
models a corrupted message, not a base. The thresholds are read from the guard
(base_linear_speed_max, base_angular_speed_max, base_stopped_hold_s); the hold time is measured from the last moving sample seen on /odom.

Cases (each from: both arms parked, lift at 0.10 m)
  01 linear speed at 0.8x the threshold stays allowed and the lift moves; at 1.2x it blocks with BASE_MOVING and neither path moves the lift
  02 the same for angular speed
  03 a NaN twist blocks (a non-finite reading never counts as stopped)
  04 hold time: after the last moving sample the guard allows no sooner than base_stopped_hold_s, and not much later
  05 arm out and base moving together: both codes; the base stopping leaves the arm's code alone
  06 odometry going missing (the stub switched off): ODOM_STALE or ODOM_MISSING; the stub's switch is required
  zz the safety properties P1-P6 over the whole run

On main before the guard exists this FAILs by design. Simulation only.
"""
import math
import os
import random
import sys
import time
import unittest

os.environ["ROS_DOMAIN_ID"] = str(random.randint(1, 100))
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import launch  # noqa: E402
import launch_testing  # noqa: E402
import launch_testing.actions  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_support import A7_ENABLED, SKIP_REASON, stack_alive, stack_healthy  # noqa: E402
from lift_guard_support import INTERFACE, GuardCase, lift_still_m, stack_under_test  # noqa: E402

PROFILE = "mock"
EXPECTED_PROCESSES = ["robot_state_publisher", "ros2_control_node", "spawner", "move_group", INTERFACE["guard_executable"]]


@pytest.mark.launch_test
def generate_test_description():
    if not A7_ENABLED:
        return launch.LaunchDescription([launch.actions.ExecuteProcess(cmd=["sleep", "30"]), launch_testing.actions.ReadyToTest()]), {}
    stack, _ = stack_under_test(PROFILE)
    return launch.LaunchDescription([stack, launch_testing.actions.ReadyToTest()]), {}


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardBaseMock(GuardCase, unittest.TestCase):
    PROFILE = PROFILE
    NODE = "l04_lift_guard"
    CHECK_ID, SLUG, NAME = "L04", "lift_guard_base_mock", "#44 lift guard on mock: base and odometry"

    def threshold_pair(self, name, linear):
        limit = float(self.params[name])
        rows = {"threshold": limit}
        # just below: every sample under the threshold, so nothing may block, and the lift must move
        with self.p.base_motion(linear=0.8 * limit if linear else 0.0, angular=0.0 if linear else 0.8 * limit) as how:
            rows["method"] = how["method"]
            time.sleep(0.5)
            rows["below_stays_allowed"] = self.p.stays(
                lambda: (self.p.current_status() or {}).get("ok") and self.p.lift_controller_state() == "active", 2.0)
            l0 = self.p.lift()
            accepted, _ = self.p.lift_goal_and_wait(l0 + 0.01)
            rows["below_moved_m"] = round(self.p.lift() - l0, 4)
        self.p.wait_allowed(self.b.allow_within)
        # just above
        with self.p.base_motion(linear=1.2 * limit if linear else 0.0, angular=0.0 if linear else 1.2 * limit):
            rows["above_blocked_after_s"] = self.p.wait_blocked("BASE_MOVING", timeout=self.b.base_reaction)
            accepted, t0 = self.try_lift_both_paths(0.03)
            rows["above_action_accepted"] = accepted
            rows["above_moved_m"] = round(self.p.lift_motion_since(t0), 5)
        rows["allowed_again_after_s"] = self.p.wait_allowed(self.b.allow_within)
        return rows

    def check_pair(self, rows, what):
        self.assertLessEqual(rows["above_moved_m"], lift_still_m(PROFILE), f"{what} at 1.2x: the lift moved {rows['above_moved_m']} m")
        self.assertFalse(rows["above_action_accepted"], f"{what} at 1.2x: a lift goal was accepted")
        self.assertIsNotNone(rows["above_blocked_after_s"], f"{what} at 1.2x the threshold did not block with BASE_MOVING")
        self.assertTrue(rows["below_stays_allowed"], f"{what} at 0.8x the threshold did not stay allowed")
        self.assertGreater(rows["below_moved_m"], 0.008, f"{what} at 0.8x: the lift did not move")
        self.assertIsNotNone(rows["allowed_again_after_s"], f"{what}: not allowed again after the base stopped")

    def test_l04_01_linear_speed_threshold(self):
        name = "01_linear"
        self.start(name)
        rows = self.threshold_pair("base_linear_speed_max", linear=True)
        self.case(name, **rows)
        self.check_pair(rows, "linear speed")
        self.passed(name)

    def test_l04_02_angular_speed_threshold(self):
        name = "02_angular"
        self.start(name)
        rows = self.threshold_pair("base_angular_speed_max", linear=False)
        self.case(name, **rows)
        self.check_pair(rows, "angular speed")
        self.passed(name)

    def test_l04_03_a_nan_twist_blocks(self):
        name = "03_nan_twist"
        self.start(name)
        with self.p.inject_odom(linear=math.nan):
            took = self.p.wait_blocked(any_of=("BASE_MOVING", "ODOM_STALE", "ODOM_MISSING"), timeout=self.b.base_reaction)
            accepted, t0 = self.try_lift_both_paths(0.03)
            moved = self.p.lift_motion_since(t0)
        self.case(name, blocked_after_s=took, action_accepted=accepted, moved_m=round(moved, 5))
        self.assertLessEqual(moved, lift_still_m(PROFILE), f"the lift moved {moved:.4f} m with a NaN twist on /odom")
        self.assertIsNotNone(took, "a NaN twist on /odom did not block: a non-finite reading must never count as stopped")
        self.passed(name)

    def test_l04_04_hold_time_after_the_base_stops(self):
        name = "04_hold_time"
        self.start(name)
        hold = float(self.params["base_stopped_hold_s"])
        lin_max, ang_max = float(self.params["base_linear_speed_max"]), float(self.params["base_angular_speed_max"])
        t_start = time.monotonic()
        with self.p.base_motion(linear=1.2 * lin_max) as how:
            self.p.wait_blocked("BASE_MOVING", timeout=self.b.base_reaction)
            time.sleep(1.0)
        took = self.p.wait_allowed(hold + self.b.allow_within)
        t_allowed = time.monotonic()
        t_last = self.p.last_moving_odom(lin_max, ang_max, after=t_start)
        since_last = None if took is None or t_last is None else round(t_allowed - t_last, 3)
        self.case(name, method=how["method"], hold_s=hold, allowed_after_last_moving_sample_s=since_last)
        self.assertIsNotNone(t_last, "no moving sample was ever seen on /odom (precondition)")
        self.assertIsNotNone(since_last, "never allowed after the base stopped")
        self.assertGreaterEqual(since_last, hold - 0.05, f"allowed {since_last:.2f} s after the last moving sample, before the hold of "
                                                       f"{hold:.2f} s")
        self.assertLessEqual(since_last, hold + self.b.allow_within, f"allowed only {since_last:.2f} s after the base stopped")
        self.passed(name)

    def test_l04_05_arm_out_and_base_moving_together(self):
        name = "05_arm_and_base"
        self.start(name)
        far = self.p.park_side("left")
        far[3] = 1.5
        self.p.arm_goal("left", far, seconds=1.5)
        with self.p.base_motion(linear=0.2):
            both = self.p.wait_for(lambda: self.p.blocked("BASE_MOVING") and self.p.blocked("ARM_NOT_STOWED_LEFT"), self.b.base_reaction)
        arm_only = self.p.wait_for(lambda: self.p.blocked("ARM_NOT_STOWED_LEFT") and not self.p.blocked("BASE_MOVING"),
                                   float(self.params["base_stopped_hold_s"]) + self.b.allow_within)
        still_inactive = self.p.lift_controller_state() != "active"
        self.p.arm_goal("left", self.p.park_side("left"), seconds=1.5)
        allowed = self.p.wait_allowed(self.b.allow_within)
        self.case(name, both_after_s=both, arm_only_after_s=arm_only, still_inactive=still_inactive, allowed_after_s=allowed)
        self.assertIsNotNone(both, "arm out and base moving did not report both codes")
        self.assertIsNotNone(arm_only, "after the base stopped the status did not keep ARM_NOT_STOWED_LEFT alone")
        self.assertTrue(still_inactive, "lift_controller was active with the left arm still out")
        self.assertIsNotNone(allowed, "not allowed after the arm was parked")
        self.passed(name)

    def test_l04_06_odometry_going_missing_blocks(self):
        name = "06_odom_missing"
        self.start(name)
        switched = self.p.odom_stub(False)
        self.case(name, stub_switch=switched)
        # Required since the names were confirmed on #44 (11 Nov); it used to skip here, which hid a missing stub (PR #60 review)
        self.assertIsNotNone(switched, f"the mock odometry stub has no switch {INTERFACE['odom_stub_enable_service']} (#44, 10 Nov)")
        try:
            took = self.p.wait_blocked(any_of=("ODOM_STALE", "ODOM_MISSING"), timeout=self.b.stale_reaction)
            accepted, t0 = self.try_lift_both_paths(0.03)
            moved = self.p.lift_motion_since(t0)
        finally:
            self.p.odom_stub(True)
        allowed = self.p.wait_allowed(self.b.allow_within)
        self.case(name, blocked_after_s=took, action_accepted=accepted, moved_m=round(moved, 5), allowed_again_after_s=allowed)
        self.assertLessEqual(moved, lift_still_m(PROFILE), f"the lift moved {moved:.4f} m with no odometry")
        self.assertIsNotNone(took, f"odometry stopping did not block with ODOM_STALE/ODOM_MISSING within {self.b.stale_reaction:.2f} s")
        self.assertIsNotNone(allowed, "not allowed again once odometry came back")
        self.passed(name)

    def test_l04_zz_safety_properties_held_throughout_the_run(self):
        self.check_properties()


class TestProcessesAlive(unittest.TestCase):
    def test_no_process_of_the_launch_died_while_the_tests_were_running(self, proc_info):
        stack_alive(proc_info)


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_l04_processes_started_and_exited_cleanly(self, proc_info):
        stack_healthy(proc_info, expected=EXPECTED_PROCESSES, sidecar=("L04", "lift_guard_base_mock"))
