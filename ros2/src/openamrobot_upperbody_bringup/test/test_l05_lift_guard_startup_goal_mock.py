# SPDX-License-Identifier: MIT
"""L05 (#44): the start-up lift goal of `moveit.launch.py lift_position:=X` on `mock` does not move the lift before the guard allows it.

moveit.launch.py:51-70 sends one FollowJointTrajectory goal to lift_controller at start-up. On main it goes straight to the controller,
with the arms at home (all zeros), not park: a command path a guard that only fronts the action would never see (#44 baseline). The design
on #44 spawns lift_controller inactive and makes the start-up goal wait for /lift_guard/status OK.

Checked from the recording of the whole start-up:
  1  lift_joint does not move before the first OK status
  2  after the first OK, the lift reaches lift_position (the start-up goal is delayed, not lost)
  +  the run-wide safety properties P1-P6

On main this FAILs by design: the lift moves at start-up with no guard. Simulation only.
"""
import os
import random
import sys
import unittest

os.environ["ROS_DOMAIN_ID"] = str(random.randint(1, 100))
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import launch  # noqa: E402
import launch_testing  # noqa: E402
import launch_testing.actions  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_support import A7_ENABLED, SKIP_REASON, stack_alive, stack_healthy  # noqa: E402
from lift_guard_support import INTERFACE, LIFT_SPEED, GuardCase, lift_still_m, stack_under_test  # noqa: E402

PROFILE = "mock"
LIFT_POSITION = 0.05
EXPECTED_PROCESSES = ["robot_state_publisher", "ros2_control_node", "spawner", "move_group", INTERFACE["guard_executable"]]


@pytest.mark.launch_test
def generate_test_description():
    if not A7_ENABLED:
        return launch.LaunchDescription([launch.actions.ExecuteProcess(cmd=["sleep", "30"]), launch_testing.actions.ReadyToTest()]), {}
    stack, _ = stack_under_test(PROFILE, lift_position=LIFT_POSITION)
    return launch.LaunchDescription([stack, launch_testing.actions.ReadyToTest()]), {}


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardStartupGoal(GuardCase, unittest.TestCase):
    PROFILE = PROFILE
    NODE = "l05_lift_guard_startup"
    CHECK_ID, SLUG, NAME = "L05", "lift_guard_startup_goal_mock", "#44 lift guard and the lift_position start-up goal on mock"

    def test_l05_1_the_start_up_goal_waits_for_the_guard(self):
        name = "1_startup_goal"
        self.case(name, lift_position=LIFT_POSITION)
        self.assertTrue(self.joints_ok, "/joint_states never listed lift_joint and all 14 arm joints")
        reach = 1.5 * LIFT_POSITION / LIFT_SPEED + 30.0
        self.p.wait_for(lambda: any(s["ok"] for _, s in self.p.status_changes) or abs((self.p.lift() or 0.0) - LIFT_POSITION) < 0.002, 60.0)
        t_ok = next((t for t, s in self.p.status_changes if s["ok"]), None)
        if t_ok is not None:
            self.p.wait_for(lambda: abs((self.p.lift() or 0.0) - LIFT_POSITION) < 0.002, reach)
        tl = self.p.timeline()
        before = [p for t, p in tl.lift if t_ok is None or t < t_ok]
        moved_before = (max(before) - min(before)) if len(before) >= 2 else 0.0
        final = self.p.lift()
        self.case(name, first_ok_found=t_ok is not None, moved_before_first_ok_m=round(moved_before, 5), final_lift=final,
                  lift_samples_before_first_ok=len(before))
        self.assertGreaterEqual(len(before), 2, "lift_joint was not observed before the guard's first OK: nothing can be said about "
                                                "the start-up goal moving it")
        self.assertLessEqual(moved_before, lift_still_m(PROFILE), f"the start-up goal moved the lift {moved_before:.4f} m before the guard "
                                                               "ever allowed it")
        self.assertIsNotNone(t_ok, "the guard never allowed the lift after start-up")
        self.assertAlmostEqual(final, LIFT_POSITION, delta=0.002, msg="the start-up goal never reached lift_position after the guard allowed")
        self.passed(name)

    def test_l05_zz_safety_properties_held_throughout_the_run(self):
        self.check_properties()


class TestProcessesAlive(unittest.TestCase):
    def test_no_process_of_the_launch_died_while_the_tests_were_running(self, proc_info):
        stack_alive(proc_info)


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_l05_processes_started_and_exited_cleanly(self, proc_info):
        stack_healthy(proc_info, expected=EXPECTED_PROCESSES, sidecar=("L05", "lift_guard_startup_goal_mock"))
