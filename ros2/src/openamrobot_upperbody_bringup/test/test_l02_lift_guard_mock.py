# SPDX-License-Identifier: MIT
"""L02 (#44): on `mock`, the lift moves only while both arms are stowed, their state is fresh and the base reports stopped; otherwise
lift_controller is inactive and no command path moves the lift.

Black box: the stack is `moveit.launch.py sim:=mock`, with the lift guard however it is built. The test reads the guard's status
(/lift_guard/status), lift_controller's state (/controller_manager/list_controllers) and lift_joint on /joint_states, and commands the lift
through both of its paths (the FollowJointTrajectory action and the /lift_controller/joint_trajectory topic). It never reads an action result as
evidence of motion (#44 acceptance 4, #49).

Cases, in order (each one leaves both arms in park and the guard reporting OK for the next):
  1  the guard starts blocked, then allows once the start-up has put both arms in park; the lift has not moved on its own
  2  allowed: a lift goal moves the lift (the positive control: without it every "did not move" below could pass on a dead stack)
  3  left arm out of park: blocked with ARM_NOT_STOWED_LEFT; the action goal is rejected and a topic trajectory does not move the lift;
     after the arm is parked again the topic trajectory is not replayed
  4  activating lift_controller from outside while blocked: the guard deactivates it again
  5  an arm trajectory still running within the pose tolerance: blocked with ARM_TRAJECTORY_ACTIVE_LEFT (not a momentary zero velocity)
  6  the arm leaves park during a lift move: the goal ends, the lift stops and holds, and nothing resumes when the arm is parked again
  7  arm joint state stops (joint_state_broadcaster deactivated): blocked with ARM_STATE_STALE or ARM_STATE_MISSING
  8  /odom reports the base moving: blocked with BASE_MOVING
  9  /odom stops: ODOM_STALE or ODOM_MISSING (skipped until the mock odom stub can be turned off; question on #44)

The thresholds come from the design's provisional defaults (lift_guard_support.PROVISIONAL); the firmware values (#20) replace them.

On main before the guard exists this test FAILs, by design: it is the red half of the integration branch (#44). Simulation only: it says
nothing about the firmware interlock (CTL-017).
"""
import os
import random
import sys
import threading
import time
import unittest

os.environ["ROS_DOMAIN_ID"] = str(random.randint(1, 100))
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import launch  # noqa: E402
import launch_testing  # noqa: E402
import launch_testing.actions  # noqa: E402
import pytest  # noqa: E402
import rclpy  # noqa: E402
from ament_index_python.packages import get_package_share_directory  # noqa: E402
from launch.launch_description_sources import PythonLaunchDescriptionSource  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_support import A7_ENABLED, SKIP_REASON, stack_alive, stack_healthy, write_run_sidecar  # noqa: E402
from lift_guard_support import (ALLOW_WITHIN_S, ARM_JOINTS, BLOCK_WITHIN_S, INTERFACE, LIFT_CONTROLLER, PROVISIONAL, GuardProbe,  # noqa: E402
                                lift_still_m, max_excursion, park_of, srdf_states)

PROFILE = "mock"
STARTUP_WITHIN_S = 120.0
EXPECTED_PROCESSES = ["robot_state_publisher", "ros2_control_node", "spawner", "move_group", INTERFACE["process_pattern"]]


@pytest.mark.launch_test
def generate_test_description():
    if not A7_ENABLED:
        return launch.LaunchDescription([launch.actions.ExecuteProcess(cmd=["sleep", "30"]), launch_testing.actions.ReadyToTest()]), {}
    moveit = os.path.join(get_package_share_directory("openamrobot_upperbody_bringup"), "launch", "moveit.launch.py")
    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(PythonLaunchDescriptionSource(moveit),
                                                launch_arguments={"sim": PROFILE, "rviz": "false"}.items()),
        launch_testing.actions.ReadyToTest(),
    ]), {}


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardMock(unittest.TestCase):
    record = {"profile": PROFILE, "interface": INTERFACE, "provisional": PROVISIONAL,
              "budgets_s": {"block_within": BLOCK_WITHIN_S, "allow_within": ALLOW_WITHIN_S}, "cases": {}}

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.p = GuardProbe(PROFILE, "l02_lift_guard")
        cls.states = srdf_states()
        cls.park = {s: park_of(cls.states, s) for s in ARM_JOINTS}
        cls.out = list(cls.park["left"])
        cls.out[3] = 1.5                       # left joint4 to 1.5 rad: SRDF hands_up's elbow, far outside any stowed tolerance

    @classmethod
    def tearDownClass(cls):
        cases = cls.record["cases"]
        ok = bool(cases) and all(c.get("ok") for c in cases.values())
        cls.record["status_log"] = [{"t": round(t - cls.p.status_log[0][0], 3), **s} for t, s in cls.p.status_log] if cls.p.status_log else []
        write_run_sidecar("L02", "lift_guard_mock", "#44 lift guard acceptance on mock", "PASS" if ok else "FAIL", cls.record)
        cls.p.close()
        rclpy.shutdown()

    # ---- helpers ----------------------------------------------------------------------------------------------------------------------
    def case(self, name, **fields):
        self.record["cases"].setdefault(name, {"ok": False}).update(fields)

    def passed(self, name):
        self.record["cases"][name]["ok"] = True

    def park_both(self):
        for side in ARM_JOINTS:
            self.p.arm_goal(side, self.park[side], seconds=2.0)

    def precondition_allowed(self):
        """Whether the guard allowed the lift before a case started. Recorded, not asserted: on main (no guard) the case must still run its
        physical check, so the red baseline fails because the lift MOVED, not because a status topic is missing."""
        return self.p.wait_allowed() is not None

    def assert_allowed(self, what):
        took = self.p.wait_allowed()
        self.assertIsNotNone(took, f"{what}: the guard did not allow the lift within {ALLOW_WITHIN_S:.1f} s "
                                   f"(status {self.p.current_status()}, {LIFT_CONTROLLER} {self.p.lift_controller_state()})")
        return took

    def assert_blocked(self, what, code=None):
        took = self.p.wait_blocked(code)
        self.assertIsNotNone(took, f"{what}: not blocked{' with ' + code if code else ''} within {BLOCK_WITHIN_S:.1f} s "
                                   f"(status {self.p.current_status()}, {LIFT_CONTROLLER} {self.p.lift_controller_state()})")
        return took

    def assert_lift_still(self, before, seconds, what):
        worst = max_excursion(self.p.sample_lift(seconds), before)
        self.assertLessEqual(worst, lift_still_m(PROFILE), f"{what}: lift_joint moved {worst:.4f} m (from {before:.4f})")
        return worst

    # ---- cases ------------------------------------------------------------------------------------------------------------------------
    def test_l02_1_starts_blocked_then_allows_once_both_arms_are_parked(self):
        self.case("1_startup")
        self.assertTrue(self.p.wait_for_joints(STARTUP_WITHIN_S), "/joint_states never listed lift_joint and all 14 arm joints")
        lift0 = self.p.lift()
        got = self.p.wait_for(lambda: self.p.current_status() is not None, 30.0)
        self.assertIsNotNone(got, f"no message on {INTERFACE['status_topic']} within 30 s of the joint states (transient-local QoS)")
        first = self.p.status_log[0][1]
        self.case("1_startup", first_status=first, lift_at_start=lift0)
        # mock starts the arms at home (all zeros); the design moves them to park at start-up. Until that has happened the lift is blocked.
        self.assertFalse(first["ok"] and first["controller"] == "active",
                         f"the first status already allowed the lift before the arms were confirmed in park: {first}")
        took = self.p.wait_allowed(STARTUP_WITHIN_S)
        self.assertIsNotNone(took, f"the guard never allowed the lift after start-up (status {self.p.current_status()})")
        for side in ARM_JOINTS:
            for j, v in zip(ARM_JOINTS[side], self.park[side]):
                self.assertAlmostEqual(self.p.joint(j), v, delta=PROVISIONAL["stowed_tolerance_rad"], msg=f"{j} not in park at start-up")
        self.assertLessEqual(abs(self.p.lift() - lift0), lift_still_m(PROFILE), "the lift moved during start-up without a lift command")
        self.case("1_startup", allowed_after_s=round(took, 2))
        self.passed("1_startup")

    def test_l02_2_allowed_a_lift_goal_moves_the_lift(self):
        self.case("2_positive_control", allowed_before=self.precondition_allowed())
        l0 = self.p.lift()
        accepted, handle, seconds = self.p.lift_goal(l0 + 0.03)
        self.assertTrue(accepted, "the lift goal was rejected while the guard reported OK")
        self.p.result(handle, seconds + 15.0)
        moved = self.p.lift() - l0
        self.case("2_positive_control", lift_from=l0, moved_m=round(moved, 4))
        self.assertGreater(moved, 0.025, f"allowed lift goal moved the lift only {moved:.4f} m of 0.03")
        accepted, handle, seconds = self.p.lift_goal(l0)
        self.p.result(handle, seconds + 15.0)
        self.passed("2_positive_control")

    def test_l02_3_arm_out_blocks_both_command_paths_and_nothing_replays(self):
        self.case("3_arm_out", allowed_before=self.precondition_allowed())
        self.p.arm_goal("left", self.out, seconds=2.0)
        took = self.p.wait_blocked("ARM_NOT_STOWED_LEFT")
        l0 = self.p.lift()
        accepted, handle, seconds = self.p.lift_goal(l0 + 0.03)
        if accepted:
            self.p.result(handle, seconds + 10.0)
        self.p.lift_topic_command(self.p.lift() + 0.03, 2.0)
        worst = max_excursion(self.p.sample_lift(3.0), l0)
        self.case("3_arm_out", blocked_after_s=took, action_accepted=accepted, worst_m=round(worst, 4))
        # the physical check first, so that on main (no guard) this fails for the reason #44 exists
        self.assertLessEqual(worst, lift_still_m(PROFILE), f"lift_joint moved {worst:.4f} m with the left arm out of park (action + topic)")
        self.assertFalse(accepted, "lift_controller accepted a goal with the left arm out of park")
        self.assertIsNotNone(took, f"not blocked with ARM_NOT_STOWED_LEFT within {BLOCK_WITHIN_S:.1f} s: {self.p.current_status()}")
        self.p.arm_goal("left", self.park["left"], seconds=2.0)
        self.assert_allowed("after parking the left arm again")
        replay = self.assert_lift_still(l0, 4.0, "after re-activation (no replay of the topic trajectory)")
        self.case("3_arm_out", replay_worst_m=replay)
        self.passed("3_arm_out")

    def test_l02_4_activating_the_controller_from_outside_while_blocked_is_undone(self):
        self.case("4_external_activation", allowed_before=self.precondition_allowed())
        self.p.arm_goal("left", self.out, seconds=2.0)
        blocked = self.p.wait_blocked("ARM_NOT_STOWED_LEFT")
        switched = self.p.switch(activate=[LIFT_CONTROLLER])
        undone = self.p.wait_for(lambda: self.p.lift_controller_state() != "active", BLOCK_WITHIN_S)
        self.case("4_external_activation", blocked_before_switch=blocked is not None, switch_ok=switched, undone_after_s=undone)
        self.p.arm_goal("left", self.park["left"], seconds=2.0)
        self.assertIsNotNone(undone, f"{LIFT_CONTROLLER} stayed active for {BLOCK_WITHIN_S:.1f} s after an outside activation with the left "
                                     "arm out of park")
        self.assertIsNotNone(blocked, "the guard was not blocking before the outside activation")
        self.assert_allowed("after parking the left arm again")
        self.passed("4_external_activation")

    def test_l02_5_an_arm_trajectory_still_running_blocks_even_inside_the_pose_tolerance(self):
        self.case("5_arm_trajectory_active", allowed_before=self.precondition_allowed())
        # 0.008 rad on joint1 over 4 s: the pose stays inside stowed_tolerance (0.01) the whole time, but the reference moves at 0.002 rad/s,
        # above reference_velocity_max (0.001). Only the trajectory condition can see this.
        nudged = list(self.park["left"])
        nudged[0] += 0.008
        n0 = len(self.p.status_log)
        accepted, handle = self.p.arm_goal("left", nudged, seconds=4.0, wait=False)
        self.assertTrue(accepted, "left_arm_controller rejected the nudge")
        self.p.result(handle, 20.0)
        codes = {c for _, s in self.p.status_log[n0:] for c in s["blocking"]}
        self.case("5_arm_trajectory_active", codes_seen=sorted(codes))
        self.p.arm_goal("left", self.park["left"], seconds=4.0)
        self.assertIn("ARM_TRAJECTORY_ACTIVE_LEFT", codes, "a running left-arm trajectory inside the pose tolerance never blocked the lift")
        self.assert_allowed("after the nudge back to park")
        self.passed("5_arm_trajectory_active")

    def test_l02_6_arm_leaves_park_mid_motion_the_lift_stops_holds_and_does_not_resume(self):
        self.case("6_mid_motion", allowed_before=self.precondition_allowed())
        l0 = self.p.lift()
        target = l0 + 0.06
        accepted, handle, seconds = self.p.lift_goal(target)
        self.assertTrue(accepted, "the lift goal was rejected before the arm moved (case precondition)")
        time.sleep(1.0)
        t_out = time.monotonic()
        self.p.arm_goal("left", self.out, seconds=1.0, wait=False)
        blocked = self.p.wait_blocked("ARM_NOT_STOWED_LEFT")
        blocked_after = None if blocked is None else round(time.monotonic() - t_out, 2)
        status, _ = self.p.result(handle, seconds + 10.0)
        time.sleep(1.0)
        stopped_at = self.p.lift()
        hold = max_excursion(self.p.sample_lift(3.0), stopped_at)
        self.case("6_mid_motion", goal_status=status, blocked_after_s=blocked_after, target=target, stopped_at=stopped_at,
                  hold_worst_m=round(hold, 4))
        self.assertLess(stopped_at, target - 0.005, f"the lift reached {stopped_at:.4f} m (target {target:.4f}) although the left arm left park")
        self.assertLessEqual(hold, lift_still_m(PROFILE), f"the lift did not hold after the stop: moved {hold:.4f} m")
        self.assertIsNotNone(blocked, "not blocked with ARM_NOT_STOWED_LEFT after the arm left park")
        self.p.arm_goal("left", self.park["left"], seconds=2.0)
        self.assert_allowed("after parking the left arm again")
        self.assert_lift_still(stopped_at, 4.0, "after re-activation (no resume of the old goal)")
        accepted, handle, seconds = self.p.lift_goal(l0)
        if accepted:
            self.p.result(handle, seconds + 15.0)
        self.passed("6_mid_motion")

    def test_l02_7_arm_joint_state_stopping_blocks(self):
        self.case("7_arm_state_stale", allowed_before=self.precondition_allowed())
        l0 = self.p.lift()
        self.assertTrue(self.p.switch(deactivate=["joint_state_broadcaster"]), "could not deactivate joint_state_broadcaster")
        try:
            took = self.p.wait_for(lambda: self.p.blocked()
                                   and bool({"ARM_STATE_STALE", "ARM_STATE_MISSING"} & set(self.p.current_status()["blocking"]))
                                   and self.p.lift_controller_state() != "active", BLOCK_WITHIN_S)
            accepted, handle, seconds = self.p.lift_goal(l0 + 0.03)
            if accepted:
                self.p.result(handle, seconds + 10.0)
            self.p.lift_topic_command(l0 + 0.06, 2.0)
            time.sleep(3.0)
        finally:
            self.p.switch(activate=["joint_state_broadcaster"])
        # lift_joint is read from /joint_states, which the broadcaster feeds: read it once the broadcaster is back
        time.sleep(1.0)
        moved = abs(self.p.lift() - l0)
        self.case("7_arm_state_stale", blocked_after_s=took, action_accepted=accepted, moved_m=round(moved, 4))
        self.assertLessEqual(moved, lift_still_m(PROFILE), f"the lift moved {moved:.4f} m while there was no arm state")
        self.assertFalse(accepted, "lift_controller accepted a goal with no arm state")
        self.assertIsNotNone(took, f"no ARM_STATE_STALE/MISSING block within {BLOCK_WITHIN_S:.1f} s of the arm state stopping")
        self.assert_allowed("after the arm state came back")
        self.passed("7_arm_state_stale")

    def test_l02_8_odom_reporting_base_motion_blocks(self):
        self.case("8_base_moving", allowed_before=self.precondition_allowed())
        l0 = self.p.lift()
        # The mock zero-odom stub keeps publishing and these messages interleave with it. "Base stopped" must hold CONTINUOUSLY for
        # base_stopped_hold_s, so one moving sample is enough to block.
        moving = Odometry()
        moving.header.frame_id, moving.child_frame_id = "odom", "base_footprint"
        moving.twist.twist.linear.x = 0.2
        stop = threading.Event()

        def publish():
            while not stop.is_set():
                moving.header.stamp = self.p.node.get_clock().now().to_msg()
                self.p.odom.publish(moving)
                time.sleep(0.02)
        publisher = threading.Thread(target=publish, daemon=True)
        publisher.start()
        try:
            took = self.p.wait_for(lambda: self.p.blocked() and "BASE_MOVING" in self.p.current_status()["blocking"]
                                   and self.p.lift_controller_state() != "active", BLOCK_WITHIN_S)
            time.sleep(max(0.0, 1.5 - (took or 0.0)))
            accepted, handle, seconds = self.p.lift_goal(l0 + 0.03)
            if accepted:
                self.p.result(handle, seconds + 10.0)
            worst = max_excursion(self.p.sample_lift(1.0), l0)
        finally:
            stop.set()
            publisher.join(2.0)
        self.case("8_base_moving", blocked_after_s=took, action_accepted=accepted, worst_m=round(worst, 4))
        self.assertLessEqual(worst, lift_still_m(PROFILE), f"the lift moved {worst:.4f} m while /odom reported the base moving")
        self.assertFalse(accepted, "lift_controller accepted a goal while /odom reported the base moving")
        self.assertIsNotNone(took, f"/odom at 0.2 m/s never blocked the lift with BASE_MOVING within {BLOCK_WITHIN_S:.1f} s")
        self.assert_allowed("after /odom went back to the stub's zero")
        self.passed("8_base_moving")

    @unittest.skip("waiting on #44: the mock zero-odom stub has no way to be stopped from a test yet")
    def test_l02_9_odom_stopping_blocks(self):
        pass


class TestProcessesAlive(unittest.TestCase):
    def test_no_process_of_the_launch_died_while_the_tests_were_running(self, proc_info):
        stack_alive(proc_info)


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_l02_processes_started_and_exited_cleanly(self, proc_info):
        stack_healthy(proc_info, expected=EXPECTED_PROCESSES, sidecar=("L02", "lift_guard_mock"))
