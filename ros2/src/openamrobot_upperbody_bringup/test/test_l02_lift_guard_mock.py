# SPDX-License-Identifier: MIT
"""L02 (#44): the lift guard on `mock`: the arms, the controller and the guard's own contract.

Black box. The stack is `moveit.launch.py sim:=mock` (or LIFT_GUARD_STACK_LAUNCH), with the guard however it is built. The test reads
/lift_guard/status, lift_controller's state (/controller_manager/activity, list_controllers), lift_joint and the arm joints on /joint_states,
and the guard's own parameters; it commands the lift through the action, the topic and MoveIt. An action result is never evidence of motion.

Every case starts from the same state (both arms in park, lift at 0.10 m, guard asked to allow) and checks the physical outcome before the
guard's report, so that on main (no guard) each case fails because the lift MOVED. The last test checks the run-wide safety properties
(lift_guard_oracle) over everything recorded, so a guard that allows the lift for a moment anywhere in the run fails even where no case looks.

Cases
  00 start-up: blocked first; lift_controller never active before the first OK; both arms in park and the lift unmoved when first allowed
  01 status contract: name, keys, publisher QoS (reliable, transient local), rate within 25 % of status_rate_hz
  02 positive controls: the action and the topic both move the lift while allowed (without them every "did not move" is vacuous)
  03 each of the 14 arm joints alone, out of park by 3x the tolerance: blocked with that arm's code, the action goal rejected
  04 the tolerance boundary, from the guard's own stowed_tolerance: 0.8x stays allowed and the lift moves; 1.2x blocks
  05 a trajectory still running INSIDE the tolerance (reference faster than reference_velocity_max): blocked, both arms
  06 an arm goal executing but holding still (zero velocity, at park): blocked (the action condition alone sees it)
  07 an arm trajectory sent on the controller's TOPIC: blocked (no action exists, only the reference check sees it)
  08 arm out: the action is rejected, the topic does not move the lift up or down, and nothing replays after the arm is parked again
  09 lift_controller activated from outside while blocked: deactivated again within the reaction budget, lift travel bounded, and every
     status in the window is ERROR naming the cause (#44, 10 Nov: "blocking naming the cause")
  10 an arm leaves park during a lift move, up (left) and down (right): the lift stops within the travel budget, holds, does not resume.
     Measured from the unsafe condition, not from a signal: from the arm command and from the joints' crossing of the tolerance, the time to
     the guard's ERROR, to lift_controller inactive, and the lift's travel (recorded; the budget is the test's own tolerance, not an
     accepted stopping limit)
  11 arm state stops (joint_state_broadcaster deactivated): blocked with ARM_STATE_STALE or ARM_STATE_MISSING
  12 injected arm state: fresh and parked allows; one joint absent from every message blocks (freshness per joint); a NaN blocks
  13 MoveIt: a lift request through move_group with an arm out does not move the lift
  14 two reasons at once: both arms out gives both codes; parking one leaves the other
  15 every stowed state the guard declares (stowed_states; park today, carry once Q7 adds it to the SRDF) allows the lift
  16 the wiring at runtime: one guard process, one odometry stub process, the stub the only /odom publisher, and publishing
  zz the safety properties P1-P7 held over the whole run

On main before the guard exists this FAILs by design (the red half of the #44 integration branch). Simulation only: it says nothing about
the firmware interlock (CTL-017).
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
import rclpy  # noqa: E402
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_support import A7_ENABLED, SKIP_REASON, stack_alive, stack_healthy  # noqa: E402
from lift_guard_oracle import ERROR, first_crossing, first_not_active_after  # noqa: E402
from lift_guard_support import (ALL_ARM_JOINTS, ARM_JOINTS, INTERFACE, LIFT, LIFT_CONTROLLER, START_HEIGHT, GuardCase,  # noqa: E402
                                joint_limits, lift_still_m, pids_running, srdf_states, stack_under_test)

PROFILE = "mock"
EXPECTED_PROCESSES = ["robot_state_publisher", "ros2_control_node", "spawner", "move_group", INTERFACE["guard_executable"]]


@pytest.mark.launch_test
def generate_test_description():
    if not A7_ENABLED:
        return launch.LaunchDescription([launch.actions.ExecuteProcess(cmd=["sleep", "30"]), launch_testing.actions.ReadyToTest()]), {}
    stack, _ = stack_under_test(PROFILE)
    return launch.LaunchDescription([stack, launch_testing.actions.ReadyToTest()]), {}


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardArmsMock(GuardCase, unittest.TestCase):
    PROFILE = PROFILE
    NODE = "l02_lift_guard"
    CHECK_ID, SLUG, NAME = "L02", "lift_guard_arms_mock", "#44 lift guard on mock: arms, controller, contract"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.limits = joint_limits()

    def out_pose(self, side, joint=None, offset=None):
        """Park with one joint moved by `offset` (default 3x the tolerance), away from that joint's nearer limit."""
        joint = joint or f"openarm_{side}_joint4"
        pose = self.p.park_side(side)
        i = ARM_JOINTS[side].index(joint)
        offset = 3 * self.tol if offset is None else offset
        lo, hi = self.limits[joint]
        pose[i] += offset if pose[i] + offset <= hi - 1e-3 else -offset
        assert lo <= pose[i] <= hi, f"{joint}: {pose[i]} outside [{lo}, {hi}]"
        return pose

    def far_out(self, side):
        """An unmistakably engaged arm: joint4 at 1.5 rad (SRDF hands_up's elbow)."""
        pose = self.p.park_side(side)
        pose[3] = 1.5
        return pose

    # ---- 00 start-up ------------------------------------------------------------------------------------------------------------------
    def test_l02_00_starts_blocked_and_allows_only_once_both_arms_are_parked(self):
        name = "00_startup"
        self.case(name)
        self.assertTrue(self.joints_ok, "/joint_states never listed lift_joint and all 14 arm joints")
        first_ok = self.p.wait_for(lambda: any(s["ok"] for _, s in self.p.status_changes), 30.0)
        tl = self.p.timeline()
        changes = list(self.p.status_changes)
        self.case(name, first_status=changes[0][1] if changes else None)
        self.assertTrue(changes, f"no message on {INTERFACE['status_topic']} since the launch")
        self.assertFalse(changes[0][1]["ok"], f"the first status already allowed the lift: {changes[0][1]}")
        self.assertIsNotNone(first_ok, "the guard never allowed the lift after start-up")
        t_ok = next(t for t, s in changes if s["ok"])
        active_before = [t for t, s in tl.controller if s == "active" and t < t_ok - 0.05]
        devs = [d for t, d in tl.arms if t <= t_ok]
        lift0 = [p for t, p in tl.lift if t <= t_ok]
        self.case(name, controller_active_before_first_ok=[round(t - tl.controller[0][0], 2) for t in active_before][:5],
                  arm_deviation_at_first_ok=devs[-1] if devs else None,
                  lift_motion_before_first_ok=(max(lift0) - min(lift0)) if lift0 else None)
        # Without an observation before the first OK, 'never active before it' would pass vacuously (seen once, 10 Nov, double run).
        self.assertTrue(any(t < t_ok for t, _ in tl.controller), f"{LIFT_CONTROLLER}'s state was never observed before the first OK")
        self.assertFalse(active_before, f"{LIFT_CONTROLLER} was active before the guard first allowed the lift (#44 acceptance 1: it "
                                        "starts inactive)")
        self.assertTrue(devs and max(devs[-1].values()) <= self.tol, f"first allowed with an arm out of park: {devs[-1] if devs else None}")
        self.assertTrue(not lift0 or max(lift0) - min(lift0) <= lift_still_m(PROFILE), "the lift moved during start-up")
        self.passed(name)

    # ---- 01 status contract -----------------------------------------------------------------------------------------------------------
    def test_l02_01_status_contract(self):
        name = "01_status_contract"
        self.start(name)
        s = self.p.current_status()
        self.assertIsNotNone(s, f"no status on {INTERFACE['status_topic']}")
        qos = self.p.status_publisher_qos()
        rate = self.p.status_rate(3.0)
        want = float(self.params["status_rate_hz"])
        self.case(name, status_name=s["name"], keys=s["keys"], publishers=[(n, int(d), int(r)) for n, d, r in qos], rate_hz=round(rate, 2))
        self.assertEqual(s["name"], INTERFACE["status_name"])
        for key in (INTERFACE["key_controller"], INTERFACE["key_blocking"]):
            self.assertIn(key, s["keys"], f"status has no {key!r} value")
        self.assertEqual(len(qos), 1, f"exactly one publisher expected on {INTERFACE['status_topic']}: {qos}")
        self.assertEqual(qos[0][1], DurabilityPolicy.TRANSIENT_LOCAL, "status publisher is not transient-local")
        self.assertEqual(qos[0][2], ReliabilityPolicy.RELIABLE, "status publisher is not reliable")
        self.assertLessEqual(abs(rate - want), 0.25 * want, f"status at {rate:.1f} Hz, status_rate_hz is {want}")
        self.passed(name)

    # ---- 02 positive controls ---------------------------------------------------------------------------------------------------------
    def test_l02_02_while_allowed_both_command_paths_move_the_lift(self):
        name = "02_positive_controls"
        self.start(name)
        l0 = self.p.lift()
        accepted, status = self.p.lift_goal_and_wait(l0 + 0.03)
        by_action = self.p.lift() - l0
        self.p.lift_topic_command(l0, 2.5)
        time.sleep(4.0)
        by_topic = self.p.lift() - (l0 + by_action)
        self.case(name, action_accepted=accepted, moved_by_action_m=round(by_action, 4), moved_by_topic_m=round(by_topic, 4))
        self.assertTrue(accepted, "the lift goal was rejected while the guard allowed the lift")
        self.assertGreater(by_action, 0.025, f"the allowed action moved the lift {by_action:.4f} m of 0.03")
        self.assertLess(by_topic, -0.025, f"the allowed topic trajectory moved the lift {by_topic:.4f} m of -0.03")
        self.passed(name)

    # ---- 03 every joint ---------------------------------------------------------------------------------------------------------------
    def test_l02_03_each_arm_joint_alone_out_of_park_blocks_with_its_arms_code(self):
        name = "03_each_joint"
        self.start(name)
        rows, failures = [], []
        for side in ARM_JOINTS:
            for joint in ARM_JOINTS[side]:
                self.p.mark(f"{name}:{joint}")
                self.p.arm_goal(side, self.out_pose(side, joint), seconds=1.0)
                took = self.p.wait_blocked(f"ARM_NOT_STOWED_{side.upper()}", timeout=self.b.pose_reaction)
                t0 = time.monotonic()
                accepted, handle, _ = self.p.lift_goal(self.p.lift() + 0.02)
                time.sleep(0.6)
                if accepted:
                    handle.cancel_goal_async()
                moved = self.p.lift_motion_since(t0)
                rows.append({"joint": joint, "blocked_after_s": None if took is None else round(took, 3), "accepted": accepted,
                             "moved_m": round(moved, 5)})
                if took is None or accepted or moved > lift_still_m(PROFILE):
                    failures.append(rows[-1])
                self.p.arm_goal(side, self.p.park_side(side), seconds=1.0)
                self.p.wait_allowed(self.b.allow_within)
        self.case(name, rows=rows)
        self.assertFalse(failures, f"{len(failures)} of 14 joints out of park did not block the lift: {failures}")
        self.passed(name)

    # ---- 04 tolerance boundary --------------------------------------------------------------------------------------------------------
    def test_l02_04_the_tolerance_boundary_comes_from_the_guards_own_parameter(self):
        name = "04_tolerance_boundary"
        self.start(name)
        rows = {}
        for side in ARM_JOINTS:
            joint = f"openarm_{side}_joint4"
            self.p.arm_goal(side, self.out_pose(side, joint, 0.8 * self.tol), seconds=1.0)
            time.sleep(0.5)
            inside_ok = self.p.stays(lambda: (self.p.current_status() or {}).get("ok") and self.p.lift_controller_state() == "active", 1.5)
            l0 = self.p.lift()
            accepted, _ = self.p.lift_goal_and_wait(l0 + 0.01)
            moved = self.p.lift() - l0
            self.p.arm_goal(side, self.out_pose(side, joint, 1.2 * self.tol), seconds=1.0)
            outside = self.p.wait_blocked(f"ARM_NOT_STOWED_{side.upper()}", timeout=self.b.pose_reaction)
            rows[side] = {"inside_stays_allowed": inside_ok, "inside_goal_accepted": accepted, "inside_moved_m": round(moved, 4),
                          "outside_blocked_after_s": outside}
            self.p.arm_goal(side, self.p.park_side(side), seconds=1.0)
            self.p.wait_allowed(self.b.allow_within)
        self.case(name, tolerance=self.tol, rows=rows)
        for side, r in rows.items():
            self.assertIsNotNone(r["outside_blocked_after_s"], f"{side}: 1.2x the tolerance did not block")
            self.assertTrue(r["inside_stays_allowed"], f"{side}: 0.8x the tolerance did not stay allowed")
            self.assertTrue(r["inside_goal_accepted"] and r["inside_moved_m"] > 0.008, f"{side}: at 0.8x the lift did not move: {r}")
        self.passed(name)

    # ---- 05 running inside the tolerance ----------------------------------------------------------------------------------------------
    def test_l02_05_a_trajectory_still_running_inside_the_tolerance_blocks(self):
        name = "05_running_inside_tolerance"
        self.start(name)
        vmax = float(self.params["reference_velocity_max"])
        offset = 0.8 * self.tol
        seconds = max(1.0, offset / (3.0 * vmax))           # the reference moves at about 3x reference_velocity_max
        rows = {}
        for side in ARM_JOINTS:
            n0 = len(self.p.status_changes)
            accepted, handle = self.p.arm_goal(side, self.out_pose(side, f"openarm_{side}_joint1", offset), seconds=seconds, wait=False)
            self.p.result(handle, seconds + 10.0)
            codes = sorted({c for _, s in self.p.status_changes[n0:] for c in s["blocking"]})
            rows[side] = {"accepted": accepted, "seconds": round(seconds, 2), "codes_seen": codes}
            self.p.arm_goal(side, self.p.park_side(side), seconds=seconds)
            self.p.wait_allowed(self.b.allow_within)
        self.case(name, rows=rows)
        for side, r in rows.items():
            self.assertIn(f"ARM_TRAJECTORY_ACTIVE_{side.upper()}", r["codes_seen"], f"{side}: a running trajectory inside the pose "
                          "tolerance never blocked (#44 acceptance 2: not a momentary zero velocity)")
        self.passed(name)

    # ---- 06 executing but holding still -----------------------------------------------------------------------------------------------
    def test_l02_06_an_arm_goal_executing_while_holding_still_blocks(self):
        name = "06_executing_holding_still"
        self.start(name)
        park = self.p.park_side("left")
        accepted, handle = self.p.arm_goal("left", None, wait=False, points=[(park, 2.0), (park, 5.0)])
        took = self.p.wait_blocked("ARM_TRAJECTORY_ACTIVE_LEFT", timeout=self.b.pose_reaction + 0.5)
        self.p.result(handle, 15.0)
        released = self.p.wait_allowed(self.b.allow_within)
        self.case(name, accepted=accepted, blocked_after_s=took, released_after_goal_s=released)
        self.assertTrue(accepted)
        self.assertIsNotNone(took, "a left-arm goal executing at park with zero velocity did not block (the action condition)")
        self.assertIsNotNone(released, "the guard did not allow again after the goal finished")
        self.passed(name)

    # ---- 07 arm trajectory on the topic -----------------------------------------------------------------------------------------------
    def test_l02_07_an_arm_trajectory_sent_on_the_controller_topic_blocks(self):
        name = "07_arm_topic"
        self.start(name)
        vmax = float(self.params["reference_velocity_max"])
        seconds = max(1.0, 0.8 * self.tol / (3.0 * vmax))
        n0 = len(self.p.status_changes)
        self.p.arm_topic_command("left", self.out_pose("left", "openarm_left_joint1", 0.8 * self.tol), seconds)
        time.sleep(seconds + 0.5)
        inside_codes = sorted({c for _, s in self.p.status_changes[n0:] for c in s["blocking"]})
        self.p.arm_topic_command("left", self.far_out("left"), 1.5)
        out = self.p.wait_blocked(any_of=("ARM_NOT_STOWED_LEFT", "ARM_TRAJECTORY_ACTIVE_LEFT"), timeout=self.b.pose_reaction + 1.5)
        self.case(name, inside_tolerance_codes=inside_codes, far_out_blocked_after_s=out)
        self.p.arm_goal("left", self.p.park_side("left"), seconds=2.0)
        self.assertIsNotNone(out, "an arm trajectory to hands_up sent on /left_arm_controller/joint_trajectory did not block")
        self.assertIn("ARM_TRAJECTORY_ACTIVE_LEFT", inside_codes, "a topic trajectory running inside the tolerance did not block: only the "
                      "controller's reference can see it (#44 design, condition 1)")
        self.passed(name)

    # ---- 08 both paths, no replay -----------------------------------------------------------------------------------------------------
    def test_l02_08_arm_out_blocks_both_command_paths_and_nothing_replays(self):
        name = "08_both_paths"
        self.start(name)
        self.p.arm_goal("left", self.far_out("left"), seconds=1.5)
        took = self.p.wait_blocked("ARM_NOT_STOWED_LEFT", timeout=self.b.pose_reaction)
        accepted, t0 = self.try_lift_both_paths(0.03)
        moved = self.p.lift_motion_since(t0)
        self.case(name, blocked_after_s=took, action_accepted=accepted, moved_m=round(moved, 5))
        self.assertLessEqual(moved, lift_still_m(PROFILE), f"the lift moved {moved:.4f} m with the left arm out (action + topic)")
        self.assertFalse(accepted, "lift_controller accepted a goal with the left arm out of park")
        self.assertIsNotNone(took, "not blocked with ARM_NOT_STOWED_LEFT")
        self.p.arm_goal("left", self.p.park_side("left"), seconds=1.5)
        self.assert_allowed("after parking the left arm")
        replay = self.assert_lift_still(time.monotonic(), "after re-activation (no replay)", seconds=3.0)
        self.case(name, replay_moved_m=replay)
        self.passed(name)

    # ---- 09 outside activation --------------------------------------------------------------------------------------------------------
    def test_l02_09_an_outside_activation_while_blocked_is_undone(self):
        name = "09_outside_activation"
        self.start(name)
        self.p.arm_goal("left", self.far_out("left"), seconds=1.5)
        blocked = self.p.wait_blocked("ARM_NOT_STOWED_LEFT", timeout=self.b.pose_reaction)
        t0 = time.monotonic()
        switched = self.p.switch(activate=[LIFT_CONTROLLER])
        self.p.lift_topic_command(self.p.lift() + 0.05, 2.0)         # the worst case: a command arrives in the window
        undone = self.p.wait_for(lambda: self.p.lift_controller_state() != "active", self.b.pose_reaction + 1.0)
        time.sleep(1.0)
        travel = self.p.lift_motion_since(t0)
        window = [(round(t - t0, 3), level, list(codes)) for t, level, _, codes in self.p.timeline().status if t >= t0]
        unnamed = [w for w in window if w[1] != 2 or "ARM_NOT_STOWED_LEFT" not in w[2]]
        self.case(name, blocked_before=blocked is not None, switch_ok=switched, undone_after_s=undone, travel_m=round(travel, 5),
                  travel_budget_m=round(self.b.travel_after_violation, 4), statuses_in_window=len(window), not_naming_the_cause=unnamed[:5])
        self.p.arm_goal("left", self.p.park_side("left"), seconds=1.5)
        self.assertIsNotNone(undone, f"{LIFT_CONTROLLER} stayed active after an outside activation with the left arm out")
        self.assertLessEqual(travel, self.b.travel_after_violation, f"the lift travelled {travel:.4f} m in the window")
        self.assertIsNotNone(blocked, "the guard was not blocking before the outside activation")
        self.assertTrue(window, "no status at all in the window after the outside activation")
        self.assertFalse(unnamed, f"a status in the window was not ERROR naming ARM_NOT_STOWED_LEFT: {unnamed[:3]}")
        self.passed(name)

    # ---- 10 mid-motion ----------------------------------------------------------------------------------------------------------------
    def test_l02_10_an_arm_leaving_park_mid_move_stops_the_lift_up_and_down(self):
        name = "10_mid_motion"
        rows = {}
        for side, delta in (("left", 0.06), ("right", -0.06)):
            self.start(f"{name}_{side}")
            l0 = self.p.lift()
            accepted, handle, seconds = self.p.lift_goal(l0 + delta)
            time.sleep(1.0)
            t_cmd = time.monotonic()
            self.p.arm_goal(side, self.far_out(side), seconds=0.5, wait=False)
            blocked = self.p.wait_blocked(f"ARM_NOT_STOWED_{side.upper()}", timeout=self.b.pose_reaction + 1.0)
            status = self.p.result(handle, seconds + 10.0)[0] if accepted else None
            time.sleep(self.b.settle + 0.5)
            stopped = self.p.lift()
            tl = self.p.timeline()
            t_cross = first_crossing(tl.arms, lambda e: e[1][side] > self.tol, after=t_cmd)
            at_cross = next((p for t, p in reversed(tl.lift) if t <= (t_cross or 0)), None)
            travel = abs(stopped - at_cross) if at_cross is not None else None
            # Two references, kept apart. FROM THE COMMAND: when the test sent the arm away, to the guard's ERROR, to lift_controller not
            # active and to the lift's stop (the guard acts on the arm controller's own reference, so this is the whole reaction). FROM THE
            # CROSSING: when /joint_states first showed the arm outside the tolerance (later than the command; 0.0 means the guard had
            # already acted). Measured values of the simulation, not accepted stopping limits (D16, platform-fw #20).
            blocks_this_arm = lambda e: e[1] == ERROR and f"ARM_NOT_STOWED_{side.upper()}" in e[3]  # noqa: E731
            t_error_cmd = first_crossing(tl.status, blocks_this_arm, after=t_cmd)
            t_inactive_cmd = first_not_active_after(tl.controller, t_cmd)
            at_cmd = next((p for t, p in reversed(tl.lift) if t <= t_cmd), None)
            t_error = None if t_cross is None else first_crossing(tl.status, blocks_this_arm, after=t_cross)
            t_inactive = None if t_cross is None else first_not_active_after(tl.controller, t_cross)
            t_hold = time.monotonic()
            time.sleep(2.0)
            hold = self.p.lift_motion_since(t_hold)
            rows[side] = {"direction": "up" if delta > 0 else "down", "accepted": accepted, "goal_status": status,
                          "blocked": blocked is not None, "travel_after_crossing_m": None if travel is None else round(travel, 5),
                          "command_to_error_s": None if t_error_cmd is None else round(t_error_cmd - t_cmd, 3),
                          "command_to_inactive_s": None if t_inactive_cmd is None else round(t_inactive_cmd - t_cmd, 3),
                          "travel_after_command_m": None if at_cmd is None else round(abs(stopped - at_cmd), 5),
                          "crossing_to_error_s": None if t_error is None else round(t_error - t_cross, 3),
                          "crossing_to_inactive_s": None if t_inactive is None else round(t_inactive - t_cross, 3),
                          "stopped_at": stopped, "target": l0 + delta, "hold_m": round(hold, 5)}
            self.p.arm_goal(side, self.p.park_side(side), seconds=1.5)
            self.p.wait_allowed(self.b.allow_within)
            rows[side]["resume_m"] = round(self.assert_lift_still(time.monotonic(), f"{side}: after re-parking (no resume)", seconds=3.0), 5)
        self.case(name, rows=rows, travel_budget_m=round(self.b.travel_after_violation, 4))
        for side, r in rows.items():
            self.assertTrue(r["accepted"], f"{side}: the lift goal was rejected before the arm moved (precondition)")
            self.assertGreater(abs(r["target"] - r["stopped_at"]), 0.005, f"{side}: the lift reached its target although the arm left park")
            self.assertIsNotNone(r["travel_after_crossing_m"], f"{side}: the arm never crossed the tolerance in the recording")
            self.assertIsNotNone(r["command_to_inactive_s"], f"{side}: lift_controller was never seen not active after the arm was sent away")
            self.assertLessEqual(r["travel_after_crossing_m"], self.b.travel_after_violation,
                                 f"{side}: the lift travelled {r['travel_after_crossing_m']:.4f} m after the arm left the tolerance")
            self.assertLessEqual(r["hold_m"], lift_still_m(PROFILE), f"{side}: no hold after the stop")
            self.assertTrue(r["blocked"], f"{side}: not blocked with ARM_NOT_STOWED_{side.upper()}")
        for side in rows:       # start() made a record per side; they pass with the case, or the sidecar says FAIL for a green run
            self.record["cases"][f"{name}_{side}"]["ok"] = True
        self.passed(name)

    # ---- 11 arm state stops -----------------------------------------------------------------------------------------------------------
    def test_l02_11_arm_joint_state_stopping_blocks(self):
        name = "11_arm_state_stops"
        self.start(name)
        lift_before = self.p.lift()      # where the lift IS, not the nominal START_HEIGHT: a start-up move cut short is not motion in this case
        with self.p.joint_states_silenced():
            self.assertTrue(self.p.switch(deactivate=["joint_state_broadcaster"]), "could not deactivate joint_state_broadcaster")
            try:
                took = self.p.wait_blocked(any_of=("ARM_STATE_STALE", "ARM_STATE_MISSING"), timeout=self.b.stale_reaction)
                accepted, _, _ = self.p.lift_goal(START_HEIGHT + 0.03)
            finally:
                reactivated = self.p.switch(activate=["joint_state_broadcaster"])
        t_back = time.monotonic()
        # the lift is judged by a sample taken AFTER the arm state came back, not by the last one cached before it stopped
        seen = self.p.wait_for(lambda: self.p.lift_sample_after(t_back), 5.0)
        time.sleep(1.0)
        moved = abs(self.p.lift() - lift_before)
        self.case(name, blocked_after_s=took, action_accepted=accepted, moved_m=round(moved, 5), lift_before_m=lift_before,
                  broadcaster_reactivated=reactivated,
                  lift_seen_again_after_s=seen)
        self.assertTrue(reactivated, "joint_state_broadcaster did not come back: the lift could not be observed")
        self.assertIsNotNone(seen, "no lift_joint sample after the arm state came back: 'the lift did not move' would rest on a stale value")
        self.assertLessEqual(moved, lift_still_m(PROFILE), f"the lift moved {moved:.4f} m while there was no arm state")
        self.assertFalse(accepted, "lift_controller accepted a goal with no arm state")
        self.assertIsNotNone(took, f"no ARM_STATE_STALE/MISSING within {self.b.stale_reaction:.2f} s of the arm state stopping")
        self.passed(name)

    # ---- 12 injected arm state --------------------------------------------------------------------------------------------------------
    def test_l02_12_injected_arm_state_per_joint_freshness_and_nan(self):
        name = "12_injected_arm_state"
        self.start(name)
        parked = {j: self.park[j] for j in ALL_ARM_JOINTS}
        current = {"pos": dict(parked)}
        rows = {}
        with self.p.joint_states_silenced():
            self.assertTrue(self.p.switch(deactivate=["joint_state_broadcaster"]), "could not deactivate joint_state_broadcaster")
            try:
                with self.p.inject_arm_states(lambda: current["pos"]):
                    rows["fresh_parked_allowed_after_s"] = self.p.wait_allowed(self.b.allow_within + self.b.stale_reaction)
                    current["pos"] = {j: v for j, v in parked.items() if j != "openarm_right_joint7"}
                    rows["one_joint_absent_blocked_after_s"] = self.p.wait_blocked(any_of=("ARM_STATE_STALE", "ARM_STATE_MISSING"),
                                                                                   timeout=self.b.stale_reaction)
                    current["pos"] = dict(parked)
                    rows["restored_allowed_after_s"] = self.p.wait_allowed(self.b.allow_within)
                    current["pos"] = {**parked, "openarm_left_joint3": math.nan}
                    rows["nan_blocked_after_s"] = self.p.wait_blocked(any_of=("ARM_NOT_STOWED_LEFT", "ARM_STATE_STALE", "ARM_STATE_MISSING"),
                                                                      timeout=self.b.pose_reaction)
                    current["pos"] = dict(parked)
            finally:
                self.p.switch(activate=["joint_state_broadcaster"])
        self.case(name, **rows)
        self.assertIsNotNone(rows["fresh_parked_allowed_after_s"], "fresh, parked injected arm state was never allowed (does the guard read "
                             f"{'/joint_states'}?)")
        self.assertIsNotNone(rows["one_joint_absent_blocked_after_s"], "openarm_right_joint7 absent from every message did not block: "
                             "freshness must be per joint (#44 design)")
        self.assertIsNotNone(rows["nan_blocked_after_s"], "a NaN position for openarm_left_joint3 did not block: NaN must never read as stowed")
        self.passed(name)

    # ---- 13 MoveIt --------------------------------------------------------------------------------------------------------------------
    def test_l02_13_a_moveit_lift_request_with_an_arm_out_does_not_move_the_lift(self):
        name = "13_moveit"
        self.start(name)
        from openamrobot_upperbody_moveit.move_group_client import MoveGroupClient
        node = rclpy.create_node("l02_moveit_client")
        try:
            client = MoveGroupClient(node)
            self.assertTrue(client.wait_for_server(60.0), "move_group's /move_action did not appear")
            self.p.arm_goal("left", self.far_out("left"), seconds=1.5)
            self.p.wait_blocked("ARM_NOT_STOWED_LEFT", timeout=self.b.pose_reaction)
            t0 = time.monotonic()
            outcome = client.move("lift", {LIFT: START_HEIGHT + 0.03}, result_timeout=30.0)
            moved = self.p.lift_motion_since(t0)
        finally:
            node.destroy_node()
        self.case(name, moveit_outcome=outcome.as_dict(), moved_m=round(moved, 5))
        self.p.arm_goal("left", self.p.park_side("left"), seconds=1.5)
        self.assertLessEqual(moved, lift_still_m(PROFILE), f"MoveIt moved the lift {moved:.4f} m with the left arm out")
        succeeded = outcome.success() if callable(outcome.success) else outcome.success
        self.assertFalse(succeeded, f"MoveIt reported success for a lift move with the left arm out: {outcome.as_dict()}")
        self.passed(name)

    # ---- 14 two reasons ---------------------------------------------------------------------------------------------------------------
    def test_l02_14_two_reasons_at_once_are_both_reported_and_cleared_separately(self):
        name = "14_two_reasons"
        self.start(name)
        self.p.arm_goal("left", self.far_out("left"), seconds=1.5)
        self.p.arm_goal("right", self.far_out("right"), seconds=1.5)
        both = self.p.wait_for(lambda: self.p.blocked("ARM_NOT_STOWED_LEFT") and self.p.blocked("ARM_NOT_STOWED_RIGHT"), self.b.pose_reaction)
        self.p.arm_goal("left", self.p.park_side("left"), seconds=1.5)
        only_right = self.p.wait_for(lambda: self.p.blocked("ARM_NOT_STOWED_RIGHT") and not self.p.blocked("ARM_NOT_STOWED_LEFT"),
                                     self.b.pose_reaction)
        still_blocked = self.p.lift_controller_state() != "active"
        self.p.arm_goal("right", self.p.park_side("right"), seconds=1.5)
        allowed = self.p.wait_allowed(self.b.allow_within)
        self.case(name, both_codes_after_s=both, only_right_after_s=only_right, still_blocked=still_blocked, allowed_after_s=allowed)
        self.assertIsNotNone(both, "both arms out did not report both ARM_NOT_STOWED codes")
        self.assertIsNotNone(only_right, "after parking the left arm the status did not keep ARM_NOT_STOWED_RIGHT alone")
        self.assertTrue(still_blocked, f"{LIFT_CONTROLLER} was active with the right arm still out")
        self.assertIsNotNone(allowed, "not allowed after both arms were parked")
        self.passed(name)

    # ---- zz run-wide --------------------------------------------------------------------------------------------------------------------
    # ---- 15 every stowed state ------------------------------------------------------------------------------------------------------
    def test_l02_15_every_stowed_state_allows_the_lift(self):
        name = "15_each_stowed_state"
        self.start(name)
        states, rows = srdf_states(), {}
        for state in self.params["stowed_states"]:
            poses = {side: states.get((state, f"{side}_arm")) for side in ARM_JOINTS}
            if not all(poses.values()) or not all(all(j in poses[s] for j in ARM_JOINTS[s]) for s in ARM_JOINTS):
                rows[state] = {"skipped": "not an SRDF group state of both arms"}
                continue
            for side in ARM_JOINTS:
                self.p.arm_goal(side, [poses[side][j] for j in ARM_JOINTS[side]], seconds=2.0)
            allowed = self.p.wait_allowed(self.b.allow_within)
            l0 = self.p.lift()
            accepted, _ = self.p.lift_goal_and_wait(l0 + 0.01)
            rows[state] = {"allowed_after_s": allowed, "accepted": accepted, "moved_m": round(self.p.lift() - l0, 4)}
        for side in ARM_JOINTS:
            self.p.arm_goal(side, self.p.park_side(side), seconds=2.0)
        self.case(name, stowed_states=list(self.params["stowed_states"]), rows=rows)
        tried = {k: r for k, r in rows.items() if "skipped" not in r}
        self.assertTrue(tried, f"no stowed state of {list(self.params['stowed_states'])} is an SRDF group state of both arms")
        for state, r in tried.items():
            self.assertIsNotNone(r["allowed_after_s"], f"both arms at stowed state {state!r}: the guard did not allow the lift")
            self.assertGreater(r["moved_m"], 0.008, f"both arms at stowed state {state!r}: the lift did not move")
        self.passed(name)

    # ---- 16 the wiring, at runtime -----------------------------------------------------------------------------------------------------
    def test_l02_16_one_guard_and_one_odometry_stub_run_and_the_stub_is_the_only_odom_source(self):
        """What test_lift_guard_static.py reads off the launch description, on the running stack (PR #60 review, 12 Nov): in this test's
        ROS domain exactly one process runs the guard and exactly one the stub, the stub is the only publisher of /odom, and it publishes."""
        name = "16_runtime_wiring"
        self.case(name)
        domain = os.environ["ROS_DOMAIN_ID"]
        guards, stubs = pids_running(INTERFACE["guard_executable"], domain), pids_running(INTERFACE["odom_stub_executable"], domain)
        odom = self.p.publisher_nodes("/odom")
        status = self.p.publisher_nodes(INTERFACE["status_topic"])
        t0 = time.monotonic()
        time.sleep(1.0)
        published = len([t for t in self.p.odom_times(published=True) if t >= t0])
        self.case(name, guard_pids=guards, stub_pids=stubs, odom_publishers=odom, status_publishers=status, odom_samples_in_1s=published)
        self.assertEqual(len(guards), 1, f"processes running {INTERFACE['guard_executable']}: {guards}")
        self.assertEqual(len(stubs), 1, f"processes running {INTERFACE['odom_stub_executable']}: {stubs}")
        self.assertEqual(odom, [INTERFACE["odom_stub_node"]], f"/odom publishers: {odom} (mock: the stub only)")
        self.assertEqual(len(status), 1, f"{INTERFACE['status_topic']} publishers: {status}")
        self.assertGreaterEqual(published, 5, f"the stub published {published} /odom samples in 1 s")
        self.passed(name)

    def test_l02_zz_safety_properties_held_throughout_the_run(self):
        self.check_properties()


class TestProcessesAlive(unittest.TestCase):
    def test_no_process_of_the_launch_died_while_the_tests_were_running(self, proc_info):
        stack_alive(proc_info)


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_l02_processes_started_and_exited_cleanly(self, proc_info):
        stack_healthy(proc_info, expected=EXPECTED_PROCESSES, sidecar=("L02", "lift_guard_arms_mock"))
