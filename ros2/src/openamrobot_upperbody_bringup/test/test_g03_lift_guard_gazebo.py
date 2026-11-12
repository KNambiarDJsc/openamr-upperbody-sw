# SPDX-License-Identifier: MIT
"""G03 (#44): the lift guard in Gazebo, where two things differ from mock and are the reason this test exists.

1. /joint_states is joint_state_publisher's merge of /arm/joint_states and /base/joint_states (sim.launch.py). When the arm source stops, the
   merge keeps publishing the arm joints with frozen positions and fresh stamps (#44 baseline, row D). The guard must read /arm/joint_states.
2. The base is simulated (DiffDrive), so "base moving" comes from a real /odom. After #49 odometry alone is not evidence of motion: the base's
   true displacement is read from the simulator and recorded next to what /odom said.

Lift motion is asserted from lift_joint on /arm/joint_states AND from the simulator's own pose of lift_carriage_link (#44 acceptance 4). An
action result is never evidence of motion: in DART a goal reported success while the lift stayed at 0 (#49). In dynamic_pose/info the model's
entry ("openamrobot") is its WORLD pose; link entries are relative to the model, which is what the carriage height needs.

Engine: bullet-featherstone, passed explicitly, as M02 does. Until #49's root cause is fixed DART does not move the lift, and a "did not move"
check would pass for the wrong reason; the positive control (case 01) is there to catch exactly that.

Budgets are twice mock's: the simulator runs below real time on a CI runner, and the guard's clock is sim time.

Cases (each from: both arms parked, lift at 0.10 m, except 00)
  00 start-up (#44, 10 Nov: a start-up trajectory to park on both arm controllers, in Gazebo too): blocked first; lift_controller never
     active before the first OK; both arms within the tolerance at the first OK; the carriage unmoved in the simulator until then
  01 allowed: a lift goal moves lift_joint and the carriage in the simulator (positive control)
  02 left arm out: blocked with ARM_NOT_STOWED_LEFT; neither the action nor the topic moves the carriage
  03 arm source stopped while the merge still publishes the arm joints: blocked with ARM_STATE_*, and the carriage does not move
  04 injected /arm/joint_states with one joint absent: blocked (per-joint freshness, on the topic the merge would hide)
  05 base driven at 0.2 m/s through /cmd_vel: blocked with BASE_MOVING; true travel and odometry travel are both recorded
  06 MEASUREMENT for the Gazebo stowed_tolerance (#44, 10 Nov: "set per profile from Karthik's settled-error measurement plus margin"):
     per-joint |position - park| on /arm/joint_states at rest, during a lift move, while the base drives, and after each arm leaves park
     and is sent back. Fails only if there is no data, or if the arms at rest in park sit outside the guard's own tolerance
  07 the wiring at runtime: one guard process, no odometry stub process, /odom from the simulated base
  zz the safety properties P1-P7 over the whole run

Simulation only; nothing here is evidence about the firmware interlock (CTL-017).
"""
import math
import os
import random
import sys
import threading
import time
import unittest

os.environ["ROS_DOMAIN_ID"] = str(random.randint(1, 100))
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["GZ_PARTITION"] = f"g03_{os.getpid()}"

import launch  # noqa: E402
import launch_testing  # noqa: E402
import launch_testing.actions  # noqa: E402
import pytest  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "openamrobot_upperbody_gazebo", "test"))
from a7_support import A7_ENABLED, SKIP_REASON, stack_alive, stack_healthy  # noqa: E402
from lift_guard_support import (ALL_ARM_JOINTS, ARM_JOINTS, INTERFACE, LIFT_CONTROLLER, GuardCase, lift_still_m,  # noqa: E402
                                pids_running, stack_under_test)
from sim_test_support import gz_link_poses  # noqa: E402

PROFILE = "gazebo"
ENGINE = "gz-physics-bullet-featherstone-plugin"
CARRIAGE, MODEL = "lift_carriage_link", "openamrobot"
EXPECTED_PROCESSES = ["robot_state_publisher", "spawner", "move_group", INTERFACE["guard_executable"]]


@pytest.mark.launch_test
def generate_test_description():
    if not A7_ENABLED:
        return launch.LaunchDescription([launch.actions.ExecuteProcess(cmd=["sleep", "30"]), launch_testing.actions.ReadyToTest()]), {}
    stack, _ = stack_under_test(PROFILE, headless="true", physics_engine=ENGINE)
    return launch.LaunchDescription([stack, launch_testing.actions.ReadyToTest()]), {}


def gz_position(name):
    try:
        pose = gz_link_poses().get(name)
    except Exception:
        return None
    if pose is None:
        return None
    pos = pose.get("position", {})
    return (pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0))


def carriage_z():
    p = gz_position(CARRIAGE)
    return None if p is None else p[2]


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardGazebo(GuardCase, unittest.TestCase):
    PROFILE = PROFILE
    NODE = "g03_lift_guard"
    CHECK_ID, SLUG, NAME = "G03", "lift_guard_gazebo", "#44 lift guard in Gazebo"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        for k in ("pose_reaction", "base_reaction", "stale_reaction", "allow_within", "status_lag", "warn_max"):
            setattr(cls.b, k, getattr(cls.b, k) * 2.0)
        cls.b.travel_after_violation = 0.015 * cls.b.pose_reaction + lift_still_m(PROFILE)
        cls.record["budgets_s"] = cls.b.as_dict()
        cls.merged_t, cls.odom_xy = {}, None
        cls.p.node.create_subscription(JointState, "/joint_states", cls._on_merged, 50)
        cls.p.node.create_subscription(Odometry, "/odom", cls._on_odom, 10)
        cls.cmd_vel = cls.p.node.create_publisher(Twist, "/cmd_vel", 10)

    @classmethod
    def _on_merged(cls, msg):
        now = time.monotonic()
        for n in msg.name:
            cls.merged_t[n] = now

    @classmethod
    def _on_odom(cls, msg):
        cls.odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def far_out(self, side):
        pose = self.p.park_side(side)
        pose[3] = 1.5
        return pose

    def assert_carriage_still(self, z0, what):
        z1 = carriage_z()
        dz = None if None in (z0, z1) else z1 - z0
        self.assertIsNotNone(dz, f"{what}: no ground truth for {CARRIAGE}")
        self.assertLessEqual(abs(dz), lift_still_m(PROFILE), f"{what}: the carriage moved {dz:.4f} m in the simulator")
        return round(dz, 5)

    def test_g03_00_starts_blocked_and_allows_only_once_both_arms_are_parked(self):
        name = "00_startup"
        self.case(name)
        z_start = carriage_z()
        first_ok = self.p.wait_for(lambda: any(s["ok"] for _, s in self.p.status_changes), 120.0)
        z_ok = carriage_z()
        tl = self.p.timeline()
        changes = list(self.p.status_changes)
        self.assertTrue(changes, f"no message on {INTERFACE['status_topic']} since the launch")
        self.assertFalse(changes[0][1]["ok"], f"the first status already allowed the lift: {changes[0][1]}")
        self.assertIsNotNone(first_ok, "the guard never allowed the lift after start-up")
        t_ok = next(t for t, s in changes if s["ok"])
        active_before = [t for t, s in tl.controller if s == "active" and t < t_ok - 0.05]
        devs = [d for t, d in tl.arms if t <= t_ok]
        dz = None if None in (z_start, z_ok) else round(z_ok - z_start, 5)
        self.case(name, first_status=changes[0][1], controller_active_before_first_ok=len(active_before),
                  arm_deviation_at_first_ok=devs[-1] if devs else None, carriage_dz_before_first_ok_m=dz)
        self.assertTrue(any(t < t_ok for t, _ in tl.controller), f"{LIFT_CONTROLLER}'s state was never observed before the first OK")
        self.assertFalse(active_before, f"{LIFT_CONTROLLER} was active before the guard first allowed the lift (#44 acceptance 1)")
        self.assertTrue(devs and max(devs[-1].values()) <= self.tol, f"first allowed with an arm out of park: {devs[-1] if devs else None}")
        self.assertIsNotNone(dz, f"no ground truth for {CARRIAGE} at start-up")
        self.assertLessEqual(abs(dz), lift_still_m(PROFILE), f"the carriage moved {dz:.4f} m in the simulator before the first OK")
        self.passed(name)

    def test_g03_01_allowed_a_lift_goal_moves_the_carriage_in_the_simulator(self):
        name = "01_positive_control"
        self.start(name)
        try:
            self.record["gz_pose_entries"] = {n: p.get("position") for n, p in gz_link_poses().items() if n in (CARRIAGE, MODEL)}
        except Exception as e:
            self.record["gz_pose_entries"] = repr(e)
        l0, z0 = self.p.lift(), carriage_z()
        accepted, _ = self.p.lift_goal_and_wait(l0 + 0.03)
        time.sleep(1.0)
        moved, z1 = self.p.lift() - l0, carriage_z()
        self.case(name, accepted=accepted, joint_moved_m=round(moved, 4), carriage_moved_m=None if None in (z0, z1) else round(z1 - z0, 4))
        self.assertTrue(accepted, "the lift goal was rejected while the guard allowed the lift")
        self.assertGreater(moved, 0.025, f"lift_joint moved {moved:.4f} m of 0.03")
        self.assertIsNotNone(z0, f"the simulator's pose message has no {CARRIAGE}: no ground truth")
        self.assertGreater(z1 - z0, 0.025, f"the simulator moved {CARRIAGE} {z1 - z0:.4f} m of 0.03 (lift_joint said {moved:.4f})")
        self.passed(name)

    def test_g03_02_arm_out_blocks_both_command_paths(self):
        name = "02_arm_out"
        self.start(name)
        self.p.arm_goal("left", self.far_out("left"), seconds=2.0)
        took = self.p.wait_blocked("ARM_NOT_STOWED_LEFT", timeout=self.b.pose_reaction + 1.0)
        z0 = carriage_z()
        accepted, t0 = self.try_lift_both_paths(0.03)
        time.sleep(1.0)
        joint_moved = self.p.lift_motion_since(t0)
        dz = carriage_z()
        dz = None if None in (z0, dz) else dz - z0
        self.case(name, blocked_after_s=took, action_accepted=accepted, joint_moved_m=round(joint_moved, 5), carriage_dz_m=dz)
        self.p.arm_goal("left", self.p.park_side("left"), seconds=2.0)
        self.assertLessEqual(joint_moved, lift_still_m(PROFILE), f"lift_joint moved {joint_moved:.4f} m with the left arm out")
        self.assertIsNotNone(dz, f"no ground truth for {CARRIAGE}")
        self.assertLessEqual(abs(dz), lift_still_m(PROFILE), f"the carriage moved {dz:.4f} m in the simulator with the left arm out")
        self.assertFalse(accepted, "lift_controller accepted a goal with the left arm out of park")
        self.assertIsNotNone(took, "not blocked with ARM_NOT_STOWED_LEFT")
        self.passed(name)

    def test_g03_03_arm_source_stopped_blocks_although_the_merge_looks_fresh(self):
        name = "03_arm_source_stopped"
        self.start(name)
        z0 = carriage_z()
        with self.p.joint_states_silenced():
            self.assertTrue(self.p.switch(deactivate=["joint_state_broadcaster"]), "could not deactivate joint_state_broadcaster")
            try:
                took = self.p.wait_blocked(any_of=("ARM_STATE_STALE", "ARM_STATE_MISSING"), timeout=self.b.stale_reaction)
                time.sleep(1.0)
                now = time.monotonic()
                merged_fresh = all(now - self.merged_t.get(j, 0.0) < 0.5 for j in ARM_JOINTS["left"])
                accepted, _ = self.try_lift_both_paths(0.03)
                time.sleep(1.0)
                dz = self.assert_carriage_still(z0, "arm source stopped")
            finally:
                self.p.switch(activate=["joint_state_broadcaster"])
        self.case(name, blocked_after_s=took, merged_still_fresh=merged_fresh, action_accepted=accepted, carriage_dz_m=dz)
        self.assertFalse(accepted, "lift_controller accepted a goal with the arm source stopped")
        self.assertIsNotNone(took, f"no ARM_STATE_STALE/MISSING within {self.b.stale_reaction:.2f} s although /arm/joint_states stopped "
                                   f"(the merged /joint_states still fresh: {merged_fresh})")
        self.passed(name)

    def test_g03_04_one_joint_absent_from_the_raw_arm_state_blocks(self):
        name = "04_one_joint_absent"
        self.start(name)
        parked = {j: self.park[j] for j in ALL_ARM_JOINTS}
        current = {"pos": dict(parked)}
        rows = {}
        with self.p.joint_states_silenced():
            self.assertTrue(self.p.switch(deactivate=["joint_state_broadcaster"]), "could not deactivate joint_state_broadcaster")
            try:
                with self.p.inject_arm_states(lambda: current["pos"]):
                    rows["fresh_parked_allowed_after_s"] = self.p.wait_allowed(self.b.allow_within + self.b.stale_reaction)
                    current["pos"] = {j: v for j, v in parked.items() if j != "openarm_left_joint7"}
                    rows["one_joint_absent_blocked_after_s"] = self.p.wait_blocked(any_of=("ARM_STATE_STALE", "ARM_STATE_MISSING"),
                                                                                   timeout=self.b.stale_reaction)
                    current["pos"] = dict(parked)
            finally:
                self.p.switch(activate=["joint_state_broadcaster"])
        self.case(name, **rows)
        self.assertIsNotNone(rows["fresh_parked_allowed_after_s"], "fresh parked arm state injected on /arm/joint_states was never allowed: "
                             "does the guard read /arm/joint_states in Gazebo?")
        self.assertIsNotNone(rows["one_joint_absent_blocked_after_s"], "openarm_left_joint7 absent from every message did not block")
        self.passed(name)

    def test_g03_05_base_driving_blocks_and_the_true_travel_is_recorded(self):
        name = "05_base_moving"
        self.start(name)
        base0, z0, odom0 = gz_position(MODEL), carriage_z(), self.odom_xy
        stop = threading.Event()

        def drive():
            twist = Twist()
            twist.linear.x = 0.2
            while not stop.is_set():
                self.cmd_vel.publish(twist)
                time.sleep(0.05)
            for _ in range(10):
                self.cmd_vel.publish(Twist())
                time.sleep(0.05)
        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        try:
            took = self.p.wait_blocked("BASE_MOVING", timeout=self.b.base_reaction + 2.0)
            accepted, _ = self.try_lift_both_paths(0.03)
            time.sleep(1.0)
            z1 = carriage_z()
        finally:
            stop.set()
            driver.join(3.0)
        base1, odom1 = gz_position(MODEL), self.odom_xy

        def dist(a, b):
            return None if None in (a, b) else round(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5, 4)
        dz = None if None in (z0, z1) else round(z1 - z0, 5)
        self.case(name, blocked_after_s=took, action_accepted=accepted, carriage_dz_m=dz, base_true_travel_m=dist(base0, base1),
                  base_odom_travel_m=dist(odom0, odom1), note="true travel from the simulator's model pose; odom from /odom (#49)")
        self.assertIsNotNone(dz, f"no ground truth for {CARRIAGE}")
        self.assertLessEqual(abs(dz), lift_still_m(PROFILE), f"the carriage moved {dz:.4f} m while the base was driven")
        self.assertFalse(accepted, "lift_controller accepted a goal while the base was driven")
        self.assertIsNotNone(took, "no BASE_MOVING while the base was driven")
        self.passed(name)

    def park_errors(self, seconds):
        """Sample /arm/joint_states for `seconds` at 50 Hz: {joint: max |position - park|} over the window (inf if never seen)."""
        worst = {j: 0.0 for j in self.park}
        seen = set()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            with self.p.lock:
                snap = {j: self.p.joints.get(j) for j in self.park}
            for j, v in snap.items():
                if v is not None:
                    seen.add(j)
                    worst[j] = max(worst[j], abs(v - self.park[j]))
            time.sleep(0.02)
        return {j: (round(worst[j], 6) if j in seen else math.inf) for j in self.park}

    def test_g03_06_measure_the_settled_park_error(self):
        name = "06_park_settled_error"
        self.start(name)
        rows = {}
        time.sleep(2.0)
        rows["at_rest"] = self.park_errors(3.0)
        l0 = self.p.lift()
        accepted, handle, seconds = self.p.lift_goal(l0 + 0.04)
        rows["during_lift_move"] = self.park_errors(max(1.0, seconds))
        if accepted:
            self.p.result(handle, seconds + 10.0)
            self.p.lift_goal_and_wait(l0)
        stop = threading.Event()

        def drive():
            twist = Twist()
            twist.linear.x = 0.2
            while not stop.is_set():
                self.cmd_vel.publish(twist)
                time.sleep(0.05)
            for _ in range(10):
                self.cmd_vel.publish(Twist())
                time.sleep(0.05)
        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        try:
            time.sleep(0.5)
            rows["while_base_drives"] = self.park_errors(2.0)
        finally:
            stop.set()
            driver.join(3.0)
        for side in ARM_JOINTS:
            self.p.arm_goal(side, self.far_out(side), seconds=2.0)
            self.p.arm_goal(side, self.p.park_side(side), seconds=2.0)
        time.sleep(2.0)
        rows["after_out_and_back"] = self.park_errors(3.0)
        summary = {}
        for cond, errs in rows.items():
            j = max(errs, key=errs.get)
            summary[cond] = {"max_rad": errs[j], "joint": j}
        self.case(name, lift_goal_accepted=accepted, tolerance_now=self.tol, summary=summary, per_joint=rows,
                  note="input for the per-profile Gazebo stowed_tolerance (#44, 10 Nov); the margin is the guard owner's call")
        self.assertTrue(all(math.isfinite(v["max_rad"]) for v in summary.values()), f"some arm joint never seen on the arm state: {summary}")
        for cond in ("at_rest", "after_out_and_back"):
            self.assertLessEqual(summary[cond]["max_rad"], self.tol,
                                 f"{cond}: {summary[cond]['joint']} sits {summary[cond]['max_rad']} rad from park, outside the "
                                 "guard's tolerance")
        self.passed(name)

    def test_g03_07_one_guard_runs_and_odometry_comes_from_the_simulator_not_a_stub(self):
        """The Gazebo half of the wiring, at runtime (PR #60 review, 12 Nov): exactly one guard process, no stub process, /odom published
        and by no stub node."""
        name = "07_runtime_wiring"
        self.case(name)
        domain = os.environ["ROS_DOMAIN_ID"]
        guards, stubs = pids_running(INTERFACE["guard_executable"], domain), pids_running(INTERFACE["odom_stub_executable"], domain)
        odom = self.p.publisher_nodes("/odom")
        status = self.p.publisher_nodes(INTERFACE["status_topic"])
        self.case(name, guard_pids=guards, stub_pids=stubs, odom_publishers=odom, status_publishers=status)
        self.assertEqual(len(guards), 1, f"processes running {INTERFACE['guard_executable']}: {guards}")
        self.assertEqual(stubs, [], f"Gazebo runs the mock odometry stub: {stubs}")
        self.assertTrue(odom, "nothing publishes /odom in Gazebo")
        self.assertFalse([n for n in odom if "odom_stub" in n], f"/odom publishers: {odom} (Gazebo: the simulated base only)")
        self.assertEqual(len(status), 1, f"{INTERFACE['status_topic']} publishers: {status}")
        self.passed(name)

    def test_g03_zz_safety_properties_held_throughout_the_run(self):
        self.check_properties()


class TestProcessesAlive(unittest.TestCase):
    def test_no_process_of_the_launch_died_while_the_tests_were_running(self, proc_info):
        stack_alive(proc_info)


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_g03_processes_started_and_exited_cleanly(self, proc_info):
        stack_healthy(proc_info, expected=EXPECTED_PROCESSES, simulator=True, sidecar=("G03", "lift_guard_gazebo"))
