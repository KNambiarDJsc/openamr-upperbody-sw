# SPDX-License-Identifier: MIT
"""G03 (#44): the lift guard in Gazebo, where two things differ from mock and are the reason this test exists.

1. /joint_states is joint_state_publisher's merge of /arm/joint_states and /base/joint_states (sim.launch.py). When the arm source stops, the
   merge keeps publishing the arm joints with frozen positions and fresh stamps (#44 baseline, row D). The guard must read /arm/joint_states.
2. The base is simulated (DiffDrive), so "base moving" comes from a real /odom. After #49, odometry alone is not evidence of motion: the
   base's true displacement is read from the simulator and recorded next to what /odom said.

Lift motion is asserted from lift_joint on /arm/joint_states AND from the simulator's own pose of lift_carriage_link (#44 acceptance 4). An
action result is never evidence of motion: in DART a goal reported success while the lift stayed at 0 (#49).

Engine: bullet-featherstone, passed explicitly, as M02 does. Until #49's root cause is fixed DART does not move the lift, and a
"did not move" check would pass for the wrong reason; the positive control (case 1) would catch it, but there is no point running it.

Cases (each leaves both arms in park and the guard allowing):
  1  allowed: a lift goal moves lift_joint and the carriage in the simulator (positive control)
  2  left arm out: blocked with ARM_NOT_STOWED_LEFT; neither the action nor the topic moves the lift
  3  arm source stopped (joint_state_broadcaster deactivated) while the merge still publishes the arm joints: blocked with ARM_STATE_STALE
     or ARM_STATE_MISSING, and the lift does not move
  4  base driven at 0.2 m/s through /cmd_vel: blocked with BASE_MOVING; the base's true displacement is recorded

Simulation only; nothing here is evidence about the firmware interlock (CTL-017).
"""
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
import rclpy  # noqa: E402
from ament_index_python.packages import get_package_share_directory  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from launch.launch_description_sources import PythonLaunchDescriptionSource  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "openamrobot_upperbody_gazebo", "test"))
from a7_support import A7_ENABLED, SKIP_REASON, stack_alive, stack_healthy, write_run_sidecar  # noqa: E402
from lift_guard_support import (ALLOW_WITHIN_S, ARM_JOINTS, BLOCK_WITHIN_S, INTERFACE, PROVISIONAL, GuardProbe, lift_still_m,  # noqa: E402
                                max_excursion, park_of, srdf_states)
from sim_test_support import gz_link_poses  # noqa: E402

PROFILE = "gazebo"
ENGINE = "gz-physics-bullet-featherstone-plugin"
STARTUP_WITHIN_S = 240.0
CARRIAGE = "lift_carriage_link"
MODEL = "openamrobot"     # the robot model's entry in dynamic_pose/info is its WORLD pose; link entries are relative to it
EXPECTED_PROCESSES = ["robot_state_publisher", "spawner", "move_group", INTERFACE["process_pattern"]]


@pytest.mark.launch_test
def generate_test_description():
    if not A7_ENABLED:
        return launch.LaunchDescription([launch.actions.ExecuteProcess(cmd=["sleep", "30"]), launch_testing.actions.ReadyToTest()]), {}
    moveit = os.path.join(get_package_share_directory("openamrobot_upperbody_bringup"), "launch", "moveit.launch.py")
    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            PythonLaunchDescriptionSource(moveit),
            launch_arguments={"sim": PROFILE, "rviz": "false", "headless": "true", "physics_engine": ENGINE}.items()),
        launch_testing.actions.ReadyToTest(),
    ]), {}


def true_z(name=CARRIAGE):
    """z of a link as the simulator has it (gz transport), or None if the pose message does not carry it."""
    try:
        pose = gz_link_poses().get(name)
    except Exception:
        return None
    return None if pose is None else pose["position"].get("z", 0.0)


def true_xy(name):
    try:
        pose = gz_link_poses().get(name)
    except Exception:
        return None
    return None if pose is None else (pose["position"].get("x", 0.0), pose["position"].get("y", 0.0))


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardGazebo(unittest.TestCase):
    record = {"profile": PROFILE, "physics_engine": ENGINE, "interface": INTERFACE, "provisional": PROVISIONAL, "cases": {}}

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.p = GuardProbe(PROFILE, "g03_lift_guard")
        cls.merged_t = {}
        cls.p.node.create_subscription(JointState, "/joint_states", cls._on_merged, 50)
        cls.cmd_vel = cls.p.node.create_publisher(Twist, "/cmd_vel", 10)
        cls.odom_xy = None
        cls.p.node.create_subscription(Odometry, "/odom", cls._on_odom, 10)
        states = srdf_states()
        cls.park = {s: park_of(states, s) for s in ARM_JOINTS}
        cls.out = list(cls.park["left"])
        cls.out[3] = 1.5

    @classmethod
    def _on_merged(cls, msg):
        now = time.monotonic()
        for n in msg.name:
            cls.merged_t[n] = now

    @classmethod
    def _on_odom(cls, msg):
        cls.odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    @classmethod
    def tearDownClass(cls):
        cases = cls.record["cases"]
        ok = bool(cases) and all(c.get("ok") for c in cases.values())
        write_run_sidecar("G03", "lift_guard_gazebo", "#44 lift guard acceptance in Gazebo", "PASS" if ok else "FAIL", cls.record)
        cls.p.close()
        rclpy.shutdown()

    def case(self, name, **fields):
        self.record["cases"].setdefault(name, {"ok": False}).update(fields)

    def passed(self, name):
        self.record["cases"][name]["ok"] = True

    def precondition_allowed(self):
        return self.p.wait_allowed() is not None

    def assert_allowed(self, what):
        self.assertIsNotNone(self.p.wait_allowed(), f"{what}: the guard did not allow the lift within {ALLOW_WITHIN_S:.1f} s "
                                                    f"(status {self.p.current_status()})")

    def test_g03_1_allowed_a_lift_goal_moves_the_lift_in_the_simulator(self):
        self.assertTrue(self.p.wait_for_joints(STARTUP_WITHIN_S), "/arm/joint_states never listed lift_joint and all 14 arm joints")
        self.case("1_positive_control", allowed_before=self.p.wait_allowed(STARTUP_WITHIN_S) is not None)
        try:
            poses = gz_link_poses()
            self.record["gz_pose_entries"] = {n: p.get("position") for n, p in poses.items()
                                              if n in (CARRIAGE, "base_footprint", "base_link") or "lift" in n or "openamrobot" in n}
        except Exception as e:  # recorded: the ground-truth cases below say what is missing
            self.record["gz_pose_entries"] = repr(e)
        l0, z0 = self.p.lift(), true_z()
        accepted, handle, seconds = self.p.lift_goal(l0 + 0.03)
        self.assertTrue(accepted, "the lift goal was rejected while the guard reported OK")
        self.p.result(handle, seconds + 20.0)
        time.sleep(1.0)
        moved, z1 = self.p.lift() - l0, true_z()
        self.case("1_positive_control", joint_moved_m=round(moved, 4), carriage_z_before=z0, carriage_z_after=z1)
        self.assertGreater(moved, 0.025, f"lift_joint moved {moved:.4f} m of 0.03")
        self.assertIsNotNone(z0, f"the simulator's pose message has no {CARRIAGE}: no ground truth")
        self.assertGreater(z1 - z0, 0.025, f"the simulator moved {CARRIAGE} {z1 - z0:.4f} m of 0.03 (lift_joint said {moved:.4f})")
        accepted, handle, seconds = self.p.lift_goal(l0)
        self.p.result(handle, seconds + 20.0)
        self.passed("1_positive_control")

    def test_g03_2_arm_out_blocks_both_command_paths(self):
        self.case("2_arm_out", allowed_before=self.precondition_allowed())
        self.p.arm_goal("left", self.out, seconds=2.0)
        took = self.p.wait_blocked("ARM_NOT_STOWED_LEFT")
        l0, z0 = self.p.lift(), true_z()
        accepted, handle, seconds = self.p.lift_goal(l0 + 0.03)
        if accepted:
            self.p.result(handle, seconds + 20.0)
        self.p.lift_topic_command(self.p.lift() + 0.03, 2.0)
        worst = max_excursion(self.p.sample_lift(4.0), l0)
        z1 = true_z()
        dz = None if None in (z0, z1) else z1 - z0
        self.case("2_arm_out", blocked_after_s=took, action_accepted=accepted, joint_worst_m=round(worst, 4), carriage_dz_m=dz)
        self.p.arm_goal("left", self.park["left"], seconds=2.0)
        self.assertLessEqual(worst, lift_still_m(PROFILE), f"lift_joint moved {worst:.4f} m with the left arm out of park")
        if dz is not None:
            self.assertLessEqual(abs(dz), lift_still_m(PROFILE), f"the carriage moved {dz:.4f} m in the simulator with the left arm out")
        self.assertFalse(accepted, "lift_controller accepted a goal with the left arm out of park")
        self.assertIsNotNone(took, f"not blocked with ARM_NOT_STOWED_LEFT within {BLOCK_WITHIN_S:.1f} s: {self.p.current_status()}")
        self.assert_allowed("after parking the left arm again")
        self.passed("2_arm_out")

    def test_g03_3_arm_source_stopped_blocks_although_the_merge_looks_fresh(self):
        self.case("3_arm_state_stale", allowed_before=self.precondition_allowed())
        z0 = true_z()
        self.assertTrue(self.p.switch(deactivate=["joint_state_broadcaster"]), "could not deactivate joint_state_broadcaster")
        try:
            took = self.p.wait_for(lambda: self.p.blocked()
                                   and bool({"ARM_STATE_STALE", "ARM_STATE_MISSING"} & set(self.p.current_status()["blocking"]))
                                   and self.p.lift_controller_state() != "active", BLOCK_WITHIN_S)
            time.sleep(1.0)
            now = time.monotonic()
            merged_fresh = all(now - self.merged_t.get(j, 0.0) < 0.5 for j in ARM_JOINTS["left"])
            accepted, handle, seconds = self.p.lift_goal((self.p.lift() or 0.0) + 0.03)
            if accepted:
                self.p.result(handle, seconds + 20.0)
            time.sleep(2.0)
            z1 = true_z()
        finally:
            self.p.switch(activate=["joint_state_broadcaster"])
        dz = None if None in (z0, z1) else z1 - z0
        self.case("3_arm_state_stale", blocked_after_s=took, merged_still_fresh=merged_fresh, action_accepted=accepted, carriage_dz_m=dz)
        self.assertIsNotNone(dz, f"no ground truth for {CARRIAGE}: with the broadcaster off it is the only reading of the lift")
        self.assertLessEqual(abs(dz), lift_still_m(PROFILE), f"the carriage moved {dz:.4f} m in the simulator with no arm state")
        self.assertFalse(accepted, "lift_controller accepted a goal with the arm source stopped")
        self.assertIsNotNone(took, f"no ARM_STATE_STALE/MISSING block within {BLOCK_WITHIN_S:.1f} s of the arm source stopping")
        self.assert_allowed("after the arm source came back")
        self.passed("3_arm_state_stale")

    def test_g03_4_base_driving_blocks_and_the_true_displacement_is_recorded(self):
        self.case("4_base_moving", allowed_before=self.precondition_allowed())
        base0, z0, odom0 = true_xy(MODEL), true_z(), self.odom_xy
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
            took = self.p.wait_blocked("BASE_MOVING")
            accepted, handle, seconds = self.p.lift_goal((self.p.lift() or 0.0) + 0.03)
            if accepted:
                self.p.result(handle, seconds + 20.0)
            time.sleep(1.0)
            z1 = true_z()
        finally:
            stop.set()
            driver.join(3.0)
        base1, odom1 = true_xy(MODEL), self.odom_xy
        travelled = None if None in (base0, base1) else ((base1[0] - base0[0]) ** 2 + (base1[1] - base0[1]) ** 2) ** 0.5
        dz = None if None in (z0, z1) else z1 - z0
        odom_travel = None if None in (odom0, odom1) else ((odom1[0] - odom0[0]) ** 2 + (odom1[1] - odom0[1]) ** 2) ** 0.5
        self.case("4_base_moving", blocked_after_s=took, action_accepted=accepted, carriage_dz_m=dz, base_true_travel_m=travelled,
                  base_odom_travel_m=odom_travel, note="true travel from the simulator's model pose; odom from /odom (#49: they can disagree)")
        if dz is not None:
            self.assertLessEqual(abs(dz), lift_still_m(PROFILE), f"the carriage moved {dz:.4f} m while the base was driven")
        self.assertFalse(accepted, "lift_controller accepted a goal while the base was driven")
        self.assertIsNotNone(took, f"no BASE_MOVING block within {BLOCK_WITHIN_S:.1f} s of driving the base")
        self.assert_allowed("after the base stopped")
        self.passed("4_base_moving")


class TestProcessesAlive(unittest.TestCase):
    def test_no_process_of_the_launch_died_while_the_tests_were_running(self, proc_info):
        stack_alive(proc_info)


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_g03_processes_started_and_exited_cleanly(self, proc_info):
        stack_healthy(proc_info, expected=EXPECTED_PROCESSES, simulator=True, sidecar=("G03", "lift_guard_gazebo"))
