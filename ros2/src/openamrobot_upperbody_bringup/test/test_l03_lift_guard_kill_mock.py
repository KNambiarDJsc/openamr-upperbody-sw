# SPDX-License-Identifier: MIT
"""L03 (#44 acceptance 1): the guard is killed during an allowed lift move on `mock`.

Measured and checked, in this order:
  1  time from the kill to lift_controller reported inactive (the design's launch OnProcessExit handler; includes the CLI's start-up)
  2  time from the kill to lift_joint standing still
  3  the hold: lift_joint stays put for HOLD_S after it stopped
  4  further commands are rejected: an action goal is not accepted and a topic trajectory does not move the lift
  5  re-activating lift_controller by hand (as an operator restarting things would) does not replay the interrupted goal

The design on #44 does not claim protection against a HUNG guard (alive but not deciding), so that is not tested here; if a heartbeat is added,
a case for it belongs here.

A launch test of its own because it kills a process of the launch on purpose: L02's "no process died" check would otherwise fail. Here the
guard alone may die, and only of the signal this test sends.

On main (no guard) it FAILs at its precondition: there is nothing to kill. Simulation only.
"""
import os
import random
import signal
import subprocess
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
from ament_index_python.packages import get_package_prefix, get_package_share_directory  # noqa: E402
from launch.launch_description_sources import PythonLaunchDescriptionSource  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_support import A7_ENABLED, SKIP_REASON, write_run_sidecar  # noqa: E402
from lift_guard_support import (INTERFACE, KILL_DEACTIVATE_WITHIN_S, LIFT_CONTROLLER, PROVISIONAL, GuardProbe,  # noqa: E402
                                lift_still_m, max_excursion)

sys.path.insert(0, os.path.join(get_package_prefix("openamrobot_upperbody_description"), "lib", "openamrobot_upperbody_description"))
from launch_guard import EXIT_OK, MOVE_GROUP_ALLOWED, assert_launch_healthy, assert_no_process_died  # noqa: E402

PROFILE = "mock"
STARTUP_WITHIN_S = 120.0
KILL_AFTER_S = 1.5          # into the lift move
STOP_WINDOW_S = 0.5         # lift_joint changing by less than STOP_EPS_M over this window counts as stopped
STOP_EPS_M = 0.0002
STOP_WITHIN_S = KILL_DEACTIVATE_WITHIN_S + 2.0
HOLD_S = 3.0
GUARD_KILL_CODES = [-signal.SIGKILL]
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


def guard_pids():
    """PIDs of the guard process in THIS test's ROS domain: its command line carries the pattern, and its environment our ROS_DOMAIN_ID
    (so a guard of another stack on the machine is never killed)."""
    out = subprocess.run(["pgrep", "-f", INTERFACE["process_pattern"]], capture_output=True, text=True).stdout.split()
    mine = []
    for pid in out:
        if int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = dict(kv.split(b"=", 1) for kv in f.read().split(b"\0") if b"=" in kv)
        except OSError:
            continue
        if env.get(b"ROS_DOMAIN_ID") == os.environ["ROS_DOMAIN_ID"].encode():
            mine.append(int(pid))
    return mine


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardKilledMidMove(unittest.TestCase):
    record = {"profile": PROFILE, "provisional": PROVISIONAL, "kill_after_s": KILL_AFTER_S,
              "budgets_s": {"deactivate_within": KILL_DEACTIVATE_WITHIN_S, "stop_within": STOP_WITHIN_S}}

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.p = GuardProbe(PROFILE, "l03_lift_guard_kill")

    @classmethod
    def tearDownClass(cls):
        write_run_sidecar("L03", "lift_guard_kill_mock", "#44 lift guard killed mid-move on mock",
                          "PASS" if cls.record.get("ok") else "FAIL", cls.record)
        cls.p.close()
        rclpy.shutdown()

    def wait_still(self, timeout):
        """-> seconds until lift_joint changed by less than STOP_EPS_M over STOP_WINDOW_S, or None."""
        start, window = time.monotonic(), []
        while time.monotonic() - start < timeout:
            window.append((time.monotonic(), self.p.lift()))
            window = [(t, v) for t, v in window if t >= window[-1][0] - STOP_WINDOW_S]
            if window[-1][0] - window[0][0] >= STOP_WINDOW_S * 0.9 and max(v for _, v in window) - min(v for _, v in window) < STOP_EPS_M:
                return window[0][0] - start          # it was already still at the start of the window
            time.sleep(0.02)
        return None

    def test_l03_guard_killed_during_an_allowed_lift_move(self):
        self.assertTrue(self.p.wait_for_joints(STARTUP_WITHIN_S), "/joint_states never listed lift_joint and all 14 arm joints")
        self.assertIsNotNone(self.p.wait_allowed(STARTUP_WITHIN_S), f"the guard never allowed the lift (status {self.p.current_status()})")
        pids = guard_pids()
        self.record["guard_pids"] = pids
        self.assertTrue(pids, f"no process matching {INTERFACE['process_pattern']!r} in this ROS domain: nothing to kill")

        l0 = self.p.lift()
        target = l0 + 0.08
        accepted, handle, seconds = self.p.lift_goal(target)
        self.assertTrue(accepted, "the lift goal was rejected while the guard reported OK")
        time.sleep(KILL_AFTER_S)
        moving_at_kill = self.p.lift()
        self.assertGreater(moving_at_kill - l0, 0.005, "the lift was not moving when the guard was to be killed (precondition)")

        t_kill = time.monotonic()
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        inactive = self.p.wait_for(lambda: self.p.lift_controller_state() != "active", KILL_DEACTIVATE_WITHIN_S)
        still = self.wait_still(max(0.1, STOP_WITHIN_S - (time.monotonic() - t_kill)))
        still_after = None if still is None else round(time.monotonic() - t_kill - STOP_WINDOW_S, 2)
        goal_status, _ = self.p.result(handle, 5.0)
        stopped_at = self.p.lift()
        hold = max_excursion(self.p.sample_lift(HOLD_S), stopped_at)
        self.record.update(lift_from=l0, target=target, lift_at_kill=moving_at_kill, deactivated_after_s=None if inactive is None else round(inactive, 2),
                           still_after_s=still_after, travel_after_kill_m=round(stopped_at - moving_at_kill, 4), goal_status=goal_status,
                           stopped_at=stopped_at, hold_worst_m=round(hold, 5))

        self.assertIsNotNone(inactive, f"{LIFT_CONTROLLER} still active {KILL_DEACTIVATE_WITHIN_S:.0f} s after the guard was killed")
        self.assertIsNotNone(still, f"lift_joint still moving {STOP_WITHIN_S:.0f} s after the guard was killed")
        self.assertLess(stopped_at, target - 0.005, "the lift finished its goal although the guard was killed during it")
        self.assertLessEqual(hold, lift_still_m(PROFILE), f"no hold after the stop: moved {hold:.4f} m in {HOLD_S:.0f} s")

        accepted, _, _ = self.p.lift_goal(stopped_at + 0.03)
        self.p.lift_topic_command(stopped_at + 0.03, 2.0)
        after_cmds = max_excursion(self.p.sample_lift(3.0), stopped_at)
        self.record.update(goal_accepted_after_kill=accepted, moved_after_commands_m=round(after_cmds, 5))
        self.assertFalse(accepted, "a lift goal was accepted with the guard dead")
        self.assertLessEqual(after_cmds, lift_still_m(PROFILE), f"a command moved the lift {after_cmds:.4f} m with the guard dead")

        reactivated = self.p.switch(activate=[LIFT_CONTROLLER])
        replay = max_excursion(self.p.sample_lift(3.0), stopped_at)
        self.record.update(manual_reactivation_ok=reactivated, moved_after_reactivation_m=round(replay, 5))
        self.assertLessEqual(replay, lift_still_m(PROFILE), f"re-activation replayed a goal: the lift moved {replay:.4f} m")
        self.record["ok"] = True


class TestProcessesAlive(unittest.TestCase):
    def test_only_the_killed_guard_died(self, proc_info):
        """assert_no_process_died, with one exception this test made itself: the guard, killed by SIGKILL. Anything else that died fails."""
        try:
            assert_no_process_died(proc_info)
        except AssertionError:
            died = {name: getattr(proc_info[name], "returncode", None) for name in proc_info.process_names()}
            died = {n: c for n, c in died.items() if c not in (None, EXIT_OK)}
            others = {n: c for n, c in died.items() if not (n.startswith(INTERFACE["process_pattern"]) and c in GUARD_KILL_CODES)}
            self.assertFalse(others, f"processes other than the killed guard died during the test: {others}")


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_l03_processes_started_and_exited_cleanly(self, proc_info):
        assert_launch_healthy(proc_info, expected=EXPECTED_PROCESSES,
                              allowed={INTERFACE["process_pattern"]: GUARD_KILL_CODES, **MOVE_GROUP_ALLOWED})
