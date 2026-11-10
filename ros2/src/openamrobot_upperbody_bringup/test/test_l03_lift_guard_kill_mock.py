# SPDX-License-Identifier: MIT
"""L03 (#44 acceptance 1): the guard process dies on `mock`.

Arguments (CMake registers several runs, like M01's lift_stop):
  kill_signal:=KILL|TERM    SIGKILL (no chance to clean up: only the launch's exit handler can act) or SIGTERM (an orderly exit)
  kill_phase:=moving|idle   during an allowed lift move, or while allowed and idle

Measured and checked, in this order:
  1  time from the signal to lift_controller inactive (the design's OnProcessExit handler; includes the `ros2 control` CLI start-up)
  2  moving: time to lift_joint standing still, and the travel after the signal
  3  the hold: lift_joint stays put afterwards
  4  further commands are rejected: the action goal is not accepted and a topic trajectory does not move the lift
  5  re-activating lift_controller by hand does not replay the interrupted goal
  +  the run-wide safety properties (P1-P6) over everything recorded

The design does not claim protection against a HUNG guard (alive but not deciding); that is not tested here.

Its own launch test because it kills a process of the launch on purpose; here the guard alone may die, and only of the signal sent.
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


def _argument(name, default):
    for argument in sys.argv[1:]:
        if argument.startswith(f"{name}:="):
            return argument.split(":=", 1)[1]
    return default


import launch  # noqa: E402
import launch_testing  # noqa: E402
import launch_testing.actions  # noqa: E402
import pytest  # noqa: E402
from ament_index_python.packages import get_package_prefix  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7_support import A7_ENABLED, SKIP_REASON  # noqa: E402
from lift_guard_support import INTERFACE, LIFT_CONTROLLER, GuardCase, lift_still_m, stack_under_test  # noqa: E402

sys.path.insert(0, os.path.join(get_package_prefix("openamrobot_upperbody_description"), "lib", "openamrobot_upperbody_description"))
from launch_guard import EXIT_OK, MOVE_GROUP_ALLOWED, assert_launch_healthy, assert_no_process_died  # noqa: E402

PROFILE = "mock"
KILL_SIGNAL = {"KILL": signal.SIGKILL, "TERM": signal.SIGTERM}[_argument("kill_signal", "KILL")]
KILL_PHASE = _argument("kill_phase", "moving")
TAG = f"{signal.Signals(KILL_SIGNAL).name}_{KILL_PHASE}"
KILL_AFTER_S = 1.5
STOP_WINDOW_S, STOP_EPS_M = 0.5, 0.0002
HOLD_S = 3.0
GUARD_EXIT_CODES = [-KILL_SIGNAL, 0, 1] if KILL_SIGNAL == signal.SIGTERM else [-KILL_SIGNAL]
EXPECTED_PROCESSES = ["robot_state_publisher", "ros2_control_node", "spawner", "move_group", INTERFACE["process_pattern"]]


@pytest.mark.launch_test
def generate_test_description():
    if not A7_ENABLED:
        return launch.LaunchDescription([launch.actions.ExecuteProcess(cmd=["sleep", "30"]), launch_testing.actions.ReadyToTest()]), {}
    stack, _ = stack_under_test(PROFILE)
    return launch.LaunchDescription([stack, launch_testing.actions.ReadyToTest()]), {}


def guard_pids():
    """PIDs of the guard in THIS test's ROS domain. The pattern must be in argv[0] or argv[1] (the program, or the script a python3 runs),
    so a process whose ARGUMENTS merely mention it (the test runner itself, a shell) is never matched; and its environment must carry this
    test's ROS_DOMAIN_ID, so a guard of another stack on the machine is never killed."""
    mine = []
    for pid in subprocess.run(["pgrep", "-f", INTERFACE["process_pattern"]], capture_output=True, text=True).stdout.split():
        pid = int(pid)
        if pid == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = dict(kv.split(b"=", 1) for kv in f.read().split(b"\0") if b"=" in kv)
        except OSError:
            continue
        head = [os.path.basename(a) for a in argv[:2]]
        if any(INTERFACE["process_pattern"] in a for a in head) and env.get(b"ROS_DOMAIN_ID") == os.environ["ROS_DOMAIN_ID"].encode():
            mine.append(pid)
    return mine


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardDies(GuardCase, unittest.TestCase):
    PROFILE = PROFILE
    NODE = "l03_lift_guard_kill"
    CHECK_ID, SLUG, NAME = "L03", f"lift_guard_kill_{TAG}", f"#44 lift guard dies ({TAG}) on mock"

    def wait_still(self, timeout):
        """-> seconds until lift_joint changed by less than STOP_EPS_M over STOP_WINDOW_S, or None."""
        start, window = time.monotonic(), []
        while time.monotonic() - start < timeout:
            window.append((time.monotonic(), self.p.lift()))
            window = [(t, v) for t, v in window if t >= window[-1][0] - STOP_WINDOW_S]
            vals = [v for _, v in window if v is not None]
            if window[-1][0] - window[0][0] >= STOP_WINDOW_S * 0.9 and vals and max(vals) - min(vals) < STOP_EPS_M:
                return window[0][0] - start
            time.sleep(0.02)
        return None

    def test_l03_1_the_guard_dies(self):
        name = f"1_{TAG}"
        pre = self.start(name)
        self.assertTrue(pre["allowed"], f"the guard never allowed the lift (status {self.p.current_status()}): nothing to test")
        pids = guard_pids()
        self.case(name, signal=TAG, guard_pids=pids)
        self.assertTrue(pids, f"no process with {INTERFACE['process_pattern']!r} in argv[0:2] in this ROS domain: nothing to kill")

        l0 = self.p.lift()
        handle, seconds = None, 0.0
        if KILL_PHASE == "moving":
            accepted, handle, seconds = self.p.lift_goal(l0 + 0.08)
            self.assertTrue(accepted, "the lift goal was rejected while the guard allowed the lift")
            time.sleep(KILL_AFTER_S)
            self.assertGreater(self.p.lift() - l0, 0.005, "the lift was not moving when the guard was to be killed (precondition)")
        at_kill = self.p.lift()
        t_kill = time.monotonic()
        for pid in pids:
            os.kill(pid, KILL_SIGNAL)
        inactive = self.p.wait_for(lambda: self.p.lift_controller_state() != "active", self.b.kill_deactivate)
        still = self.wait_still(max(0.5, self.b.kill_deactivate + 2.0 - (time.monotonic() - t_kill)))
        goal_status = self.p.result(handle, 5.0)[0] if handle is not None else None
        stopped = self.p.lift()
        t_hold = time.monotonic()
        time.sleep(HOLD_S)
        hold = self.p.lift_motion_since(t_hold)
        self.case(name, lift_at_kill=at_kill, deactivated_after_s=None if inactive is None else round(inactive, 3),
                  still_after_s=None if still is None else round(time.monotonic() - t_kill, 2), goal_status=goal_status,
                  travel_after_kill_m=round(abs(stopped - at_kill), 5), stopped_at=stopped, hold_m=round(hold, 5))
        self.assertIsNotNone(inactive, f"{LIFT_CONTROLLER} still active {self.b.kill_deactivate:.0f} s after the guard got {TAG}")
        self.assertIsNotNone(still, "lift_joint still moving after the guard died")
        if KILL_PHASE == "moving":
            self.assertLess(stopped, l0 + 0.08 - 0.005, "the lift finished its goal although the guard died during it")
        self.assertLessEqual(hold, lift_still_m(PROFILE), f"no hold after the stop: moved {hold:.4f} m in {HOLD_S:.0f} s")

        accepted, t0 = self.try_lift_both_paths(0.03)
        after = self.p.lift_motion_since(t0)
        self.case(name, goal_accepted_after_death=accepted, moved_after_commands_m=round(after, 5))
        self.assertFalse(accepted, "a lift goal was accepted with the guard dead")
        self.assertLessEqual(after, lift_still_m(PROFILE), f"a command moved the lift {after:.4f} m with the guard dead")

        reactivated = self.p.switch(activate=[LIFT_CONTROLLER])
        t_re = time.monotonic()
        time.sleep(3.0)
        replay = self.p.lift_motion_since(t_re)
        self.case(name, manual_reactivation_ok=reactivated, moved_after_reactivation_m=round(replay, 5))
        self.assertLessEqual(replay, lift_still_m(PROFILE), f"re-activation replayed a goal: the lift moved {replay:.4f} m")
        self.passed(name)

    def test_l03_zz_safety_properties_held_throughout_the_run(self):
        self.check_properties()


class TestProcessesAlive(unittest.TestCase):
    def test_only_the_killed_guard_died(self, proc_info):
        """assert_no_process_died, with one exception this test made itself: the guard, ended by the signal it was sent."""
        try:
            assert_no_process_died(proc_info)
        except AssertionError:
            died = {name: getattr(proc_info[name], "returncode", None) for name in proc_info.process_names()}
            died = {n: c for n, c in died.items() if c not in (None, EXIT_OK)}
            others = {n: c for n, c in died.items() if not (n.startswith(INTERFACE["process_pattern"]) and c in GUARD_EXIT_CODES)}
            self.assertFalse(others, f"processes other than the killed guard died during the test: {others}")


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_l03_processes_started_and_exited_cleanly(self, proc_info):
        assert_launch_healthy(proc_info, expected=EXPECTED_PROCESSES,
                              allowed={INTERFACE["process_pattern"]: GUARD_EXIT_CODES, **MOVE_GROUP_ALLOWED})
