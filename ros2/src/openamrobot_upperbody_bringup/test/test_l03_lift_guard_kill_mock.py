# SPDX-License-Identifier: MIT
"""L03 (#44 acceptance 1): the guard process dies on `mock`.

Arguments (CMake registers several runs, like M01's lift_stop):
  kill_signal:=KILL|TERM    SIGKILL (no chance to clean up: only the launch's exit handler can act) or SIGTERM (an orderly exit)
  kill_phase:=moving|idle   during an allowed lift move, or while allowed and idle

Measured and checked, in this order:
  1  time from the signal to lift_controller inactive, and to the guard process exiting. Decided on #44 (10 Nov): on SIGTERM the node
     deactivates lift_controller ITSELF before it exits, so with TERM the deactivation must come no later than the exit (+0.3 s for event
     delivery; the OnProcessExit handler cannot act before the exit and needs over a second for the CLI); SIGKILL is the handler's case
  2  moving: time to lift_joint standing still, and the travel after the signal
  3  the hold: lift_joint stays put afterwards
  4  further commands are rejected: the action goal is not accepted and a topic trajectory does not move the lift
  5  re-activating lift_controller by hand does not replay the interrupted goal
  +  the run-wide safety properties (P1-P7) over everything recorded
  2  the stopping budget: time to inactive and lift travel after the signal, against OPENAMR_LIFT_GUARD_STOP_BUDGET_S / _M. The budget is
     NOT set yet: it comes from the lift speed in platform-fw #20, and the 19-30 mm measured on #53 is not acceptable as the final number
     (#44, 10 Nov; Evans, 11 Nov). Until it is set this case SKIPS with the measured values in its reason, so they stay visible and are
     never reported as a pass

The design does not claim protection against a HUNG guard (alive but not deciding); that is not tested here.

Its own launch test because it kills a process of the launch on purpose; here the guard alone may die, and only of the signal sent.
On main (no guard) it FAILs at its precondition: there is nothing to kill. Simulation only.
"""
import os
import random
import signal
import subprocess
import sys
import threading
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
SELF_DEACTIVATE_GRACE_S = 0.3     # TERM: inactive no later than the exit plus event delivery
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


def _alive(pid):
    """False once the process is gone or a zombie (exited, not yet reaped by launch)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def watch_exits(pids, t0, timeout):
    """Start a 5 ms watcher; the returned dict fills with {pid: seconds after t0 at which it exited}."""
    exits = {}

    def run():
        while len(exits) < len(pids) and time.monotonic() - t0 < timeout:
            for pid in pids:
                if pid not in exits and not _alive(pid):
                    exits[pid] = time.monotonic() - t0
            time.sleep(0.005)
    threading.Thread(target=run, daemon=True).start()
    return exits


@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestLiftGuardDies(GuardCase, unittest.TestCase):
    measured = None
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
        self.assertEqual(self.p.lift_controller_state(), "active", "lift_controller was not active before the kill (precondition)")
        t_kill = time.monotonic()
        exits = watch_exits(pids, t_kill, self.b.kill_deactivate + 10.0)
        for pid in pids:
            os.kill(pid, KILL_SIGNAL)
        inactive = self.p.wait_for(lambda: self.p.lift_controller_state() != "active", self.b.kill_deactivate)
        self.p.wait_for(lambda: len(exits) == len(pids), 5.0)
        exited = max(exits.values()) if len(exits) == len(pids) else None
        still = self.wait_still(max(0.5, self.b.kill_deactivate + 2.0 - (time.monotonic() - t_kill)))
        goal_status = self.p.result(handle, 5.0)[0] if handle is not None else None
        stopped = self.p.lift()
        t_hold = time.monotonic()
        time.sleep(HOLD_S)
        hold = self.p.lift_motion_since(t_hold)
        travel = abs(stopped - at_kill)
        self.case(name, lift_at_kill=at_kill, deactivated_after_s=None if inactive is None else round(inactive, 3),
                  exited_after_s=None if exited is None else round(exited, 3),
                  still_after_s=None if still is None else round(time.monotonic() - t_kill, 2), goal_status=goal_status,
                  travel_after_kill_m=round(travel, 5), stopped_at=stopped, hold_m=round(hold, 5))
        type(self).measured = {"signal": TAG, "deactivated_after_s": None if inactive is None else round(inactive, 3),
                               "exited_after_s": None if exited is None else round(exited, 3), "travel_after_kill_m": round(travel, 5)}
        self.assertIsNotNone(inactive, f"{LIFT_CONTROLLER} still active {self.b.kill_deactivate:.0f} s after the guard got {TAG}")
        self.assertIsNotNone(exited, f"the guard did not exit within 5 s of {LIFT_CONTROLLER} going inactive after {TAG}")
        if KILL_SIGNAL == signal.SIGTERM:
            self.assertLessEqual(inactive, exited + SELF_DEACTIVATE_GRACE_S,
                                 f"on SIGTERM {LIFT_CONTROLLER} went inactive {inactive:.2f} s after the signal but the guard exited at "
                                 f"{exited:.2f} s: it did not deactivate the controller itself before exiting (#44, 10 Nov), the launch's "
                                 "exit handler did")
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

    def test_l03_2_stopping_within_the_budget(self):
        """Skips, with the measurement in the reason, until the budget from platform-fw #20 is set in the environment."""
        m = type(self).measured
        budget = {"s": self.b.stop_budget_s, "m": self.b.stop_budget_m}
        self.record["stopping_budget"] = {**budget, "source": "environment" if any(v is not None for v in budget.values())
                                          else "not set: pending platform-fw #20 (#44, 10 Nov)"}
        if m is None or m["deactivated_after_s"] is None:
            self.skipTest("case 1 measured nothing to compare")
        if budget["s"] is None and budget["m"] is None:
            self.skipTest(f"stopping budget not set yet (platform-fw #20). Measured with {TAG}: inactive after "
                          f"{m['deactivated_after_s']} s, guard exited after {m['exited_after_s']} s, lift travel "
                          f"{m['travel_after_kill_m'] * 1000:.1f} mm")
        if budget["s"] is not None:
            self.assertLessEqual(m["deactivated_after_s"], budget["s"], f"{TAG}: inactive after {m['deactivated_after_s']} s, "
                                                                        f"budget {budget['s']} s")
        if budget["m"] is not None:
            self.assertLessEqual(m["travel_after_kill_m"], budget["m"],
                                 f"{TAG}: travel {m['travel_after_kill_m']} m, budget {budget['m']} m")

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
