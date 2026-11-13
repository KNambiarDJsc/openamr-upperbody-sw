# SPDX-License-Identifier: MIT
"""L03 (#44 acceptance 1): the guard process dies on `mock`.

Arguments (CMake registers several runs, like M01's lift_stop):
  kill_signal:=KILL|TERM    SIGKILL (no chance to clean up: only the launch's exit handler can act) or SIGTERM (an orderly exit)
  kill_phase:=moving|idle   during an allowed lift move, or while allowed and idle

Measured and checked, in this order:
  1  time from the signal to lift_controller inactive, and to the guard process exiting. Decided on #44 (10 Nov): on SIGTERM the node
     deactivates lift_controller ITSELF before it exits, so with TERM the deactivation must come no later than the exit (+0.3 s for event
     delivery; the OnProcessExit handler cannot act before the exit and needs over a second for the CLI); SIGKILL is the handler's case
     (from the SIGNAL to the guard, a process-death case; the time from an unsafe CONDITION to deactivation is measured in L02-10)
  2  moving: time to lift_joint standing still, and the travel after the signal
  3  the hold: lift_joint stays put afterwards
  4  further commands are rejected: the action goal is not accepted and a topic trajectory does not move the lift
  5  re-activating lift_controller by hand does not replay the interrupted goal
  6  nothing re-activates lift_controller between its deactivation and the test's own re-activation in 5: not the dying guard (a
     drive path still running during its exit, PR #53 review, 11 Nov) and nothing else
  7  the kill hit the guard ALONE: the mock odometry stub's process stayed alive from the signal to the end of the case, and /odom kept
     arriving, with no gap longer than the guard's own state_timeout_s (PR #60 review, 12 Nov: a substring selector killed both)
  +  the run-wide safety properties (P1-P7) over everything recorded
  2  the stopping budget: time to inactive and lift travel after the signal, against OPENAMR_LIFT_GUARD_STOP_BUDGET_S / _M. The budget is
     NOT set yet: it comes from the lift speed in platform-fw #20, and the 19-30 mm measured on #53 is not acceptable as the final number
     (#44, 10 Nov; Evans, 11 Nov). Until it is set this case SKIPS with the measured values in its reason, so they stay visible and are
     never reported as a pass

The design does not claim protection against a HUNG guard (alive but not deciding); that is not tested here.

Its own launch test because it kills a process of the launch on purpose; here the guard alone may die, and only of the signal sent.
The guard is the process that runs INTERFACE["guard_executable"] exactly (lift_guard.py on #53) in this test's ROS domain; see
lift_guard_support.program_names().
On main (no guard) it FAILs at its precondition: there is nothing to kill. Simulation only.
"""
import os
import random
import signal
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
from lift_guard_support import (INTERFACE, LIFT_CONTROLLER, SLACK, GuardCase, launch_process_is, lift_still_m, longest_gap,  # noqa: E402
                                pids_running, stack_under_test)

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
GUARD, STUB = INTERFACE["guard_executable"], INTERFACE["odom_stub_executable"]
EXPECTED_PROCESSES = ["robot_state_publisher", "ros2_control_node", "spawner", "move_group", GUARD, STUB]


@pytest.mark.launch_test
def generate_test_description():
    if not A7_ENABLED:
        return launch.LaunchDescription([launch.actions.ExecuteProcess(cmd=["sleep", "30"]), launch_testing.actions.ReadyToTest()]), {}
    stack, _ = stack_under_test(PROFILE)
    return launch.LaunchDescription([stack, launch_testing.actions.ReadyToTest()]), {}


def guard_pids():
    """PIDs of the guard in THIS test's ROS domain: the processes that run GUARD exactly. Not the odometry stub (lift_guard_odom_stub.py),
    not a launcher or shell that only mentions it, not a guard of another stack on the machine."""
    return pids_running(GUARD, os.environ["ROS_DOMAIN_ID"])


def odom_stub_pids():
    return pids_running(STUB, os.environ["ROS_DOMAIN_ID"])


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
        pids, stubs = guard_pids(), odom_stub_pids()
        self.case(name, signal=TAG, guard_executable=GUARD, guard_pids=pids, odom_stub_executable=STUB, odom_stub_pids=stubs)
        self.assertTrue(pids, f"no process runs {GUARD} in this ROS domain: nothing to kill")
        self.assertTrue(stubs, f"no process runs {STUB} in this ROS domain: mock needs its odometry stub, and this case must show that "
                               "it survives the kill")

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
        stub_exits = watch_exits(stubs, t_kill, 600.0)      # for the whole case: the stub must outlive all of it
        for pid in pids:
            os.kill(pid, KILL_SIGNAL)
        inactive = self.p.wait_for(lambda: self.p.lift_controller_state() != "active", self.b.kill_deactivate)
        self.p.wait_for(lambda: len(exits) == len(pids), 5.0)
        exited = max(exits.values()) if len(exits) == len(pids) else None
        if exited is not None:                  # the oracle judges the guard's statuses up to its exit (GuardCase.status_valid_until)
            type(self).status_valid_until = t_kill + exited
        self.assert_the_guard_alone_was_hit(name, stubs, stub_exits, t_kill)
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

        # 6: from its first deactivation after the signal until the test re-activates it here, lift_controller must stay off
        t_manual = time.monotonic()
        changes = [(round(t - t_kill, 3), st) for t, st in self.p.controller_changes(t_kill, t_manual)]
        first_off = next((i for i, (_, st) in enumerate(changes) if st != "active"), None)
        back_on = [c for c in changes[first_off + 1:] if c[1] == "active"] if first_off is not None else []
        self.case(name, controller_changes_after_kill=changes, reactivated_without_the_test=back_on)
        self.assertFalse(back_on, f"{LIFT_CONTROLLER} went active again after it was deactivated following {TAG}, before the test "
                                  f"re-activated it (s after the signal, state): {back_on}. Nothing may switch it back on while the guard "
                                  "is dying or dead (PR #53 review, 11 Nov)")

        reactivated = self.p.switch(activate=[LIFT_CONTROLLER])
        t_re = time.monotonic()
        time.sleep(3.0)
        replay = self.p.lift_motion_since(t_re)
        self.case(name, manual_reactivation_ok=reactivated, moved_after_reactivation_m=round(replay, 5))
        self.assertLessEqual(replay, lift_still_m(PROFILE), f"re-activation replayed a goal: the lift moved {replay:.4f} m")

        # 7: throughout, not only at the kill
        self.assert_the_guard_alone_was_hit(name, stubs, stub_exits, t_kill)
        self.passed(name)

    def assert_the_guard_alone_was_hit(self, name, stubs, stub_exits, t_kill):
        """The odometry stub's processes are alive and none has exited since the signal, and it has kept publishing /odom: no gap since the
        signal, by the samples' own stamps, longer than the guard's own state_timeout_s (times SLACK, the point at which the guard itself
        would call odometry stale) plus the backward wall-clock steps measured meanwhile (in mock the stub's timer runs on the wall clock,
        so a step holds it up by its size: 0.76 s and 1.07 s gaps on the WSL2 evidence host, 12 Nov). The gap by arrival here is recorded
        beside it (a busy test process is not a silent source). A dead stub leaves a gap that runs to the end of the case."""
        now = time.monotonic()
        dead = sorted(set(stub_exits) | {pid for pid in stubs if not _alive(pid)})
        gap = longest_gap(self.p.odom_times(published=True), t_kill, now)
        gap_arrival = longest_gap(self.p.odom_times(), t_kill, now)
        stepped = self.p.clock_stepped_back(t_kill - 1.0, now)
        gap_max = float(self.params["state_timeout_s"]) * SLACK + stepped
        self.case(name, odom_stub_exited=dead, odom_longest_gap_since_kill_s=round(gap, 3),
                  odom_longest_arrival_gap_since_kill_s=round(gap_arrival, 3), odom_gap_max_s=round(gap_max, 3),
                  wall_clock_stepped_back_s=round(stepped, 3), odom_checked_until_s=round(now - t_kill, 2))
        died_at = {pid: round(stub_exits[pid], 3) for pid in dead if pid in stub_exits}
        self.assertFalse(dead, f"the odometry stub ({STUB}, pids {dead}) died after the guard got {TAG} (s after the signal: {died_at}): "
                               "the kill must hit the guard alone")
        self.assertLessEqual(gap, gap_max, f"/odom stopped for {gap:.2f} s after the guard got {TAG} (allowed {gap_max:.2f} s: the guard's "
                                           f"state_timeout_s x SLACK, plus {stepped:.2f} s the wall clock stepped back): the odometry "
                                           "source did not survive the kill")

    def test_l03_2_stopping_within_the_budget(self):
        """Skips, with the measurement in the reason, until the budget from platform-fw #20 is set in the environment."""
        self.require_guard_parameters()
        m = type(self).measured
        budget = {"s": self.b.stop_budget_s, "m": self.b.stop_budget_m}
        self.record["stopping_budget"] = {**budget, "source": "environment" if any(v is not None for v in budget.values())
                                          else "not set: pending platform-fw #20 (#44, 10 Nov)"}
        if m is None or m["deactivated_after_s"] is None:
            self.skipTest("case 1 measured nothing to compare")
        if budget["s"] is None and budget["m"] is None:
            self.record["acceptance"]["pending"].append("stopping budget not set (platform-fw #20): measured, not checked")
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
            others = {n: c for n, c in died.items() if not (launch_process_is(n, GUARD) and c in GUARD_EXIT_CODES)}
            self.assertFalse(others, f"processes other than the killed guard died during the test: {others}")


@launch_testing.post_shutdown_test()
@unittest.skipUnless(A7_ENABLED, SKIP_REASON)
class TestShutdown(unittest.TestCase):
    def test_l03_processes_started_and_exited_cleanly(self, proc_info):
        assert_launch_healthy(proc_info, expected=EXPECTED_PROCESSES,
                              allowed={GUARD: GUARD_EXIT_CODES, **MOVE_GROUP_ALLOWED})
