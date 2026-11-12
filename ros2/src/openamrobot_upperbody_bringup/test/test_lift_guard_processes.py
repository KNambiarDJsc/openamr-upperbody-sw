# SPDX-License-Identifier: MIT
"""L03's process selection (PR #60 review, 12 Nov), without a ROS graph.

L03 kills the guard and checks that nothing else died. It found the guard by a substring ("lift_guard") in argv[0:2], which on #53's mock
wiring also matches the odometry stub (lift_guard_odom_stub.py): both were killed, and the exemption for the killed guard (any launch
process name starting with "lift_guard") hid the stub's death. Now the guard is the process that runs the guard's executable EXACTLY, in
the test's ROS domain, and only that launch process is exempt. These tests pin the selector to the command lines the launch produces.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift_guard_support import INTERFACE, launch_process_is, longest_gap, program_names, runs_executable  # noqa: E402

LIB = "/home/u/ws/install/openamrobot_upperbody_control/lib/openamrobot_upperbody_control"
ENV = {"ROS_DOMAIN_ID": "42"}
GUARD, STUB = "lift_guard.py", "lift_guard_odom_stub.py"

# What a launch_ros Node(executable=...) of a Python script runs: the shebang's interpreter, then the script, then --ros-args.
GUARD_ARGV = ["/usr/bin/python3", f"{LIB}/lift_guard.py", "--ros-args", "-r", "__node:=lift_guard", "--params-file", "/tmp/p.yaml"]
STUB_ARGV = ["/usr/bin/python3", f"{LIB}/lift_guard_odom_stub.py", "--ros-args", "-r", "__node:=lift_guard_odom_stub"]
NOT_THE_GUARD = {
    "the odometry stub (the review's case)": STUB_ARGV,
    "the start-up waiter": ["/usr/bin/python3", f"{LIB}/wait_for_lift_guard.py", "--ros-args"],
    "the park helper": ["/usr/bin/python3", f"{LIB}/park_arms.py", "--ros-args"],
    "ros2 run naming the guard": ["/usr/bin/python3", "/opt/ros/jazzy/bin/ros2", "run", "openamrobot_upperbody_control", "lift_guard.py"],
    "a shell mentioning it": ["/bin/bash", "-c", f"python3 {LIB}/lift_guard.py"],
    "the launch test itself": ["/usr/bin/python3", "/opt/ros/jazzy/bin/launch_test", "test_l03_lift_guard_kill_mock.py", "kill_signal:=TERM"],
    "the exit handler's CLI": ["/usr/bin/python3", "/opt/ros/jazzy/bin/ros2", "control", "switch_controllers", "--deactivate",
                               "lift_controller"],
    "a backup copy": ["/usr/bin/python3", f"{LIB}/lift_guard.py.orig"],
    "the stand-in guard": ["python3", "/mnt/d/harness/lift_guard_double/lift_guard_double.py", "--ros-args"],
}


def test_the_defaults_are_the_executables_of_53():
    assert INTERFACE["guard_executable"] == os.environ.get("LIFT_GUARD_EXECUTABLE", GUARD)
    assert INTERFACE["odom_stub_executable"] == os.environ.get("LIFT_GUARD_ODOM_STUB_EXECUTABLE", STUB)


@pytest.mark.parametrize("argv", [
    GUARD_ARGV,
    ["python3", f"{LIB}/lift_guard.py"],                                     # #!/usr/bin/env python3
    ["/usr/bin/python3.12", "-u", f"{LIB}/lift_guard.py", "--ros-args"],      # interpreter options before the script
    [f"{LIB}/lift_guard.py", "--ros-args"],                                   # run directly
])
def test_the_guard_is_selected(argv):
    assert runs_executable(argv, ENV, GUARD, 42)


@pytest.mark.parametrize("what", sorted(NOT_THE_GUARD))
def test_nothing_else_is_the_guard(what):
    assert not runs_executable(NOT_THE_GUARD[what], ENV, GUARD, 42), (what, program_names(NOT_THE_GUARD[what]))


def test_the_stub_is_found_as_itself_and_the_guard_is_not_the_stub():
    assert runs_executable(STUB_ARGV, ENV, STUB, 42)
    assert not runs_executable(GUARD_ARGV, ENV, STUB, 42)


@pytest.mark.parametrize("env", [{"ROS_DOMAIN_ID": "7"}, {}])
def test_a_guard_of_another_ros_domain_is_never_selected(env):
    assert not runs_executable(GUARD_ARGV, env, GUARD, 42)


def test_another_guard_can_be_named():
    assert runs_executable(NOT_THE_GUARD["the stand-in guard"], ENV, "lift_guard_double.py", 42)


@pytest.mark.parametrize("name, expected", [
    ("lift_guard.py-7", True), ("lift_guard.py-12", True),
    ("lift_guard_odom_stub.py-8", False), ("wait_for_lift_guard.py-9", False), ("lift_guard-7", False), ("lift_guard.py", False),
])
def test_only_the_guards_launch_process_is_exempt_from_the_survival_check(name, expected):
    assert launch_process_is(name, GUARD) is expected


@pytest.mark.parametrize("times, start, end, gap", [
    ([0.05 * i for i in range(201)], 0.0, 10.0, 0.05),       # 20 Hz throughout
    ([0.1, 0.2, 2.2, 2.3], 0.0, 2.4, 2.0),                    # stopped for 2 s in the middle
    ([0.1, 0.2], 0.0, 3.0, 2.8),                              # stopped for good: the gap runs to the end
    ([], 1.0, 4.0, 3.0),                                      # never arrived
    ([0.5, 5.0, 5.5], 4.0, 6.0, 1.0),                         # samples before the window do not count
])
def test_longest_gap(times, start, end, gap):
    assert longest_gap(times, start, end) == pytest.approx(gap)
