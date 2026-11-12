# SPDX-License-Identifier: MIT
"""#44, static: the guard's scope as the design on #44 states it ("sim:=mock and sim:=gazebo only. No change to the real profile, the URDF
or lift_controller's config"), checked on the source tree. No ROS graph.

1. Both simulated launches start the guard and spawn lift_controller with --inactive (#44 acceptance 1: it starts inactive). A launch
   file is read together with the modules of the five packages it imports: #53 shares the guard's wiring between mock and Gazebo in
   openamrobot_upperbody_control/lift_guard_launch.py (LIFT_SPAWNER_ARGUMENTS, lift_guard_actions()).
2. No other launch file of the five packages STARTS it: runs lift_guard.py or calls lift_guard_actions(). Naming the guard is fine:
   moveit.launch.py's start-up goal waits for its OK (wait_for_lift_guard.py), as decided on #44.
3. The real profile's ros2_control description is untouched by the guard (it never mentions it).
4. The zero-odom stub is started by the mock profile only (#44, 10 Nov: "mock only"; Gazebo has a simulated base whose /odom the guard
   reads, and the real profile has a real one). Assumes the stub's name contains "odom_stub", as on #44.

Text checks, deliberately simple: they say where the guard is wired, not that it works (L02-L05 and G03 do that). On main they FAIL by design.
"""
import os
import re

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGES = [f"openamrobot_upperbody_{name}" for name in ("description", "control", "bringup", "gazebo", "moveit")]
SIM_LAUNCHES = {"mock": ("openamrobot_upperbody_bringup", "mock.launch.py"), "gazebo": ("openamrobot_upperbody_gazebo", "sim.launch.py")}
GUARD = re.compile(r"lift_guard")
# Starting the guard: its executable as a string of its own ("lift_guard.py", not "wait_for_lift_guard.py"), or #53's shared actions.
GUARD_STARTED = re.compile(r"""["']lift_guard\.py["']|\blift_guard_actions\s*\(""")
IMPORTED = re.compile(r"^\s*from\s+(openamrobot_upperbody_\w+)\.(\w+)\s+import\b", re.M)
ODOM_STUB = re.compile(r"odom_stub")
MOCK_LAUNCH = "openamrobot_upperbody_bringup/launch/mock.launch.py"


def read(*parts):
    with open(os.path.join(SRC, *parts)) as f:
        return f.read()


def wiring(package, name):
    """A launch file's text, followed by the text of each module of the five packages that it imports."""
    text = read(package, "launch", name)
    for pkg, module in IMPORTED.findall(text):
        path = os.path.join(SRC, pkg, pkg, module + ".py")
        if pkg in PACKAGES and os.path.isfile(path):
            with open(path) as f:
                text += "\n" + f.read()
    return text


def launch_files():
    for p in PACKAGES:
        d = os.path.join(SRC, p, "launch")
        if os.path.isdir(d):
            for n in sorted(os.listdir(d)):
                if n.endswith(".py"):
                    yield p, n


@pytest.mark.parametrize("profile", sorted(SIM_LAUNCHES))
def test_each_simulated_launch_starts_the_guard(profile):
    package, name = SIM_LAUNCHES[profile]
    assert GUARD_STARTED.search(wiring(package, name)), f"{package}/launch/{name} does not start the lift guard ({profile})"


@pytest.mark.parametrize("profile", sorted(SIM_LAUNCHES))
def test_each_simulated_launch_spawns_lift_controller_inactive(profile):
    package, name = SIM_LAUNCHES[profile]
    assert "lift_controller" in read(package, "launch", name)
    assert "--inactive" in wiring(package, name), f"{package}/launch/{name}: lift_controller is not spawned with --inactive (#44 acceptance 1)"


def test_no_other_launch_file_starts_the_guard():
    allowed = set(SIM_LAUNCHES.values())
    others = []
    for package, name in launch_files():
        if (package, name) in allowed:
            continue
        if GUARD_STARTED.search(wiring(package, name)):
            others.append(f"{package}/launch/{name}")
    assert not others, f"launch files other than the two simulated ones start or configure the guard: {others}"


def test_the_real_profile_description_does_not_mention_the_guard():
    urdf = os.path.join(SRC, "openamrobot_upperbody_description", "urdf")
    hits = [n for n in sorted(os.listdir(urdf)) if n.endswith(".xacro") and GUARD.search(read("openamrobot_upperbody_description", "urdf", n))]
    assert not hits, f"the description (shared by the real profile) mentions the guard: {hits}"


def test_the_zero_odom_stub_is_started_by_the_mock_profile_only():
    hits = [f"{p}/launch/{n}" for p, n in launch_files() if ODOM_STUB.search(read(p, "launch", n))]
    assert hits, "no launch file starts the zero-odom stub: mock has no base, so without it the guard blocks the lift for ever"
    assert set(hits) <= {MOCK_LAUNCH}, f"the zero-odom stub is started outside the mock profile (#44: mock only): {hits}"
