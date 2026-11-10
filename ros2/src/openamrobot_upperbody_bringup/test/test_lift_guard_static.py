# SPDX-License-Identifier: MIT
"""#44, static: the guard's scope as the design on #44 states it ("sim:=mock and sim:=gazebo only. No change to the real profile, the URDF
or lift_controller's config"), checked on the source tree. No ROS graph.

1. Both simulated launches start the guard and spawn lift_controller with --inactive (#44 acceptance 1: it starts inactive).
2. No other launch file of the five packages starts it (moveit.launch.py may name the status topic: the start-up goal waits for it).
3. The real profile's ros2_control description is untouched by the guard (it never mentions it).

Text checks, deliberately simple: they say where the guard is wired, not that it works (L02-L05 and G03 do that). On main they FAIL by design.
"""
import os
import re

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGES = [f"openamrobot_upperbody_{name}" for name in ("description", "control", "bringup", "gazebo", "moveit")]
SIM_LAUNCHES = {"mock": ("openamrobot_upperbody_bringup", "mock.launch.py"), "gazebo": ("openamrobot_upperbody_gazebo", "sim.launch.py")}
GUARD = re.compile(r"lift_guard")
STATUS_ONLY = re.compile(r"/lift_guard/status")


def read(*parts):
    with open(os.path.join(SRC, *parts)) as f:
        return f.read()


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
    assert GUARD.search(read(package, "launch", name)), f"{package}/launch/{name} does not start the lift guard ({profile})"


@pytest.mark.parametrize("profile", sorted(SIM_LAUNCHES))
def test_each_simulated_launch_spawns_lift_controller_inactive(profile):
    package, name = SIM_LAUNCHES[profile]
    text = read(package, "launch", name)
    assert "lift_controller" in text
    assert "--inactive" in text, f"{package}/launch/{name}: lift_controller is not spawned with --inactive (#44 acceptance 1)"


def test_no_other_launch_file_starts_the_guard():
    allowed = set(SIM_LAUNCHES.values())
    others = []
    for package, name in launch_files():
        if (package, name) in allowed:
            continue
        text = STATUS_ONLY.sub("", read(package, "launch", name))
        if GUARD.search(text):
            others.append(f"{package}/launch/{name}")
    assert not others, f"launch files other than the two simulated ones start or configure the guard: {others}"


def test_the_real_profile_description_does_not_mention_the_guard():
    urdf = os.path.join(SRC, "openamrobot_upperbody_description", "urdf")
    hits = [n for n in sorted(os.listdir(urdf)) if n.endswith(".xacro") and GUARD.search(read("openamrobot_upperbody_description", "urdf", n))]
    assert not hits, f"the description (shared by the real profile) mentions the guard: {hits}"
