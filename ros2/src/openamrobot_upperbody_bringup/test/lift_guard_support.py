# SPDX-License-Identifier: MIT
"""Shared support for the #44 lift guard acceptance tests (L02, L03, G03).

These are black-box tests: they talk to the running stack only through the ROS interface that the guard design on #44 publishes
(garvitkaushik-123, 9 Nov 2026) and the controllers' own interfaces. They do not import the guard, so any implementation of that interface
can be tested by them.

Everything the tests assume about the guard is named once, in INTERFACE and PROVISIONAL below. The values in PROVISIONAL are the design's
provisional defaults, not firmware values: they change when platform-fw #20 sets them (Evans, due 13 Nov 2026).

What the tests assert, in the words of the acceptance list on #44 (BotshareAI, 9 Nov 2026):
- lift motion is read from lift_joint in the reported joint state (/joint_states in mock, /arm/joint_states in Gazebo) and, in Gazebo, from
  the simulator's own link pose. Never from an action result (see #49: in DART a goal reported success while the lift did not move).
- the lift controller starts inactive and is active only while the guard reports OK.

Simulation only. Passing these says nothing about the robot's firmware interlock (CTL-017), which stays the authority.
"""
import math
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from diagnostic_msgs.msg import DiagnosticStatus
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ---- the guard's interface, as proposed on #44 (one place to change if it changes) -------------------------------------------------------
INTERFACE = {
    "status_topic": "/lift_guard/status",
    "status_name": "lift_guard",
    "key_controller": "lift_controller",      # value: active | inactive
    "key_blocking": "blocking",               # value: comma-separated codes, empty when OK
    "process_pattern": "lift_guard",          # how the kill test finds the guard process (pgrep -f)
}
CODES = ("ARM_NOT_STOWED_LEFT", "ARM_NOT_STOWED_RIGHT", "ARM_TRAJECTORY_ACTIVE_LEFT", "ARM_TRAJECTORY_ACTIVE_RIGHT", "ARM_STATE_MISSING",
         "ARM_STATE_STALE", "BASE_MOVING", "ODOM_MISSING", "ODOM_STALE", "SWITCH_FAILED")

# ---- provisional values (the design's defaults; not firmware values, see the module docstring) --------------------------------------------
PROVISIONAL = {
    "state_timeout_s": 0.5,
    "base_stopped_hold_s": 0.5,
    "status_rate_hz": 10.0,
    "stowed_tolerance_rad": 0.01,
}
# Budgets the tests allow, built from the values above plus the time a controller switch takes. Recorded with every run, so a failure says
# which budget it broke. They are test budgets, not requirements: the requirement is whatever #20 and the owners set.
BLOCK_WITHIN_S = PROVISIONAL["state_timeout_s"] + 2.0 / PROVISIONAL["status_rate_hz"] + 1.5   # condition fails -> controller inactive
ALLOW_WITHIN_S = PROVISIONAL["base_stopped_hold_s"] + PROVISIONAL["state_timeout_s"] + 3.0      # conditions hold -> controller active
KILL_DEACTIVATE_WITHIN_S = 10.0   # guard killed -> lift_controller inactive; includes the CLI start-up of the exit handler (measured, #44)

LIFT = "lift_joint"
LIFT_CONTROLLER = "lift_controller"
ARM_JOINTS = {side: [f"openarm_{side}_joint{i}" for i in range(1, 8)] for side in ("left", "right")}
LIFT_SPEED_MAX = 0.015            # m/s the tests ever ask for: lift_joint's limit is 0.018 m/s, which Gazebo enforces and mock does not
STATUS_NAMES = {v: k for k, v in GoalStatus.__dict__.items() if k.startswith("STATUS_")}


def lift_still_m(profile):
    """How far lift_joint may move and still count as 'did not move'. mock reflects commands exactly; Gazebo settled 0.14 to 1.8 mm on a
    cancel or a re-activation in the #44 baseline."""
    return 0.0005 if profile == "mock" else 0.003


def arm_state_topic(profile):
    """The topic the guard and these tests read the arms (and the lift) from. In Gazebo /joint_states is joint_state_publisher's merge,
    which keeps the arm joints fresh after their source stops (#44 baseline, row D), so the raw broadcaster output is read instead."""
    return "/joint_states" if profile == "mock" else "/arm/joint_states"


def dur(seconds):
    return Duration(sec=int(seconds), nanosec=int(round((seconds % 1) * 1e9)))


def parse_status(msg):
    values = {kv.key: kv.value for kv in msg.values}
    blocking = [c for c in values.get(INTERFACE["key_blocking"], "").split(",") if c]
    return {"level": msg.level, "ok": msg.level == DiagnosticStatus.OK, "controller": values.get(INTERFACE["key_controller"]),
            "blocking": blocking, "message": msg.message, "name": msg.name}


class GuardProbe:
    """A client of the stack with its own executor thread, so a test can sample state while a goal runs.

    Every reading carries the monotonic time it arrived; the tests measure latencies from those, not from the node clock, because a
    budget is a wall-time promise to whoever uses the simulator."""

    def __init__(self, profile, node_name="lift_guard_acceptance"):
        self.profile = profile
        self.node = rclpy.create_node(node_name)
        self.lock = threading.Lock()
        self.joints, self.joint_t = {}, {}
        self.status, self.status_t, self.status_log = None, None, []     # status_log: (t, parsed) on every change
        qos_latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(DiagnosticStatus, INTERFACE["status_topic"], self._on_status, qos_latched)
        self.node.create_subscription(JointState, arm_state_topic(profile), self._on_joints, 50)
        self.lift_ac = ActionClient(self.node, FollowJointTrajectory, f"/{LIFT_CONTROLLER}/follow_joint_trajectory")
        self.arm_ac = {s: ActionClient(self.node, FollowJointTrajectory, f"/{s}_arm_controller/follow_joint_trajectory") for s in ARM_JOINTS}
        self.lift_topic = self.node.create_publisher(JointTrajectory, f"/{LIFT_CONTROLLER}/joint_trajectory", 10)
        self.odom = self.node.create_publisher(Odometry, "/odom", 10)
        self.switch_cli = self.node.create_client(SwitchController, "/controller_manager/switch_controller")
        self.list_cli = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()

    # ---- callbacks ----------------------------------------------------------------------------------------------------------------------
    def _on_status(self, msg):
        parsed = parse_status(msg)
        now = time.monotonic()
        with self.lock:
            changed = self.status is None or (parsed["ok"], parsed["controller"], parsed["blocking"]) != (
                self.status["ok"], self.status["controller"], self.status["blocking"])
            self.status, self.status_t = parsed, now
            if changed:
                self.status_log.append((now, parsed))

    def _on_joints(self, msg):
        now = time.monotonic()
        with self.lock:
            for n, p in zip(msg.name, msg.position):
                self.joints[n] = p
                self.joint_t[n] = now

    # ---- readings -----------------------------------------------------------------------------------------------------------------------
    def lift(self):
        with self.lock:
            return self.joints.get(LIFT)

    def joint(self, name):
        with self.lock:
            return self.joints.get(name)

    def current_status(self):
        with self.lock:
            return self.status

    def wait_for(self, predicate, timeout):
        """-> seconds it took, or None if `predicate` never held within `timeout`."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if predicate():
                return time.monotonic() - start
            time.sleep(0.02)
        return None

    def wait_for_joints(self, timeout=60.0):
        need = [LIFT] + ARM_JOINTS["left"] + ARM_JOINTS["right"]
        return self.wait_for(lambda: all(self.joint(j) is not None for j in need), timeout) is not None

    def controller_states(self, timeout=5.0):
        if not self.list_cli.wait_for_service(timeout_sec=timeout):
            return {}
        fut = self.list_cli.call_async(ListControllers.Request())
        if self.wait_for(fut.done, timeout) is None or fut.result() is None:
            return {}
        return {c.name: c.state for c in fut.result().controller}

    def lift_controller_state(self):
        return self.controller_states().get(LIFT_CONTROLLER)

    def blocked(self):
        s = self.current_status()
        return s is not None and not s["ok"]

    def allowed(self):
        s = self.current_status()
        return s is not None and s["ok"]

    def wait_blocked(self, code=None, timeout=BLOCK_WITHIN_S):
        """Wait until the guard reports ERROR (with `code` among its blocking codes, if given) AND lift_controller is not active."""
        def done():
            s = self.current_status()
            return (s is not None and not s["ok"] and (code is None or code in s["blocking"])
                    and self.lift_controller_state() != "active")
        return self.wait_for(done, timeout)

    def wait_allowed(self, timeout=ALLOW_WITHIN_S):
        return self.wait_for(lambda: self.allowed() and self.lift_controller_state() == "active", timeout)

    def sample_lift(self, seconds, every=0.1):
        out, end = [], time.monotonic() + seconds
        while time.monotonic() < end:
            v = self.lift()
            out.append(float("nan") if v is None else v)
            time.sleep(every)
        return out

    # ---- commands -----------------------------------------------------------------------------------------------------------------------
    def _send(self, ac, goal, timeout=20.0):
        """-> (accepted, goal_handle or None). Does not wait for the result."""
        if not ac.wait_for_server(timeout_sec=timeout):
            return None, None
        fut = ac.send_goal_async(goal)
        if self.wait_for(fut.done, timeout) is None or fut.result() is None:
            return None, None
        return bool(fut.result().accepted), fut.result()

    def result(self, handle, timeout):
        """-> (status name, error_code) of a goal, or ("TIMEOUT", None)."""
        fut = handle.get_result_async()
        if self.wait_for(fut.done, timeout) is None:
            return "TIMEOUT", None
        return STATUS_NAMES.get(fut.result().status, fut.result().status), fut.result().result.error_code

    @staticmethod
    def traj(joints, points):
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = list(joints)
        for pos, t in points:
            g.trajectory.points.append(JointTrajectoryPoint(positions=list(pos), velocities=[0.0] * len(joints), time_from_start=dur(t)))
        return g

    def lift_goal(self, target, min_seconds=2.0):
        here = self.lift() or 0.0
        seconds = max(min_seconds, abs(target - here) / LIFT_SPEED_MAX)
        return self._send(self.lift_ac, self.traj([LIFT], [([target], seconds)])) + (seconds,)

    def lift_topic_command(self, target, seconds):
        msg = JointTrajectory(joint_names=[LIFT])
        msg.points = [JointTrajectoryPoint(positions=[target], velocities=[0.0], time_from_start=dur(seconds))]
        self.lift_topic.publish(msg)

    def arm_goal(self, side, positions, seconds=3.0, wait=True):
        accepted, handle = self._send(self.arm_ac[side], self.traj(ARM_JOINTS[side], [(positions, seconds)]))
        if wait and accepted:
            self.result(handle, seconds + 15.0)
        return accepted, handle

    def switch(self, activate=(), deactivate=(), timeout=15.0):
        if not self.switch_cli.wait_for_service(timeout_sec=timeout):
            return None
        req = SwitchController.Request(activate_controllers=list(activate), deactivate_controllers=list(deactivate),
                                       strictness=SwitchController.Request.BEST_EFFORT, activate_asap=True)
        fut = self.switch_cli.call_async(req)
        if self.wait_for(fut.done, timeout) is None or fut.result() is None:
            return None
        return bool(fut.result().ok)


def park_of(srdf_states, side):
    return [srdf_states[("park", f"{side}_arm")][j] for j in ARM_JOINTS[side]]


def srdf_states():
    """{(group_state, group): {joint: value}} from the installed SRDF the stack uses."""
    import os
    import xml.etree.ElementTree as ET
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory("openamrobot_upperbody_moveit"), "config", "openamrobot.srdf")
    out = {}
    for gs in ET.parse(path).getroot().iter("group_state"):
        out[(gs.get("name"), gs.get("group"))] = {j.get("name"): float(j.get("value")) for j in gs.iter("joint")}
    return out


def max_excursion(samples, reference):
    finite = [s for s in samples if not math.isnan(s)]
    return max((abs(s - reference) for s in finite), default=float("nan"))
