# SPDX-License-Identifier: MIT
"""Shared support for the #44 lift guard acceptance tests (L02, L03, L04, L05, G03).

Black box: the tests talk to the running stack only through the interface the guard design on #44 publishes (garvitkaushik-123, 9 Nov
2026) and the controllers' own interfaces. They never import the guard, so any implementation of that interface can be tested.

Three things here make the tests more than a list of scripted checks:
- GuardProbe records the whole run into a lift_guard_oracle.Timeline (every lift_joint and arm sample, every status, every lift_controller
  transition), so the oracle can check the safety properties at every moment of the run, not only where a case looks.
- The thresholds are READ FROM THE RUNNING GUARD (its ROS parameters), so the tests follow the firmware values (#20) when they replace the
  provisional ones, and the boundary cases sit just inside and just outside whatever the guard is actually configured with. If the guard's
  parameters cannot be read, PROVISIONAL is used and the sidecar says so.
- Every reaction budget is derived from those parameters (and recorded), so "as soon as one fails" is a measured number, not a sleep.

Simulation only. Nothing here is evidence about the firmware interlock (CTL-017), which stays the authority.
"""
import contextlib
import math
import os
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.msg import ControllerManagerActivity
from controller_manager_msgs.srv import ListControllers, SwitchController
from diagnostic_msgs.msg import DiagnosticStatus
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lift_guard_oracle import Timeline, arm_deviation

# ---- the guard's interface, as proposed on #44 (one place to change if it changes) -------------------------------------------------------
INTERFACE = {
    "status_topic": "/lift_guard/status",
    "status_name": "lift_guard",
    "key_controller": "lift_controller",      # value: active | inactive
    "key_blocking": "blocking",               # value: comma-separated codes, empty when OK
    "process_pattern": "lift_guard",          # how the kill test finds the guard process
    # PROPOSED on #44, not agreed yet: a way for a test to stop the mock zero-odom stub. The odom-loss cases skip while it is absent.
    "odom_stub_enable_service": "/lift_guard_odom_stub/enable",
}
CODES = ("ARM_NOT_STOWED_LEFT", "ARM_NOT_STOWED_RIGHT", "ARM_TRAJECTORY_ACTIVE_LEFT", "ARM_TRAJECTORY_ACTIVE_RIGHT", "ARM_STATE_MISSING",
         "ARM_STATE_STALE", "BASE_MOVING", "ODOM_MISSING", "ODOM_STALE", "SWITCH_FAILED")
PARAMETERS = ("stowed_states", "stowed_tolerance", "reference_velocity_max", "arm_state_topic", "odom_topic", "base_linear_speed_max",
              "base_angular_speed_max", "base_stopped_hold_s", "state_timeout_s", "status_rate_hz")
# The design's provisional defaults (#44): used only when the guard's own parameters cannot be read. Not firmware values (#20).
PROVISIONAL = {"stowed_states": ["park"], "stowed_tolerance": 0.01, "reference_velocity_max": 0.001, "base_linear_speed_max": 0.01,
               "base_angular_speed_max": 0.01, "base_stopped_hold_s": 0.5, "state_timeout_s": 0.5, "status_rate_hz": 10.0}

LIFT = "lift_joint"
LIFT_CONTROLLER = "lift_controller"
ARM_JOINTS = {side: [f"openarm_{side}_joint{i}" for i in range(1, 8)] for side in ("left", "right")}
ALL_ARM_JOINTS = ARM_JOINTS["left"] + ARM_JOINTS["right"]
LIFT_SPEED = 0.015          # m/s PEAK the tests ever ask for: lift_joint's limit is 0.018 m/s, which Gazebo enforces and mock does not
PEAK_OVER_MEAN = 1.5        # a one-point JointTrajectoryController goal from rest to rest peaks at 1.5x its mean speed (cubic spline);
                            # measured: an "0.015 m/s" goal moved 37 mm in 1.74 s on mock (21 mm/s) before this factor was applied
START_HEIGHT = 0.10         # m: every case starts here, so that motion both ways is possible and no case runs into a limit
SWITCH_S = 0.5              # s a controller switch through the controller manager may take (activity event to state change)
SLACK = float(os.environ.get("OPENAMR_LIFT_GUARD_SLACK", "1.5"))   # multiplies every budget; raise it on a slow CI runner, never lower
STATUS_NAMES = {v: k for k, v in GoalStatus.__dict__.items() if k.startswith("STATUS_")}


def lift_still_m(profile):
    """How far lift_joint may move and still count as 'did not move'. mock reflects commands exactly; Gazebo settled 0.14 to 1.8 mm on a
    cancel or a re-activation in the #44 baseline."""
    return 0.0005 if profile == "mock" else 0.003


def arm_state_topic(profile):
    """In Gazebo /joint_states is joint_state_publisher's merge, which keeps the arm joints fresh after their source stops (#44 baseline,
    row D), so the raw broadcaster output is what the guard and these tests read."""
    return "/joint_states" if profile == "mock" else "/arm/joint_states"


class Budgets:
    """Reaction budgets derived from the guard's parameters (or PROVISIONAL), times SLACK."""

    def __init__(self, params, profile):
        tick = 1.0 / float(params["status_rate_hz"])
        self.pose_reaction = (2 * tick + SWITCH_S) * SLACK                                  # an arm leaves park -> blocked and inactive
        self.base_reaction = (2 * tick + SWITCH_S) * SLACK                                  # odom over the threshold -> blocked
        self.stale_reaction = (params["state_timeout_s"] + 2 * tick + SWITCH_S) * SLACK      # arm state stops -> blocked
        self.allow_within = (params["base_stopped_hold_s"] + params["state_timeout_s"] + 2 * tick + SWITCH_S) * SLACK + 1.0
        self.kill_deactivate = 10.0 * SLACK   # includes the start-up of the `ros2 control` CLI in the design's exit handler
        self.status_lag = 2 * tick + 0.3
        self.settle = 0.3 if profile == "mock" else 0.6                                     # after a block, before "did not move" applies
        self.travel_after_violation = LIFT_SPEED * self.pose_reaction + lift_still_m(profile)

    def as_dict(self):
        return {k: round(v, 3) for k, v in vars(self).items()}


def dur(seconds):
    return Duration(sec=int(seconds), nanosec=int(round((seconds % 1) * 1e9)))


def level_int(level):
    """diagnostic_msgs/DiagnosticStatus.level is a `byte`: rclpy hands it over as bytes (b'\x00'), not int."""
    return level[0] if isinstance(level, (bytes, bytearray)) else int(level)


def parse_status(msg):
    values = {kv.key: kv.value for kv in msg.values}
    blocking = tuple(c for c in values.get(INTERFACE["key_blocking"], "").split(",") if c)
    level = level_int(msg.level)
    return {"level": level, "ok": level == level_int(DiagnosticStatus.OK), "controller": values.get(INTERFACE["key_controller"]),
            "blocking": blocking, "message": msg.message, "name": msg.name, "keys": sorted(values)}


def srdf_states():
    """{(group_state, group): {joint: value}} from the installed SRDF the stack uses."""
    import xml.etree.ElementTree as ET
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory("openamrobot_upperbody_moveit"), "config", "openamrobot.srdf")
    out = {}
    for gs in ET.parse(path).getroot().iter("group_state"):
        out[(gs.get("name"), gs.get("group"))] = {j.get("name"): float(j.get("value")) for j in gs.iter("joint")}
    return out


def park_positions(states):
    return {j: states[("park", f"{side}_arm")][j] for side in ARM_JOINTS for j in ARM_JOINTS[side]}


class GuardProbe:
    """A client of the stack with its own executor thread, recording everything the oracle needs. All times are time.monotonic()."""

    def __init__(self, profile, node_name, park):
        self.profile = profile
        self.park = park
        self.node = rclpy.create_node(node_name)
        self.lock = threading.Lock()
        self.tl = Timeline()
        self.joints, self.joint_t = {}, {}
        self.status, self.status_t, self.status_count = None, None, 0
        self.status_changes = []                    # (t, parsed) on every change of (ok, controller, blocking)
        self.ctl_state = None
        self.params, self.params_source = dict(PROVISIONAL), "provisional (guard parameters not read yet)"
        self.state_timeout = PROVISIONAL["state_timeout_s"]

        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(DiagnosticStatus, INTERFACE["status_topic"], self._on_status, latched)
        self.node.create_subscription(JointState, arm_state_topic(profile), self._on_joints, 100)
        self.node.create_subscription(ControllerManagerActivity, "/controller_manager/activity", self._on_activity, 10)
        self.lift_ac = ActionClient(self.node, FollowJointTrajectory, f"/{LIFT_CONTROLLER}/follow_joint_trajectory")
        self.arm_ac = {s: ActionClient(self.node, FollowJointTrajectory, f"/{s}_arm_controller/follow_joint_trajectory") for s in ARM_JOINTS}
        self.lift_topic = self.node.create_publisher(JointTrajectory, f"/{LIFT_CONTROLLER}/joint_trajectory", 10)
        self.arm_topic = {s: self.node.create_publisher(JointTrajectory, f"/{s}_arm_controller/joint_trajectory", 10) for s in ARM_JOINTS}
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.joint_pub = self.node.create_publisher(JointState, arm_state_topic(profile), 10)
        self.switch_cli = self.node.create_client(SwitchController, "/controller_manager/switch_controller")
        self.list_cli = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        self.executor = MultiThreadedExecutor(num_threads=6)
        self.executor.add_node(self.node)
        self._stop = threading.Event()
        threading.Thread(target=self.executor.spin, daemon=True).start()
        threading.Thread(target=self._poll, daemon=True).start()

    def close(self):
        self._stop.set()
        self.executor.shutdown()
        self.node.destroy_node()

    # ---- recording ----------------------------------------------------------------------------------------------------------------------
    def _deviations(self, now):
        fresh = {j: p for j, p in self.joints.items() if now - self.joint_t.get(j, -1e9) <= self.state_timeout}
        return {side: arm_deviation(fresh, self.park, ARM_JOINTS[side]) for side in ARM_JOINTS}

    def _on_joints(self, msg):
        now = time.monotonic()
        with self.lock:
            for n, p in zip(msg.name, msg.position):
                self.joints[n] = p
                self.joint_t[n] = now
            if LIFT in msg.name:
                self.tl.lift.append((now, self.joints[LIFT]))
            self.tl.arms.append((now, self._deviations(now)))

    def _on_status(self, msg):
        parsed = parse_status(msg)
        now = time.monotonic()
        with self.lock:
            prev = self.status
            self.status, self.status_t = parsed, now
            self.status_count += 1
            self.tl.status.append((now, parsed["level"], parsed["controller"], parsed["blocking"]))
            if prev is None or (parsed["ok"], parsed["controller"], parsed["blocking"]) != (prev["ok"], prev["controller"], prev["blocking"]):
                self.status_changes.append((now, parsed))

    def _set_ctl(self, state, now):
        with self.lock:
            if state != self.ctl_state:
                self.ctl_state = state
                self.tl.controller.append((now, state))

    def _on_activity(self, msg):
        for c in msg.controllers:
            if c.name == LIFT_CONTROLLER:
                self._set_ctl(c.state.label, time.monotonic())

    def _poll(self):
        """5 Hz: lift_controller's state from the manager (a backstop for activity events missed at start-up), and an arm sample, so that
        arm state going silent shows up in the timeline as unreadable (deviation inf) instead of as the last value forever."""
        while not self._stop.is_set():
            state = self.controller_states(timeout=1.0).get(LIFT_CONTROLLER)
            if state is not None:
                self._set_ctl(state, time.monotonic())
            now = time.monotonic()
            with self.lock:
                if self.joints:
                    self.tl.arms.append((now, self._deviations(now)))
            self._stop.wait(0.2)

    def mark(self, label):
        with self.lock:
            self.tl.marks.append((time.monotonic(), label))

    def timeline(self):
        with self.lock:
            tl = Timeline(lift=list(self.tl.lift), arms=sorted(self.tl.arms, key=lambda e: e[0]), status=list(self.tl.status),
                          controller=list(self.tl.controller), marks=list(self.tl.marks))
        tl.end = time.monotonic()
        return tl

    # ---- the guard's parameters ---------------------------------------------------------------------------------------------------------
    def read_guard_parameters(self, timeout=10.0):
        """Fill self.params from the running guard. -> True if read. Falls back to PROVISIONAL (recorded in params_source)."""
        from rclpy.parameter import parameter_value_to_python
        from rclpy.parameter_client import AsyncParameterClient
        deadline = time.monotonic() + timeout
        name = None
        # The guard is the node that PUBLISHES its status. (Matching node names was wrong: this test's own node is called l02_lift_guard.)
        while time.monotonic() < deadline and name is None:
            pubs = [i for i in self.node.get_publishers_info_by_topic(INTERFACE["status_topic"]) if i.node_name != self.node.get_name()]
            if pubs:
                ns = pubs[0].node_namespace
                name = (ns.rstrip("/") + "/" + pubs[0].node_name) if ns not in ("", "/") else pubs[0].node_name
            else:
                time.sleep(0.2)
        if name is None:
            self.params_source = f"provisional (nothing publishes {INTERFACE['status_topic']})"
            return False
        client = AsyncParameterClient(self.node, name)
        if not client.wait_for_services(timeout_sec=max(0.5, deadline - time.monotonic())):
            self.params_source = f"provisional ({name}: no parameter services)"
            return False
        fut = client.get_parameters(list(PARAMETERS))
        if self.wait_for(fut.done, 5.0) is None or fut.result() is None:
            self.params_source = f"provisional ({name}: get_parameters did not answer)"
            return False
        read = {}
        for pname, value in zip(PARAMETERS, fut.result().values):
            v = parameter_value_to_python(value)
            if v is not None:
                read[pname] = v
        self.params.update(read)
        self.state_timeout = float(self.params["state_timeout_s"])
        missing = [p for p in PARAMETERS if p not in read]
        self.params_source = f"guard node {name}" + (f" (not set there, provisional used: {missing})" if missing else "")
        return True

    def budgets(self):
        return Budgets(self.params, self.profile)

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
            time.sleep(0.01)
        return None

    def wait_for_joints(self, timeout=60.0):
        need = [LIFT] + ALL_ARM_JOINTS
        return self.wait_for(lambda: all(self.joint(j) is not None for j in need), timeout) is not None

    def controller_states(self, timeout=5.0):
        if not self.list_cli.wait_for_service(timeout_sec=timeout):
            return {}
        fut = self.list_cli.call_async(ListControllers.Request())
        if self.wait_for(fut.done, timeout) is None or fut.result() is None:
            return {}
        return {c.name: c.state for c in fut.result().controller}

    def lift_controller_state(self):
        with self.lock:
            return self.ctl_state

    def blocked(self, code=None, any_of=None):
        s = self.current_status()
        if s is None or s["ok"]:
            return False
        if code is not None and code not in s["blocking"]:
            return False
        if any_of is not None and not set(any_of) & set(s["blocking"]):
            return False
        return True

    def wait_blocked(self, code=None, any_of=None, timeout=10.0):
        """Until the guard reports ERROR (with `code`, or one of `any_of`) AND lift_controller is not active. -> seconds or None."""
        return self.wait_for(lambda: self.blocked(code, any_of) and self.lift_controller_state() != "active", timeout)

    def wait_allowed(self, timeout):
        return self.wait_for(lambda: (self.current_status() or {}).get("ok") and self.lift_controller_state() == "active", timeout)

    def stays(self, predicate, seconds):
        """True if `predicate` holds at every check for `seconds`."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if not predicate():
                return False
            time.sleep(0.01)
        return True

    def lift_motion_since(self, t0, t1=None):
        t1 = t1 or time.monotonic()
        with self.lock:
            vals = [p for t, p in self.tl.lift if t0 <= t <= t1 and not math.isnan(p)]
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    def status_rate(self, seconds=3.0):
        n0 = self.status_count
        time.sleep(seconds)
        return (self.status_count - n0) / seconds

    def status_publisher_qos(self):
        return [(i.node_name, i.qos_profile.durability, i.qos_profile.reliability)
                for i in self.node.get_publishers_info_by_topic(INTERFACE["status_topic"])]

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
        """-> (accepted, handle, seconds)."""
        here = self.lift() if self.lift() is not None else 0.0
        seconds = max(min_seconds, PEAK_OVER_MEAN * abs(target - here) / LIFT_SPEED)
        accepted, handle = self._send(self.lift_ac, self.traj([LIFT], [([target], seconds)]))
        return accepted, handle, seconds

    def lift_goal_and_wait(self, target):
        accepted, handle, seconds = self.lift_goal(target)
        status = self.result(handle, seconds + 15.0)[0] if accepted else None
        return accepted, status

    def lift_topic_command(self, target, seconds):
        msg = JointTrajectory(joint_names=[LIFT])
        msg.points = [JointTrajectoryPoint(positions=[target], velocities=[0.0], time_from_start=dur(seconds))]
        self.lift_topic.publish(msg)

    def arm_goal(self, side, positions, seconds=2.0, wait=True, points=None):
        goal = self.traj(ARM_JOINTS[side], points or [(positions, seconds)])
        accepted, handle = self._send(self.arm_ac[side], goal)
        if wait and accepted:
            self.result(handle, (points[-1][1] if points else seconds) + 15.0)
        return accepted, handle

    def arm_topic_command(self, side, positions, seconds):
        msg = JointTrajectory(joint_names=list(ARM_JOINTS[side]))
        msg.points = [JointTrajectoryPoint(positions=list(positions), velocities=[0.0] * 7, time_from_start=dur(seconds))]
        self.arm_topic[side].publish(msg)

    def park_side(self, side):
        return [self.park[j] for j in ARM_JOINTS[side]]

    def park_both(self, seconds=2.0):
        for side in ARM_JOINTS:
            self.arm_goal(side, self.park_side(side), seconds=seconds)

    def switch(self, activate=(), deactivate=(), timeout=15.0):
        if not self.switch_cli.wait_for_service(timeout_sec=timeout):
            return None
        req = SwitchController.Request(activate_controllers=list(activate), deactivate_controllers=list(deactivate),
                                       strictness=SwitchController.Request.BEST_EFFORT, activate_asap=True)
        fut = self.switch_cli.call_async(req)
        if self.wait_for(fut.done, timeout) is None or fut.result() is None:
            return None
        return bool(fut.result().ok)

    # ---- injection ----------------------------------------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def inject_odom(self, linear=0.0, angular=0.0, rate=50.0):
        """Publish /odom with this twist at `rate` while the block runs. The mock zero-odom stub keeps publishing too; the guard's
        'continuously stopped for base_stopped_hold_s' rule means one moving sample must be enough to block."""
        stop, sent = threading.Event(), {"last": None}
        msg = Odometry()
        msg.header.frame_id, msg.child_frame_id = "odom", "base_footprint"
        msg.twist.twist.linear.x, msg.twist.twist.angular.z = float(linear), float(angular)

        def run():
            while not stop.is_set():
                msg.header.stamp = self.node.get_clock().now().to_msg()
                self.odom_pub.publish(msg)
                sent["last"] = time.monotonic()
                time.sleep(1.0 / rate)
        th = threading.Thread(target=run, daemon=True)
        th.start()
        try:
            yield sent
        finally:
            stop.set()
            th.join(2.0)

    @contextlib.contextmanager
    def inject_arm_states(self, positions, rate=50.0):
        """Publish JointState on the arm state topic with `positions` ({joint: value}; a callable is re-read every message) while the block
        runs. Used with joint_state_broadcaster deactivated, so the guard sees only these."""
        stop = threading.Event()

        def run():
            while not stop.is_set():
                pos = positions() if callable(positions) else positions
                msg = JointState(name=list(pos), position=[float(v) for v in pos.values()])
                msg.header.stamp = self.node.get_clock().now().to_msg()
                self.joint_pub.publish(msg)
                time.sleep(1.0 / rate)
        th = threading.Thread(target=run, daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join(2.0)

    def odom_stub(self, enable, timeout=3.0):
        """The PROPOSED stub switch (INTERFACE['odom_stub_enable_service']). -> True/False, or None if the service does not exist."""
        from std_srvs.srv import SetBool
        cli = self.node.create_client(SetBool, INTERFACE["odom_stub_enable_service"])
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(SetBool.Request(data=bool(enable)))
        if self.wait_for(fut.done, timeout) is None or fut.result() is None:
            return None
        return bool(fut.result().success)

    # ---- one known start state for every case -------------------------------------------------------------------------------------------
    def reset(self, label, allow_timeout):
        """Both arms in park, joint_state_broadcaster active, lift at START_HEIGHT, the guard asked to allow. -> {"allowed": bool, ...}.
        Never asserts: on main (no guard) a case must still reach its physical check, so that the red baseline fails because the lift
        MOVED, not because a precondition did."""
        self.mark(label)
        if self.lift_controller_state() is None:
            self.wait_for(lambda: self.lift_controller_state() is not None, 5.0)
        self.switch(activate=["joint_state_broadcaster"])
        self.wait_for(lambda: self.lift() is not None, 5.0)
        self.park_both()
        allowed = self.wait_allowed(allow_timeout) is not None
        here = self.lift()
        if here is not None and abs(here - START_HEIGHT) > 0.002:
            accepted, _ = self.lift_goal_and_wait(START_HEIGHT)
            allowed = self.wait_allowed(allow_timeout) is not None
        return {"allowed": allowed, "lift": self.lift()}


# ---- the stack under test, and the common test case ------------------------------------------------------------------------------------
STARTUP_WITHIN_S = {"mock": 120.0, "gazebo": 300.0}


def stack_under_test(profile, **args):
    """The launch the tests start: bringup's moveit.launch.py, or LIFT_GUARD_STACK_LAUNCH if set (a launch file taking the same arguments,
    e.g. a guard branch's own wiring, or the test-double wrapper used to validate these tests). -> (IncludeLaunchDescription, path)."""
    import launch
    from ament_index_python.packages import get_package_share_directory
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    path = os.environ.get("LIFT_GUARD_STACK_LAUNCH") or os.path.join(
        get_package_share_directory("openamrobot_upperbody_bringup"), "launch", "moveit.launch.py")
    launch_args = {"sim": profile, "rviz": "false", **{k: str(v) for k, v in args.items()}}
    return launch.actions.IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=launch_args.items()), path


def joint_limits():
    """{joint: (lower, upper)} of the arm joints, from the mock expansion (as L01 does)."""
    import xml.etree.ElementTree as ET
    import xacro
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory("openamrobot_upperbody_description"), "urdf", "combined_robot.xacro")
    root = ET.fromstring(xacro.process_file(path, mappings={"hardware_profile": "mock"}).toxml())
    return {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
            for j in root.findall("joint") if j.get("name") in ALL_ARM_JOINTS}


def jsonable(value):
    """The record as strict JSON: non-finite floats as strings, bytes as ints, tuples as lists (the evidence reader is strict)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class GuardCase:
    """Mixin for a unittest.TestCase class. Subclasses set PROFILE, NODE, CHECK_ID, SLUG, NAME. Provides the probe (self.p), the guard's
    parameters and budgets (self.params, self.b), one record per case, and the run-wide safety properties (check_properties)."""
    PROFILE = "mock"
    NODE = "lift_guard_acceptance"
    CHECK_ID, SLUG, NAME = "LXX", "lift_guard", "#44 lift guard acceptance"

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.park = park_positions(srdf_states())
        cls.p = GuardProbe(cls.PROFILE, cls.NODE, cls.park)
        cls.joints_ok = cls.p.wait_for_joints(STARTUP_WITHIN_S[cls.PROFILE])
        cls.p.read_guard_parameters(timeout=30.0 if cls.PROFILE == "mock" else 90.0)
        cls.params, cls.b = cls.p.params, cls.p.budgets()
        cls.tol = float(cls.params["stowed_tolerance"])
        cls.record = {"profile": cls.PROFILE, "stack_launch": os.environ.get("LIFT_GUARD_STACK_LAUNCH", "bringup moveit.launch.py"),
                      "double_mutant": os.environ.get("LIFT_GUARD_DOUBLE_MUTANT"), "interface": INTERFACE,
                      "parameters": {k: cls.params.get(k) for k in PARAMETERS}, "parameters_source": cls.p.params_source,
                      "budgets_s": cls.b.as_dict(), "slack": SLACK, "cases": {}}

    @classmethod
    def tearDownClass(cls):
        from a7_support import write_run_sidecar
        cases = cls.record["cases"]
        ok = bool(cases) and all(c.get("ok") for c in cases.values())
        cls.record["status_changes"] = [{"t": round(t - cls.p.status_changes[0][0], 3), **s} for t, s in cls.p.status_changes][:400] \
            if cls.p.status_changes else []
        write_run_sidecar(cls.CHECK_ID, cls.SLUG, cls.NAME, "PASS" if ok else "FAIL", jsonable(cls.record))
        cls.p.close()
        rclpy.shutdown()

    # ---- per-case record ------------------------------------------------------------------------------------------------------------
    def case(self, name, **fields):
        self.record["cases"].setdefault(name, {"ok": False}).update(fields)

    def passed(self, name):
        self.record["cases"][name]["ok"] = True

    def start(self, name):
        pre = self.p.reset(name, self.b.allow_within)
        self.case(name, allowed_at_start=pre["allowed"], lift_at_start=pre["lift"])
        return pre

    # ---- assertions -------------------------------------------------------------------------------------------------------------------
    def assert_blocked(self, what, code=None, any_of=None, within=None):
        within = within or self.b.pose_reaction
        took = self.p.wait_blocked(code, any_of, within)
        want = code or (" or ".join(any_of) if any_of else "any code")
        self.assertIsNotNone(took, f"{what}: not blocked with {want} and {LIFT_CONTROLLER} inactive within {within:.2f} s "
                                   f"(status {self.p.current_status()}, {LIFT_CONTROLLER} {self.p.lift_controller_state()})")
        return round(took, 3)

    def assert_allowed(self, what, within=None):
        within = within or self.b.allow_within
        took = self.p.wait_allowed(within)
        self.assertIsNotNone(took, f"{what}: the guard did not allow the lift within {within:.2f} s "
                                   f"(status {self.p.current_status()}, {LIFT_CONTROLLER} {self.p.lift_controller_state()})")
        return round(took, 3)

    def assert_stays_allowed(self, what, seconds):
        ok = self.p.stays(lambda: (self.p.current_status() or {}).get("ok") and self.p.lift_controller_state() == "active", seconds)
        self.assertTrue(ok, f"{what}: the guard did not stay allowing for {seconds:.1f} s (status {self.p.current_status()})")

    def assert_lift_still(self, t0, what, seconds=0.0):
        if seconds:
            time.sleep(seconds)
        moved = self.p.lift_motion_since(t0)
        self.assertLessEqual(moved, lift_still_m(self.PROFILE), f"{what}: lift_joint moved {moved:.4f} m")
        return round(moved, 5)

    def try_lift_both_paths(self, delta=0.03):
        """Send a lift goal (action) and a lift trajectory (topic), let them run, cancel an accepted goal. -> (accepted, t0)."""
        t0 = time.monotonic()
        here = self.p.lift() if self.p.lift() is not None else START_HEIGHT
        accepted, handle, _ = self.p.lift_goal(here + delta)
        self.p.lift_topic_command(here - delta, 2.0)
        time.sleep(2.5)
        if accepted and handle is not None:
            handle.cancel_goal_async()
        return accepted, t0

    # ---- the run-wide properties ------------------------------------------------------------------------------------------------------
    def check_properties(self):
        from lift_guard_oracle import evaluate
        result = evaluate(self.p.timeline(), tolerance=self.tol, known_codes=CODES, settle=self.b.settle, eps=lift_still_m(self.PROFILE),
                          reaction=self.b.pose_reaction, lag=self.b.status_lag)
        self.record["properties"] = {k: {"violations": len(v), "first": v[:5]} for k, v in result.items()}
        broken = {k: v[:3] for k, v in result.items() if v}
        self.assertFalse(broken, f"safety properties broken during the run: {broken}")
