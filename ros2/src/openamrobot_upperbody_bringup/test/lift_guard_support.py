# SPDX-License-Identifier: MIT
"""Shared support for the #44 lift guard acceptance tests (L02, L03, L04, L05, G03).

Black box: the tests talk to the running stack only through the interface the guard design on #44 publishes (garvitkaushik-123, 9 Nov
2026) and the controllers' own interfaces. They never import the guard, so any implementation of that interface can be tested.

Three things here make the tests more than a list of scripted checks:
- GuardProbe records the whole run into a lift_guard_oracle.Timeline (every lift_joint and arm sample, every status, every lift_controller
  transition), so the oracle can check the safety properties at every moment of the run, not only where a case looks.
- The thresholds are READ FROM THE RUNNING GUARD (its ROS parameters), so the tests follow the firmware values (#20) when they replace the
  provisional ones, and the boundary cases sit just inside and just outside whatever the guard is actually configured with. If the guard's
  parameters cannot be read, PROVISIONAL is used so that a baseline still runs, and the run is not acceptance (parameter_problems()).
- Every reaction budget is derived from those parameters (and recorded), so "as soon as one fails" is a measured number, not a sleep.

Simulation only. Nothing here is evidence about the firmware interlock (CTL-017), which stays the authority.

Not covered here, and owned elsewhere (#44, 12 Nov):
- Safe arm poses: the robot rule is stowed OR carry; the guard enables `park` only until the owners define carry (an SRDF state plus
  stowed_states). The oracle follows the guard's stowed_states and the SRDF, and L02-15 drives every state the guard declares; the cases
  that DRIVE the arms (start-up, 'out of park') use park, the only state defined today. Carry itself is untested until it exists.
- Base-stopped thresholds: read from the guard and tested on both sides of whatever it is configured with. Whether they equal the firmware
  values (D14, platform-fw #20) is open; this suite does not choose them.
- Stopping: the times and travel it records are measurements of the simulation, not accepted stopping limits (D16, platform-fw #20). The
  budgets below are the test's own tolerances, derived from the lift speed the tests ask for.
- A guard that is alive but not evaluating (D12 keep-alive/heartbeat) and base or arm motion while the lift moves (motion arbitration,
  platform-sw #87, manipulation #24) are not the guard's job and are not tested here.
"""
import contextlib
import math
import os
import re
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

from lift_guard_oracle import Timeline, stowed_deviation

# ---- the guard's interface, as proposed on #44 (one place to change if it changes) -------------------------------------------------------
INTERFACE = {
    "status_topic": "/lift_guard/status",
    "status_name": "lift_guard",
    "key_controller": "lift_controller",      # value: active | inactive | activating (WARN, switching on; #44 decision of 10 Nov)
    "key_blocking": "blocking",               # value: comma-separated codes, empty when OK
    # The guard's and the mock odometry stub's executables, matched EXACTLY (see program_names()). A substring ("lift_guard") also matched
    # lift_guard_odom_stub.py, so L03 killed the stub with the guard (PR #60 review, 12 Nov). The environment can name another guard
    # (the harness's stand-in); the defaults are #53's executables.
    "guard_executable": os.environ.get("LIFT_GUARD_EXECUTABLE", "lift_guard.py"),
    "odom_stub_executable": os.environ.get("LIFT_GUARD_ODOM_STUB_EXECUTABLE", "lift_guard_odom_stub.py"),
    # The mock zero-odom stub (#44 decision of 10 Nov: std_srvs/SetBool on this service, and a velocity parameter, mock only). The
    # service name is agreed; the node and parameter NAMES below were proposed by these tests and confirmed on #44 (11 Nov).
    # In mock they are required: L04 fails without them (until 12 Nov it skipped or published /odom itself).
    "odom_stub_enable_service": "/lift_guard_odom_stub/enable",
    "odom_stub_node": "/lift_guard_odom_stub",
    "odom_stub_velocity_parameters": ("linear_x", "angular_z"),   # m/s along x, rad/s about z, published in the twist
}
CODES = ("ARM_NOT_STOWED_LEFT", "ARM_NOT_STOWED_RIGHT", "ARM_TRAJECTORY_ACTIVE_LEFT", "ARM_TRAJECTORY_ACTIVE_RIGHT", "ARM_STATE_MISSING",
         "ARM_STATE_STALE", "BASE_MOVING", "ODOM_MISSING", "ODOM_STALE", "SWITCH_FAILED")
PARAMETERS = ("stowed_states", "stowed_tolerance", "reference_velocity_max", "arm_state_topic", "odom_topic", "base_linear_speed_max",
              "base_angular_speed_max", "base_stopped_hold_s", "state_timeout_s", "status_rate_hz")
# Read when the guard declares them, not part of the posted interface: #53 times out a switch request after switch_timeout_s (+1 s).
OPTIONAL_PARAMETERS = ("switch_timeout_s",)
# The design's provisional defaults (#44): used only when the guard's own parameters cannot be read. Not firmware values (#20).
PROVISIONAL = {"stowed_states": ["park"], "stowed_tolerance": 0.01, "reference_velocity_max": 0.001, "base_linear_speed_max": 0.01,
               "base_angular_speed_max": 0.01, "base_stopped_hold_s": 0.5, "state_timeout_s": 0.5, "status_rate_hz": 10.0,
               "switch_timeout_s": 2.0}

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


POSITIVE_NUMBERS = ("stowed_tolerance", "reference_velocity_max", "base_linear_speed_max", "base_angular_speed_max", "base_stopped_hold_s",
                    "state_timeout_s", "status_rate_hz")


def parameter_problems(read, unavailable=None, known_states=None):
    """Why the run cannot count as acceptance of the guard's configuration: [] only when every required parameter (PARAMETERS) was read
    from the running guard and has a usable value (PR #60 review: never pass on provisional values).

    `read`: {name: value} as read from the guard, or None when nothing could be read (`unavailable` says why). `known_states`: the SRDF
    group states that exist for BOTH arms; when given, every name in stowed_states must be one of them."""
    if read is None:
        return [unavailable or "the guard's parameters could not be read"]
    problems = [f"{name}: not set on the guard" for name in PARAMETERS if name not in read]
    problems += [p for p in (value_problem(name, v) for name, v in read.items()) if p]
    if known_states is not None and not value_problem("stowed_states", read.get("stowed_states")):
        unknown = [x for x in read["stowed_states"] if x not in known_states]
        if unknown:
            problems.append(f"stowed_states {unknown}: not SRDF group states of both arms")
    return problems


def value_problem(name, v):
    """What is wrong with one parameter value, or None if it is usable."""
    if name in POSITIVE_NUMBERS + OPTIONAL_PARAMETERS:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            return f"{name} = {v!r}: not a finite number > 0"
    elif name == "stowed_states":
        if isinstance(v, str) or not isinstance(v, (list, tuple)) or not v or not all(isinstance(x, str) and x for x in v):
            return f"stowed_states = {v!r}: not a non-empty list of state names"
    elif name in ("arm_state_topic", "odom_topic"):
        if not isinstance(v, str) or not v.strip():
            return f"{name} = {v!r}: not a topic name"
    return None


def srdf_states_of_both_arms(states):
    """Names of the group states that exist for left_arm AND right_arm in srdf_states()."""
    return {name for (name, group) in states if group == "left_arm"} & {name for (name, group) in states if group == "right_arm"}


_INTERPRETER = re.compile(r"python(\d+(\.\d+)*)?$")


def program_names(argv):
    """The executable a process runs, as basenames: argv[0], and when argv[0] is a Python interpreter, the script it runs (its first
    argument that is not an option). `/usr/bin/python3 -u .../lift_guard.py --ros-args ...` -> ["python3", "lift_guard.py"]. Arguments
    after the program are never looked at, so a launcher or a shell that merely MENTIONS the executable (`ros2 run <pkg> lift_guard.py`,
    `bash -c "... lift_guard.py"`, this test's own command line) is not that program."""
    if not argv:
        return []
    names = [os.path.basename(argv[0])]
    if _INTERPRETER.match(names[0]):
        script = next((a for a in argv[1:] if not a.startswith("-")), None)
        if script is not None:
            names.append(os.path.basename(script))
    return names


def runs_executable(argv, env, executable, domain_id):
    """True for a process that runs `executable` EXACTLY (a basename, e.g. "lift_guard.py": not lift_guard_odom_stub.py, not
    wait_for_lift_guard.py) in ROS domain `domain_id`, so a guard of another stack on the machine is never selected. `env`: str -> str."""
    return executable in program_names(argv) and env.get("ROS_DOMAIN_ID") == str(domain_id)


def pids_running(executable, domain_id):
    """PIDs of the processes that run `executable` in ROS domain `domain_id` (Linux /proc), never this process."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
            with open(f"/proc/{entry}/environ", "rb") as f:
                env = dict(kv.decode(errors="replace").split("=", 1) for kv in f.read().split(b"\0") if b"=" in kv)
        except OSError:                      # gone, or another user's process
            continue
        if runs_executable(argv, env, executable, domain_id):
            pids.append(int(entry))
    return sorted(pids)


def launch_process_is(name, executable):
    """Whether a launch_testing process name ("lift_guard.py-7": the executable's basename and a launch counter) is `executable`'s."""
    return re.fullmatch(re.escape(executable) + r"-\d+", name) is not None


def longest_gap(times, start, end):
    """Longest stretch in [start, end] with no sample (receipt times), counting the edges: no sample at all -> end - start."""
    inside = sorted(t for t in times if start <= t <= end)
    edges = [start] + inside + [end]
    return max(b - a for a, b in zip(edges, edges[1:]))


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
        # the TEST's tolerance for an arm leaving park (what the guard's reaction time allows at the speed the tests command), not a stopping limit
        self.travel_after_violation = LIFT_SPEED * self.pose_reaction + lift_still_m(profile)
        # WARN ("activating") must end: OK, or ERROR with SWITCH_FAILED once the guard's own switch timeout has run out
        self.warn_max = (float(params.get("switch_timeout_s", 2.0)) + 1.0 + 2 * tick + SWITCH_S) * SLACK
        # The stopping budget after the guard dies is NOT set yet: #44 (10 Nov) and Evans: it comes from the lift speed in platform-fw #20,
        # and the 19-30 mm measured on #53 is not acceptable as the final number. Until it is set, the kill test reports the measurement
        # and skips the budget check visibly instead of passing it.
        self.stop_budget_s = _env_float("OPENAMR_LIFT_GUARD_STOP_BUDGET_S")
        self.stop_budget_m = _env_float("OPENAMR_LIFT_GUARD_STOP_BUDGET_M")

    def as_dict(self):
        return {k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in vars(self).items()}


def _env_float(name):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def dur(seconds):
    return Duration(sec=int(seconds), nanosec=int(round((seconds % 1) * 1e9)))


def level_int(level):
    """diagnostic_msgs/DiagnosticStatus.level is a `byte`: rclpy hands it over as bytes (b'\x00'), not int."""
    return level[0] if isinstance(level, (bytes, bytearray)) else int(level)


def parse_status(msg):
    values = {kv.key: kv.value for kv in msg.values}
    blocking = tuple(c for c in values.get(INTERFACE["key_blocking"], "").split(",") if c)
    level = level_int(msg.level)
    return {"level": level, "ok": level == level_int(DiagnosticStatus.OK), "warn": level == level_int(DiagnosticStatus.WARN),
            "error": level == level_int(DiagnosticStatus.ERROR), "controller": values.get(INTERFACE["key_controller"]),
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


def stowed_poses(names, states):
    """The full arm pose of each named group state that exists for BOTH arms in `states` (srdf_states()): the poses the guard accepts as
    stowed when it is configured with `names`. Names that are not group states of both arms are left out (parameter_problems reports them)."""
    return [{j: states[(n, f"{side}_arm")][j] for side in ARM_JOINTS for j in ARM_JOINTS[side]} for n in names
            if all(j in states.get((n, f"{side}_arm"), {}) for side in ARM_JOINTS for j in ARM_JOINTS[side])]


class GuardProbe:
    """A client of the stack with its own executor thread, recording everything the oracle needs. All times are time.monotonic()."""

    def __init__(self, profile, node_name, park):
        self.profile = profile
        self.park = park
        self.stowed = [park]            # what the oracle calls stowed; follows the guard's stowed_states once they are read (use_stowed_states)
        self.node = rclpy.create_node(node_name)
        self.lock = threading.Lock()
        self.tl = Timeline()
        self.joints, self.joint_t = {}, {}
        self.status, self.status_t, self.status_count = None, None, 0
        self.status_changes = []                    # (t, parsed) on every change of (ok, controller, blocking)
        self.ctl_state, self.ctl_event_t = None, None
        self.ctl_counts = {"activity_events": 0, "polls_applied": 0, "polls_dropped": 0, "polls_without_lift": 0}
        self.t_created = time.monotonic()
        self.receipt_lags = []                      # s between a JointState's stamp and its receipt here (mock only)
        self.odom_samples = []                      # (t, |linear|, |angular|) of every /odom seen, for the hold-time measurement
        self.odom_published = []                    # the same samples, each at the time its source stamped it (mock; else receipt)
        # (t, step_s) whenever the wall clock moved against the monotonic one by more than 20 ms between two polls. In mock, ROS time is
        # the wall clock, so a step delays every ROS timer (a stub's /odom, the guard's tick) by its size; the WSL2 host used for the
        # evidence steps back 0.3-0.7 s several times a run (12 Nov), and the guard fails closed on it (*_STALE).
        self.clock_steps = []
        self._clock_offset = None
        self.blackouts = []                         # (t0, t1) spans in which a case stopped the joint states itself (joint_states_silenced)
        self.params, self.params_source = dict(PROVISIONAL), "provisional (guard parameters not read yet)"
        self.params_read, self.params_unavailable = None, "the guard's parameters were not read"   # see parameter_problems()
        self.state_timeout = PROVISIONAL["state_timeout_s"]

        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(DiagnosticStatus, INTERFACE["status_topic"], self._on_status, latched)
        self.node.create_subscription(JointState, arm_state_topic(profile), self._on_joints, 100)
        self.node.create_subscription(ControllerManagerActivity, "/controller_manager/activity", self._on_activity, 10)
        self.node.create_subscription(Odometry, "/odom", self._on_odom, 100)
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
        return {side: stowed_deviation(fresh, self.stowed, ARM_JOINTS[side]) for side in ARM_JOINTS}

    def use_stowed_states(self, names, states):
        """The oracle's definition of 'stowed' follows the guard's own stowed_states (read from it) and the SRDF, so it does not encode park as
        the permanent rule: the robot-level rule is stowed OR carry, park is today's conservative simulation state, and carry joins when the
        owners define it. Park stays the reference for what the cases DRIVE (park_side, start-up, 'out of park')."""
        poses = stowed_poses(names, states)
        if poses:
            self.stowed = poses

    def _on_joints(self, msg):
        """Each sample is placed at the moment it was MEASURED, not when this process got to it. The guard decides from the same samples,
        so a backlog here (a loaded runner, the status topic served first) must not make the oracle see a status before the sample that
        caused it (PR #53 run, 10 Nov: P3 flagged a status the guard issued 71 ms after it had seen the arm back in tolerance). The stamp is
        on the system clock in mock; Gazebo stamps are sim time, and receipt time is used there."""
        recv_mono, recv_wall = time.monotonic(), time.time()
        lag = recv_wall - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        now = recv_mono
        if 0.0 <= lag <= 2.0:
            now = recv_mono - lag
            self.receipt_lags.append(lag)
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

    def _on_odom(self, msg):
        tw = msg.twist.twist
        lin = math.sqrt(tw.linear.x ** 2 + tw.linear.y ** 2 + tw.linear.z ** 2)
        ang = math.sqrt(tw.angular.x ** 2 + tw.angular.y ** 2 + tw.angular.z ** 2)
        recv_mono, recv_wall = time.monotonic(), time.time()
        lag = recv_wall - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        with self.lock:
            self.odom_samples.append((recv_mono, lin, ang))
            del self.odom_samples[:-20000]
            # when the source PUBLISHED it (its stamp, system clock in mock), apart from when this process got to it (as _on_joints)
            self.odom_published.append(recv_mono - lag if 0.0 <= lag <= 2.0 else recv_mono)
            del self.odom_published[:-20000]

    def clock_stepped_back(self, start, end):
        """Total size (s, >= 0) of the backward wall-clock steps detected in [start, end]: how long a ROS timer may have been held up."""
        with self.lock:
            return -sum(s for t, s in self.clock_steps if start <= t <= end and s < 0)

    def odom_times(self, published=False):
        """Times of the /odom samples seen (the last 20000): when they arrived here, or with `published`, when their source stamped them."""
        with self.lock:
            return list(self.odom_published) if published else [t for t, _, _ in self.odom_samples]

    def controller_changes(self, start, end):
        """[(t, state)] of lift_controller's recorded state changes with start < t < end."""
        with self.lock:
            return [(t, s) for t, s in self.tl.controller if start < t < end]

    def last_moving_odom(self, linear_max, angular_max, after=0.0):
        """Receipt time of the last /odom sample at or above a threshold (or non-finite) after `after`, or None."""
        with self.lock:
            samples = list(self.odom_samples)
        moving = [t for t, lin, ang in samples
                  if t >= after and (not math.isfinite(lin) or not math.isfinite(ang) or lin >= linear_max or ang >= angular_max)]
        return moving[-1] if moving else None

    def _set_ctl(self, state, now, polled_since=None):
        """`polled_since`: a list_controllers answer to a request sent then. It is dropped if an activity event arrived after the request:
        the answer may describe the manager before that event (PR #53 run, 10 Nov: a stale 'inactive' landed after the activation event)."""
        with self.lock:
            if polled_since is not None and self.ctl_event_t is not None and self.ctl_event_t >= polled_since:
                self.ctl_counts["polls_dropped"] += 1
                return
            self.ctl_counts["activity_events" if polled_since is None else "polls_applied"] += 1
            if polled_since is None:
                self.ctl_event_t = now
            if state != self.ctl_state:
                self.ctl_state = state
                self.tl.controller.append((now, state))

    def _on_activity(self, msg):
        for c in msg.controllers:
            if c.name == LIFT_CONTROLLER:
                self._set_ctl(c.state.label, time.monotonic())

    def _poll(self):
        """5 Hz: lift_controller's state from the manager (a backstop for activity events missed at start-up), and, when arm state has gone
        silent, an arm sample, so that it shows up in the timeline as unreadable (deviation inf) instead of as the last value forever. A
        readable sample is not added here: it would place old joint values at the current time."""
        while not self._stop.is_set():
            t_req = time.monotonic()
            state = self.controller_states(timeout=1.0).get(LIFT_CONTROLLER)
            if state is not None:
                self._set_ctl(state, time.monotonic(), polled_since=t_req)
            else:
                self.ctl_counts["polls_without_lift"] += 1
            now = time.monotonic()
            offset = time.time() - now
            with self.lock:
                if self._clock_offset is not None and abs(offset - self._clock_offset) > 0.02:
                    self.clock_steps.append((now, round(offset - self._clock_offset, 3)))
                self._clock_offset = offset
                if self.joints:
                    devs = self._deviations(now)
                    if any(math.isinf(v) for v in devs.values()):
                        self.tl.arms.append((now, devs))
            self._stop.wait(0.2)

    def mark(self, label):
        with self.lock:
            self.tl.marks.append((time.monotonic(), label))

    @contextlib.contextmanager
    def joint_states_silenced(self):
        """A case that deactivates joint_state_broadcaster (so lift_joint is not sampled either) runs inside this and re-activates it
        before leaving. The span is declared to the oracle: its P1/P2 do not ask for lift samples inside it, and ask for them everywhere
        else. Anything else that silences lift_joint is a violation."""
        start = time.monotonic()
        try:
            yield
        finally:
            with self.lock:
                self.blackouts.append((start, time.monotonic()))

    def lift_sample_after(self, t):
        """True once a lift_joint sample has arrived after monotonic time `t`."""
        with self.lock:
            return self.joint_t.get(LIFT, -math.inf) > t

    def timeline(self):
        with self.lock:
            tl = Timeline(lift=sorted(self.tl.lift, key=lambda e: e[0]), arms=sorted(self.tl.arms, key=lambda e: e[0]),
                          status=list(self.tl.status), blackouts=list(self.blackouts),
                          controller=list(self.tl.controller), marks=list(self.tl.marks))
        tl.end = time.monotonic()
        return tl

    # ---- the guard's parameters ---------------------------------------------------------------------------------------------------------
    def read_guard_parameters(self, timeout=10.0):
        """Fill self.params from the running guard. -> True if read. Whatever could not be read stays PROVISIONAL so that a baseline can
        still run (main has no guard), but such a run is not acceptance: self.params_read / params_unavailable feed parameter_problems(),
        and GuardCase fails every case and the properties while it reports a problem."""
        from rclpy.parameter import parameter_value_to_python
        from rclpy.parameter_client import AsyncParameterClient
        deadline = time.monotonic() + timeout
        name = None
        # The guard is the node that PUBLISHES its status. (Matching node names was wrong: this test's own node is called l02_lift_guard.)
        while time.monotonic() < deadline and name is None:
            # An endpoint can be discovered before its node's name: "_NODE_NAME_UNKNOWN_" is not a guard to ask (11 Nov, found by the
            # acceptance gate: asking it failed and the run fell back to provisional values).
            pubs = [i for i in self.node.get_publishers_info_by_topic(INTERFACE["status_topic"])
                    if i.node_name != self.node.get_name() and "_UNKNOWN_" not in i.node_name and "_UNKNOWN_" not in i.node_namespace]
            if pubs:
                ns = pubs[0].node_namespace
                name = (ns.rstrip("/") + "/" + pubs[0].node_name) if ns not in ("", "/") else pubs[0].node_name
            else:
                time.sleep(0.2)
        if name is None:
            self.params_source = f"provisional (nothing publishes {INTERFACE['status_topic']})"
            self.params_unavailable = f"no node publishes {INTERFACE['status_topic']}: no guard to read parameters from"
            return False
        client = AsyncParameterClient(self.node, name)
        if not client.wait_for_services(timeout_sec=max(0.5, deadline - time.monotonic())):
            self.params_source = f"provisional ({name}: no parameter services)"
            self.params_unavailable = f"the guard ({name}) offers no parameter services"
            return False
        # One request per name, one at a time. rclpy answers a get_parameters that names ANY undeclared parameter with no values at all, so
        # asking for an optional name the guard does not declare (switch_timeout_s) hid every other value; and eleven requests at once
        # overflow a service's default request queue (depth 10), so one could go unanswered (both 11 Nov, found by the acceptance gate).
        read = {}
        for pname in list(PARAMETERS) + list(OPTIONAL_PARAMETERS):
            fut = client.get_parameters([pname])
            if self.wait_for(fut.done, max(5.0, deadline - time.monotonic())) is None or fut.result() is None:
                self.params_source = f"provisional ({name}: get_parameters did not answer for {pname})"
                self.params_unavailable = f"the guard ({name}) did not answer get_parameters for {pname}"
                return False
            if fut.result().values:
                v = parameter_value_to_python(fut.result().values[0])
                if v is not None:
                    read[pname] = v
        self.params_read = dict(read)
        self.params.update({k: v for k, v in read.items() if value_problem(k, v) is None})   # unusable values stay provisional
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
        if s is None or not s["error"]:                  # ERROR always means blocked (#44, 10 Nov); WARN is switching on
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
        """Peak-to-peak lift_joint over [t0, t1]. A window of half a second or more with fewer than two samples FAILS: 'the lift did not
        move' is not said of a lift that was not seen (it used to return 0.0)."""
        t1 = t1 or time.monotonic()
        with self.lock:
            vals = [p for t, p in self.tl.lift if t0 <= t <= t1 and not math.isnan(p)]
        if len(vals) < 2 and t1 - t0 >= 0.5:
            raise AssertionError(f"lift_joint was not observed: {len(vals)} sample(s) in {t1 - t0:.1f} s; 'did not move' would be vacuous")
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    def status_rate(self, seconds=3.0):
        n0 = self.status_count
        time.sleep(seconds)
        return (self.status_count - n0) / seconds

    def publisher_nodes(self, topic, settle=3.0):
        """Fully qualified names of the nodes publishing `topic`, once discovery has named them (a node can be seen before its name)."""
        def names():
            infos = self.node.get_publishers_info_by_topic(topic)
            return [f"{i.node_namespace.rstrip('/')}/{i.node_name}" for i in infos], infos
        self.wait_for(lambda: all("_UNKNOWN_" not in n for n in names()[0]) and names()[1], settle)
        return sorted(n for n in names()[0] if n != f"/{self.node.get_name()}")

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

    def odom_stub_velocity(self, linear, angular, timeout=3.0):
        """Set the mock stub's velocity parameters (INTERFACE). -> True, or None if the stub has no such parameters."""
        from rcl_interfaces.srv import SetParameters
        from rclpy.parameter import Parameter
        cli = self.node.create_client(SetParameters, INTERFACE["odom_stub_node"] + "/set_parameters")
        try:
            if not cli.wait_for_service(timeout_sec=timeout):
                return None
            names = INTERFACE["odom_stub_velocity_parameters"]
            req = SetParameters.Request(parameters=[Parameter(n, value=float(v)).to_parameter_msg() for n, v in zip(names, (linear, angular))])
            fut = cli.call_async(req)
            if self.wait_for(fut.done, timeout) is None or fut.result() is None:
                return None
            return True if all(r.successful for r in fut.result().results) else None
        finally:
            self.node.destroy_client(cli)

    @contextlib.contextmanager
    def base_motion(self, linear=0.0, angular=0.0):
        """The base 'moving' with this twist while the block runs, through the mock stub's velocity parameters (decided on #44, 10 Nov;
        names confirmed 11 Nov), so the guard sees one source of /odom. The stub is required: without it this FAILS the case. Until 12 Nov
        it fell back to publishing /odom itself, so the base cases could pass with no stub in the launch (PR #60 review). -> {"method"}"""
        if not self.odom_stub_velocity(linear, angular):
            raise AssertionError(f"the mock odometry stub {INTERFACE['odom_stub_node']} does not take "
                                 f"{INTERFACE['odom_stub_velocity_parameters']}: mock must start it (#44, 10 Nov)")
        try:
            yield {"method": "stub"}
        finally:
            self.odom_stub_velocity(0.0, 0.0)

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
        cls.params_problems = parameter_problems(cls.p.params_read, cls.p.params_unavailable, srdf_states_of_both_arms(srdf_states()))
        cls.params, cls.b = cls.p.params, cls.p.budgets()
        if not cls.params_problems:
            cls.p.use_stowed_states(cls.params["stowed_states"], srdf_states())
        cls.tol = float(cls.params["stowed_tolerance"])
        # The stack's own start-up must be over before a case drives anything: #53's launch moves both arms to park once (park_arms.py),
        # and a case's own park goal sent meanwhile preempted it, so the helper exited 1 and every survival check failed (12 Nov, first
        # run on #53's wiring). The guard's first OK is the end of start-up as the design defines it (arms stowed, no arm goal executing,
        # the lift allowed). Only observed, never asserted here (L02-00 judges start-up); without a guard (main's baseline) no wait.
        startup = cls.p.wait_allowed(STARTUP_WITHIN_S[cls.PROFILE]) if cls.p.params_read is not None else None
        cls.startup_complete_s = None if startup is None else round(time.monotonic() - cls.p.t_created, 3)
        cls.record = {"profile": cls.PROFILE, "stack_launch": os.environ.get("LIFT_GUARD_STACK_LAUNCH", "bringup moveit.launch.py"),
                      "double_mutant": os.environ.get("LIFT_GUARD_DOUBLE_MUTANT"), "interface": INTERFACE,
                      "parameters": {k: cls.params.get(k) for k in PARAMETERS}, "parameters_source": cls.p.params_source,
                      "budgets_s": cls.b.as_dict(), "slack": SLACK, "cases": {},
                      "startup_complete_after_probe_s": cls.startup_complete_s,
                      "acceptance": {"eligible": not cls.params_problems, "parameter_problems": cls.params_problems, "pending": [],
                                     "note": None if not cls.params_problems else
                                     "NOT ACCEPTANCE: provisional values were used; every case and the properties fail (PR #60 review)"}}

    @classmethod
    def tearDownClass(cls):
        from a7_support import write_run_sidecar
        cases = cls.record["cases"]
        ok = bool(cases) and all(c.get("ok") for c in cases.values())
        cls.record["status_changes"] = [{"t": round(t - cls.p.status_changes[0][0], 3), **s} for t, s in cls.p.status_changes][:400] \
            if cls.p.status_changes else []
        lags = sorted(cls.p.receipt_lags)
        t0 = cls.p.status_changes[0][0] if cls.p.status_changes else cls.p.t_created
        cls.record["controller_recording"] = {**cls.p.ctl_counts, "probe_created_s": round(cls.p.t_created - t0, 3),
                                              "first_entries": [(round(t - t0, 3), st) for t, st in cls.p.tl.controller[:12]]}
        from lift_guard_oracle import label_at
        tl = cls.p.timeline()
        cls.record["wall_clock_steps"] = [{"t": round(t - t0, 3), "step_s": s, "case": label_at(tl, t)} for t, s in cls.p.clock_steps][:100]
        cls.record["arm_state_receipt_lag_s"] = {"samples": len(lags), "p50": round(lags[len(lags) // 2], 4),
                                                 "p99": round(lags[int(len(lags) * 0.99)], 4),
                                                 "max": round(lags[-1], 4)} if lags else None
        ok = ok and not cls.params_problems
        pending = cls.record["acceptance"]["pending"]
        result = "FAIL" if not ok else "WARN" if pending else "PASS"       # a part that could not be checked yet is not a pass
        write_run_sidecar(cls.CHECK_ID, cls.SLUG, cls.NAME, result, jsonable(cls.record))
        cls.p.close()
        rclpy.shutdown()

    # ---- per-case record ------------------------------------------------------------------------------------------------------------
    def case(self, name, **fields):
        self.record["cases"].setdefault(name, {"ok": False}).update(fields)

    def require_guard_parameters(self):
        """The acceptance gate: a case, the properties or a budget may pass only on the guard's own parameters, read and valid."""
        self.assertFalse(self.params_problems, "not acceptance: the guard's own parameters were not read and validated, so this ran on "
                                               f"provisional values: {self.params_problems}")

    def passed(self, name):
        self.require_guard_parameters()
        self.record["cases"][name]["ok"] = True

    def test_a_guard_parameters_are_read_and_valid(self):
        """Its own case, first, so a report names the reason at the top; passed() and check_properties() enforce it as well."""
        self.record["cases"].setdefault("a_guard_parameters", {"ok": False}).update(problems=self.params_problems)
        self.require_guard_parameters()
        self.record["cases"]["a_guard_parameters"]["ok"] = True

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
    # Set by a case that ends the guard (L03): statuses received after the guard's process exited are not a live guard's statements.
    # A SIGKILLed writer's last samples can be held by reliable delivery and released seconds later (12 Nov, #53: two OK statuses
    # arrived 3.1 s after the kill, with the exit handler's deactivation), so the oracle judges statuses up to the exit only.
    status_valid_until = None

    def check_properties(self):
        from lift_guard_oracle import evaluate, label_at
        tl = self.p.timeline()
        if self.status_valid_until is not None:
            dropped = [e for e in tl.status if e[0] > self.status_valid_until]
            tl.status = [e for e in tl.status if e[0] <= self.status_valid_until]
            tl.status_end = self.status_valid_until            # and the last one stops being a statement there
            self.record["statuses_after_the_guard_exited"] = {"count": len(dropped), "judged": False,
                                                              "first": [(round(e[0] - self.status_valid_until, 3), e[1]) for e in dropped[:5]]}
        result = evaluate(tl, tolerance=self.tol, known_codes=CODES, settle=self.b.settle, eps=lift_still_m(self.PROFILE),
                          reaction=self.b.pose_reaction, lag=self.b.status_lag, warn_max=self.b.warn_max)
        self.record["properties"] = {k: {"violations": len(v), "first": v[:5]} for k, v in result.items()}
        # Context for each of the first violations: what the guard said, what lift_controller did and how the wall clock stepped around it
        context = []
        for v in [v for vs in result.values() for v in vs[:3]]:
            t = v.get("at", v.get("from"))
            if t is None:
                continue
            near = lambda events: [(round(e[0] - t, 3), *e[1:]) for e in events if abs(e[0] - t) <= 1.5]  # noqa: E731
            context.append({"property": v["property"], "case": label_at(tl, t), "statuses": near(tl.status)[:40],
                            "controller": near(tl.controller), "wall_clock_steps": near(self.p.clock_steps)})
        if context:
            self.record["violation_context"] = context
        from lift_guard_oracle import unobserved
        covered = unobserved(tl, tl.lift[0][0], tl.end, max_gap=0.0) if tl.lift else []
        self.record["lift_observation"] = {"samples": len(tl.lift), "arm_samples": len(tl.arms), "statuses": len(tl.status),
                                           "controller_changes": len(tl.controller),
                                           "longest_gap_s_outside_blackouts": max((g for _, _, g in covered), default=None),
                                           "declared_blackouts_s": [round(b - a, 2) for a, b in tl.blackouts]}
        broken = {k: v[:3] for k, v in result.items() if v}
        self.assertFalse(broken, f"safety properties broken during the run: {broken}")
        self.assertTrue(len(tl.lift) >= 2 and len(tl.arms) >= 2 and tl.status and tl.controller,
                        f"the run recorded too little for the properties to mean anything: {self.record['lift_observation']}")
        self.require_guard_parameters()
