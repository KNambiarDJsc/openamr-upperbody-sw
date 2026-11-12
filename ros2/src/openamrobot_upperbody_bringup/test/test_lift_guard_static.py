# SPDX-License-Identifier: MIT
"""#44, static: where the lift guard is wired, checked on the launch descriptions the source tree builds. No ROS graph, no process runs.

The design on #44: "sim:=mock and sim:=gazebo only. No change to the real profile, the URDF or lift_controller's config".
1. Each simulated launch instantiates exactly one guard Node (lift_guard.py), configured for its profile, with an OnProcessExit on that
   very Node that deactivates lift_controller (SIGKILL; on SIGTERM the guard does it itself).
2. The lift_controller spawner the launch actually starts gets --inactive, and nothing at start-up activates it.
3. The odometry stub Node runs in mock exactly once, and nothing named like a stub runs in Gazebo.
4. No other launch file of the five packages starts the guard or the stub (evaluated, and checked on its syntax tree for branches the
   evaluation cannot reach).

How: each launch file is imported from THIS source tree (with this tree's openamrobot_upperbody_control first on sys.path, so the module
under review is what runs), generate_launch_description() is called, and the result is walked with the launch library itself: launch
arguments take their defaults, conditions are evaluated, OpaqueFunctions are executed, includes are recorded (not followed), and the
OnProcessExit handlers on spawners are expanded as if each spawner exited 0, which is how the controller chain starts. The walker's own
behaviour is pinned by the tests at the end of this file.

PR #60 review (12 Nov): the text searches used until then passed on a docstring, on the helper's definition and on an unused constant.
What this proves: what the launch would start. Not that it runs: L02-L05 and G03 check that at runtime. On main these FAIL by design.
"""
import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, field

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
CONTROL = "openamrobot_upperbody_control"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SRC, CONTROL))       # the PR's own lift_guard_launch, not an installed copy

from launch import LaunchContext, LaunchDescription  # noqa: E402
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, OpaqueFunction  # noqa: E402
from launch.actions import RegisterEventHandler  # noqa: E402
from launch.event_handlers import OnProcessExit  # noqa: E402
from launch.events.process import ProcessExited  # noqa: E402
from launch.utilities import perform_substitutions  # noqa: E402
from launch_ros.actions import Node  # noqa: E402

from lift_guard_support import INTERFACE, LIFT_CONTROLLER, arm_state_topic  # noqa: E402

PACKAGES = [f"openamrobot_upperbody_{name}" for name in ("description", "control", "bringup", "gazebo", "moveit")]
SIM_LAUNCHES = {"mock": ("openamrobot_upperbody_bringup", "mock.launch.py"), "gazebo": ("openamrobot_upperbody_gazebo", "sim.launch.py")}
# Launch files outside the two simulated ones, with the launch arguments each is evaluated with (moveit.launch.py chooses its stack in an
# OpaqueFunction, so both of its branches are evaluated).
OTHER_LAUNCHES = {("openamrobot_upperbody_bringup", "moveit.launch.py"): [{"sim": "mock"}, {"sim": "gazebo"}]}
GUARD, STUB = INTERFACE["guard_executable"], INTERFACE["odom_stub_executable"]


# ---- the walker ------------------------------------------------------------------------------------------------------------------------
@dataclass
class Launch:
    context: LaunchContext
    nodes: list = field(default_factory=list)
    processes: list = field(default_factory=list)        # ExecuteProcess that are not Nodes
    includes: list = field(default_factory=list)         # resolved paths of IncludeLaunchDescription
    exit_handlers: list = field(default_factory=list)    # OnProcessExit


def text(substitutions):
    """A plain-text launch value; a substitution that is not plain text shows as <Name>."""
    if isinstance(substitutions, str):
        return substitutions
    return "".join(getattr(s, "text", f"<{type(s).__name__}>") for s in substitutions)


def walk(entities, launch):
    for e in entities:
        if isinstance(e, LaunchDescription):
            walk(e.entities, launch)
            continue
        condition = getattr(e, "condition", None)
        if condition is not None and not condition.evaluate(launch.context):
            continue
        if isinstance(e, DeclareLaunchArgument):
            if e.name not in launch.context.launch_configurations and e.default_value is not None:
                launch.context.launch_configurations[e.name] = perform_substitutions(launch.context, e.default_value)
        elif isinstance(e, OpaqueFunction):
            walk(e.execute(launch.context) or [], launch)
        elif isinstance(e, GroupAction):
            walk(e.get_sub_entities(), launch)
        elif isinstance(e, IncludeLaunchDescription):
            launch.includes.append(perform_substitutions(launch.context, e.launch_description_source._LaunchDescriptionSource__location))
        elif isinstance(e, RegisterEventHandler):
            if isinstance(e.event_handler, OnProcessExit):
                launch.exit_handlers.append(e.event_handler)
        elif isinstance(e, Node):
            launch.nodes.append(e)
        elif isinstance(e, ExecuteProcess):
            launch.processes.append(e)


def exit_event(action, returncode):
    return ProcessExited(action=action, name="process-1", cmd=["process"], cwd=None, env=None, pid=1, returncode=returncode)


def on_exit(launch, action, returncode):
    """What the launch starts when `action` exits with `returncode`: a Launch of its own (sharing the context)."""
    out = Launch(launch.context)
    for handler in list(launch.exit_handlers):
        event = exit_event(action, returncode)
        if handler.matches(event):
            walk(handler.handle(event, launch.context) or [], out)
    return out


def is_spawner(node):
    return node.node_package == "controller_manager" and node.node_executable == "spawner"


def at_startup(launch):
    """The description plus whatever the spawners' OnProcessExit handlers start when they exit 0 (a spawner exits 0 once its controller is
    loaded: that is how the controller chain starts), until nothing new appears."""
    done, frontier = set(), [n for n in launch.nodes if is_spawner(n)]
    while frontier:
        spawner = frontier.pop()
        if id(spawner) in done:
            continue
        done.add(id(spawner))
        more = on_exit(launch, spawner, 0)
        launch.nodes += more.nodes
        launch.processes += more.processes
        launch.includes += more.includes
        launch.exit_handlers += more.exit_handlers
        frontier += [n for n in more.nodes if is_spawner(n)]
    return launch


def describe(package, name, **arguments):
    """The launch description that SRC/<package>/launch/<name> builds, walked, at start-up."""
    path = os.path.join(SRC, package, "launch", name)
    spec = importlib.util.spec_from_file_location(f"under_test_{package}_{name.replace('.', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launch = Launch(LaunchContext())
    launch.context.launch_configurations.update(arguments)
    walk([module.generate_launch_description()], launch)
    return at_startup(launch)


def arguments(node):
    """A Node's own arguments, before --ros-args."""
    args = [text(a) for a in node.cmd[1:]]
    return args[:args.index("--ros-args")] if "--ros-args" in args else args


def parameters(node):
    """A Node's parameters given as dictionaries: {name: value}. launch_ros keeps a text value as YAML ('/joint_states\\n...\\n'), so it
    is read back as YAML, as the node will get it."""
    import yaml
    out = {}
    for entry in node._Node__parameters:
        if isinstance(entry, dict):
            for k, v in entry.items():
                out[text(k)] = yaml.safe_load(text(v)) if isinstance(v, tuple) and all(hasattr(s, "perform") for s in v) else v
    return out


def command(process):
    return [text(part) for part in process.cmd]


def deactivates_lift_controller(c):
    """Whether command `c` is `ros2 control switch_controllers ... --deactivate lift_controller` with no activation in it. Wrapped is fine
    (#53 bounds the fallback: `timeout --kill-after=2 15 ros2 control ...`); what must be there is the call and the deactivation."""
    for i in range(len(c) - 2):
        if c[i:i + 3] == ["ros2", "control", "switch_controllers"]:
            rest = c[i + 3:]
            return "--activate" not in rest and "--deactivate" in rest and rest[rest.index("--deactivate") + 1:][:1] == [LIFT_CONTROLLER]
    return False


def guards(launch):
    return [n for n in launch.nodes if n.node_package == CONTROL and n.node_executable == GUARD]


def anything_stub(launch):
    """Every Node or process that looks like an odometry stub, by executable or command."""
    return [n.node_executable for n in launch.nodes if "odom_stub" in n.node_executable] + \
           [" ".join(command(p)) for p in launch.processes if "odom_stub" in " ".join(command(p))]


# ---- the simulated launches ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim():
    return {profile: describe(*where) for profile, where in SIM_LAUNCHES.items()}


@pytest.mark.parametrize("profile", sorted(SIM_LAUNCHES))
def test_each_simulated_launch_starts_exactly_one_guard_configured_for_its_profile(sim, profile):
    found = guards(sim[profile])
    assert len(found) == 1, f"{profile}: expected one {CONTROL} {GUARD} Node in the launch description, found {len(found)}"
    p = parameters(found[0])
    assert p.get("arm_state_topic") == arm_state_topic(profile), f"{profile}: the guard reads {p.get('arm_state_topic')!r}"
    assert bool(p.get("use_sim_time", False)) is (profile == "gazebo"), f"{profile}: use_sim_time {p.get('use_sim_time')!r}"


@pytest.mark.parametrize("profile", sorted(SIM_LAUNCHES))
def test_the_guards_own_exit_deactivates_lift_controller(sim, profile):
    found = guards(sim[profile])
    assert len(found) == 1, f"{profile}: no single guard Node"
    handlers = [h for h in sim[profile].exit_handlers if h.matches(exit_event(found[0], -9))]
    assert len(handlers) == 1, f"{profile}: expected one OnProcessExit on the guard Node itself, found {len(handlers)}"
    started = on_exit(sim[profile], found[0], -9)
    commands = [command(p) for p in started.processes]
    assert [c for c in commands if deactivates_lift_controller(c)],         f"{profile}: the guard's exit starts no deactivation of {LIFT_CONTROLLER}: {commands}"


@pytest.mark.parametrize("profile", sorted(SIM_LAUNCHES))
def test_the_lift_controller_spawner_that_starts_gets_inactive_and_nothing_activates_it(sim, profile):
    lift = [n for n in sim[profile].nodes if is_spawner(n) and arguments(n)[:1] == [LIFT_CONTROLLER]]
    assert len(lift) == 1, f"{profile}: expected one {LIFT_CONTROLLER} spawner at start-up, found {[arguments(n) for n in lift]}"
    assert "--inactive" in arguments(lift[0]), f"{profile}: {LIFT_CONTROLLER} spawned with {arguments(lift[0])}: not --inactive"
    activations = [command(p) for p in sim[profile].processes if "--activate" in command(p) and LIFT_CONTROLLER in command(p)]
    assert not activations, f"{profile}: a start-up action activates {LIFT_CONTROLLER}: {activations}"


def test_the_odometry_stub_runs_in_mock_exactly_once(sim):
    stubs = [n for n in sim["mock"].nodes if n.node_package == CONTROL and n.node_executable == STUB]
    assert len(stubs) == 1, f"mock: expected one {STUB} Node, found {len(stubs)}"
    assert anything_stub(sim["mock"]) == [STUB], f"mock: other stub-like actions: {anything_stub(sim['mock'])}"


def test_nothing_like_an_odometry_stub_runs_in_gazebo(sim):
    assert not anything_stub(sim["gazebo"]), f"gazebo has the DiffDrive /odom, no stub: {anything_stub(sim['gazebo'])}"


# ---- everywhere else -------------------------------------------------------------------------------------------------------------------
def launch_files():
    for p in PACKAGES:
        d = os.path.join(SRC, p, "launch")
        if os.path.isdir(d):
            for n in sorted(os.listdir(d)):
                if n.endswith(".py"):
                    yield p, n


def other_launch_files():
    return [(p, n) for p, n in launch_files() if (p, n) not in SIM_LAUNCHES.values()]


@pytest.mark.parametrize("package, name", other_launch_files())
def test_no_other_launch_description_starts_the_guard_or_the_stub(package, name):
    for args in OTHER_LAUNCHES.get((package, name), [{}]):
        launch = describe(package, name, **args)
        found = [n.node_executable for n in launch.nodes if n.node_executable in (GUARD, STUB)] + anything_stub(launch)
        assert not found, f"{package}/launch/{name} {args or ''} starts {found} itself"
        # the stack the launch tests start (moveit.launch.py) gets the guard only by including a simulated launch
        sims = [i for i in launch.includes if os.path.basename(i) in ("mock.launch.py", "sim.launch.py")]
        if args.get("sim"):
            assert [os.path.basename(i) for i in sims] == [SIM_LAUNCHES[args["sim"]][1]], f"{name} {args}: includes {launch.includes}"


def wiring_in_source(path):
    """What the syntax tree of a launch file says about the guard, docstrings and comments aside: imports of #53's lift_guard_launch,
    and string constants that are exactly the guard's or the stub's executable."""
    tree = ast.parse(open(path).read())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("lift_guard_launch"):
            found.append(f"from {n.module} import ...")
        elif isinstance(n, ast.Import) and any(a.name.endswith("lift_guard_launch") for a in n.names):
            found.append("import ...lift_guard_launch")
        elif isinstance(n, ast.Constant) and n.value in (GUARD, STUB) and id(n) not in docstrings:
            found.append(repr(n.value))
    return found


@pytest.mark.parametrize("package, name", other_launch_files())
def test_no_other_launch_file_wires_the_guard_in_any_branch(package, name):
    found = wiring_in_source(os.path.join(SRC, package, "launch", name))
    assert not found, f"{package}/launch/{name} wires the guard or the stub (#44: mock and gazebo only): {found}"


def test_the_real_profile_description_does_not_mention_the_guard():
    """A text check of absence only: the guard is a ROS node, launched; the launch-level checks above are what keep it out of profiles."""
    urdf = os.path.join(SRC, "openamrobot_upperbody_description", "urdf")
    hits = [n for n in sorted(os.listdir(urdf)) if n.endswith(".xacro") and "lift_guard" in open(os.path.join(urdf, n)).read()]
    assert not hits, f"the description (shared by the real profile) mentions the guard: {hits}"


# ---- the walker's own behaviour (synthetic launch descriptions) --------------------------------------------------------------------------
def walked(*entities, **arguments):
    launch = Launch(LaunchContext())
    launch.context.launch_configurations.update(arguments)
    walk([LaunchDescription(list(entities))], launch)
    return at_startup(launch)


def test_walker_skips_what_a_false_condition_leaves_out_and_keeps_it_when_the_argument_says_so():
    from launch.conditions import IfCondition
    from launch.substitutions import LaunchConfiguration
    entities = [DeclareLaunchArgument("on", default_value="false"),
                Node(package="p", executable="x", condition=IfCondition(LaunchConfiguration("on")))]
    assert walked(*entities).nodes == []
    assert [n.node_executable for n in walked(*entities, on="true").nodes] == ["x"]


def test_walker_runs_opaque_functions_with_the_launch_arguments_and_records_includes():
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    def setup(context):
        which = context.launch_configurations["sim"]
        return [Node(package="p", executable=f"in_{which}"), IncludeLaunchDescription(PythonLaunchDescriptionSource(f"/x/{which}.launch.py"))]
    launch = walked(DeclareLaunchArgument("sim", default_value="mock"), OpaqueFunction(function=setup), sim="gazebo")
    assert [n.node_executable for n in launch.nodes] == ["in_gazebo"] and launch.includes == ["/x/gazebo.launch.py"]


def test_walker_starts_what_a_spawner_exit_starts_in_list_and_callable_form_but_not_other_exits():
    first = Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"])
    other = Node(package="p", executable="guard")
    chain = RegisterEventHandler(OnProcessExit(target_action=first, on_exit=[
        Node(package="controller_manager", executable="spawner", arguments=["lift_controller", "--inactive"])]))
    on_guard = RegisterEventHandler(OnProcessExit(target_action=other, on_exit=[ExecuteProcess(cmd=["echo", "guard", "exited"])]))
    launch = walked(first, other, chain, on_guard)
    assert sorted(arguments(n)[0] for n in launch.nodes if is_spawner(n)) == ["joint_state_broadcaster", "lift_controller"]
    assert launch.processes == [], "a non-spawner's exit handler ran at start-up"
    assert [command(p) for p in on_exit(launch, other, -9).processes] == [["echo", "guard", "exited"]]
    gated = RegisterEventHandler(OnProcessExit(target_action=first, on_exit=lambda event, context: (
        [Node(package="controller_manager", executable="spawner", arguments=["arm"])] if event.returncode == 0 else [])))
    assert sorted(arguments(n)[0] for n in walked(first, gated).nodes if is_spawner(n)) == ["arm", "joint_state_broadcaster"]


def test_node_arguments_and_parameters_are_read_as_given():
    n = Node(package="p", executable="x", arguments=["lift_controller", "--inactive"],
             parameters=[{"arm_state_topic": "/joint_states", "use_sim_time": True, "tolerance": 0.01}])
    assert arguments(n) == ["lift_controller", "--inactive"]
    assert parameters(n) == {"arm_state_topic": "/joint_states", "use_sim_time": True, "tolerance": 0.01}


def test_the_syntax_check_ignores_docstrings_and_names_but_sees_imports_and_executables(tmp_path):
    clean = tmp_path / "clean.launch.py"
    clean.write_text(f'"""Waits for {GUARD} and {STUB}."""\nNode(executable="wait_for_lift_guard.py")\n')
    assert wiring_in_source(str(clean)) == []
    wired = tmp_path / "wired.launch.py"
    wired.write_text(f'from {CONTROL}.lift_guard_launch import lift_guard_actions as nodes\nNode(executable="{STUB}")\n')
    assert len(wiring_in_source(str(wired))) == 2


@pytest.mark.parametrize("c, expected", [
    (["ros2", "control", "switch_controllers", "--deactivate", "lift_controller", "--best-effort"], True),
    (["timeout", "--kill-after=2", "15", "ros2", "control", "switch_controllers", "--deactivate", "lift_controller", "--best-effort"], True),
    (["ros2", "control", "switch_controllers", "--activate", "lift_controller"], False),
    (["ros2", "control", "switch_controllers", "--deactivate", "lift_controller", "--activate", "lift_controller"], False),
    (["ros2", "control", "switch_controllers", "--deactivate", "left_arm_controller"], False),
    (["ros2", "control", "switch_controllers", "--deactivate"], False),
    (["echo", "--deactivate", "lift_controller"], False),
    (["ros2", "control", "list_controllers", "--deactivate", "lift_controller"], False),
])
def test_the_exit_command_check_accepts_a_wrapped_deactivation_and_nothing_else(c, expected):
    assert deactivates_lift_controller(c) is expected
