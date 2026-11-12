# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.8)
project(openamrobot_upperbody_bringup)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch config DESTINATION share/openamrobot_upperbody_bringup)
install(PROGRAMS
  scripts/base_joint_states_stub.py
  scripts/build_a7_evidence.py
  scripts/run_a7_regression.sh
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  # H01: every test file is registered, and every launch test checks that its processes started and exited cleanly (static, over all five packages).
  ament_add_pytest_test(h01_harness_guards test/test_h01_harness_guards.py TIMEOUT 120)
  # the two numbers M02 gates the clamp on, and that counts gate nothing (P17); no ROS graph
  ament_add_pytest_test(a7_clamp_gate test/test_a7_clamp_gate.py TIMEOUT 120)
  # W04 (a force-stopped gz server is a WARN) from the launch tests' sidecars (P18)
  ament_add_pytest_test(a7_w04 test/test_a7_w04.py TIMEOUT 120)
  ament_add_pytest_test(a7_process_exits test/test_a7_process_exits.py TIMEOUT 120)
  # M02's wait for the controllers to be active, and its FAIL when they never are, against a fake controller manager (P17)
  ament_add_pytest_test(a7_controller_wait test/test_a7_controller_wait.py TIMEOUT 180)
  # the controller wait honours one absolute deadline (PR #14 review); fake clock and client, no ROS graph
  ament_add_pytest_test(a7_controller_deadline test/test_a7_controller_deadline.py TIMEOUT 60)
  # M07 fails on incomplete joint-state coverage (PR #14 review); the model and synthetic samples, no ROS graph
  ament_add_pytest_test(m07_coverage test/test_m07_coverage.py TIMEOUT 60)
  # M07 rejects non-finite positions before it counts or takes an extremum, and FAILs on them (PR #14 review, P27); synthetic samples and the model, no ROS graph
  ament_add_pytest_test(m07_non_finite test/test_m07_non_finite.py TIMEOUT 60)
  # L01 starts a real controller_manager: it picks its own ROS_DOMAIN_ID (see the test), so it cannot see or be seen by
  # another stack on the machine.
  add_launch_test(test/test_l01_mock_controllers.py TIMEOUT 60)
  # M01 (A7): every named pose plans and executes on mock through move_group, once per lift calibration stop (option B's L1000, L1175,
  # L1350: positions of lift_joint, not configuration IDs). Each run starts the whole mock stack in its own ROS_DOMAIN_ID (see the test)
  # and records its own sidecar. Since issue #17 each run also PLANS the lift to its stop and back (at most 0.35 m each way at half the
  # DOLD no-load speed, about 40 s of motion each way), so the budget is 480 s, not 240.
  foreach(stop L1000 L1175 L1350)
    add_launch_test(test/test_m01_mock_named_poses.py TARGET m01_mock_${stop} ARGS "lift_stop:=${stop}" TIMEOUT 480)
  endforeach()
  # M02: the same in Gazebo at the baseline configuration (L1000), the lift planned to L1175 and back first (issue #17). It starts a headless
  # Gazebo server and picks its own ROS_DOMAIN_ID and GZ_PARTITION, like G01 and G02; startup alone is about a minute, and the two lift moves
  # are about 20 s of simulated time each, so the budget is 900 s (600 before the lift was planned).
  add_launch_test(test/test_m02_gazebo_named_poses.py TARGET m02_gazebo_L1000 TIMEOUT 900)
  # M07 (P17): at rest in Gazebo no joint sits within the joint_rest_margin_rad datum of a limit. Starts its own headless server like M02,
  # on its own domain and partition; FAILs by design while the right finger rests on its open limit.
  add_launch_test(test/test_m07_rest_position_margin.py TARGET m07_rest_position_margin TIMEOUT 300)
  # --- #44 sim lift guard acceptance (black box: /lift_guard/status, the controller manager, lift_joint, the guard's own parameters) ---
  # The oracle that checks the run-wide safety properties, on synthetic timelines (no ROS graph): it must find every fault it exists for.
  ament_add_pytest_test(lift_guard_oracle test/test_lift_guard_oracle.py TIMEOUT 60)
  ament_add_pytest_test(lift_guard_parameters test/test_lift_guard_parameters.py TIMEOUT 60)
  ament_add_pytest_test(lift_guard_processes test/test_lift_guard_processes.py TIMEOUT 60)
  # Where the guard is wired: both simulated launches, lift_controller spawned --inactive, nothing in the real profile (source tree only).
  ament_add_pytest_test(lift_guard_static test/test_lift_guard_static.py TIMEOUT 60)
  # L02: arms, controller and the guard's contract on mock. L04: base and odometry on mock. Each in its own ROS_DOMAIN_ID, like L01.
  # On main before the guard they FAIL by design (the red half of the #44 integration branch). LIFT_GUARD_STACK_LAUNCH overrides the stack.
  add_launch_test(test/test_l02_lift_guard_mock.py TARGET l02_lift_guard_arms_mock TIMEOUT 900)
  add_launch_test(test/test_l04_lift_guard_base_mock.py TARGET l04_lift_guard_base_mock TIMEOUT 600)
  # L03 (#44 acceptance 1): the guard process dies; SIGKILL while the lift moves (the acceptance case), SIGTERM while it moves, SIGKILL idle.
  add_launch_test(test/test_l03_lift_guard_kill_mock.py TARGET l03_lift_guard_kill_moving ARGS "kill_signal:=KILL" "kill_phase:=moving" TIMEOUT 300)
  add_launch_test(test/test_l03_lift_guard_kill_mock.py TARGET l03_lift_guard_term_moving ARGS "kill_signal:=TERM" "kill_phase:=moving" TIMEOUT 300)
  add_launch_test(test/test_l03_lift_guard_kill_mock.py TARGET l03_lift_guard_kill_idle ARGS "kill_signal:=KILL" "kill_phase:=idle" TIMEOUT 300)
  # L05: moveit.launch.py lift_position:=X sends a start-up goal; it must not move the lift before the guard allows it.
  add_launch_test(test/test_l05_lift_guard_startup_goal_mock.py TARGET l05_lift_guard_startup_goal_mock TIMEOUT 300)
  # G03: the guard in Gazebo (bullet-featherstone, like M02): raw /arm/joint_states vs the merge, a driven base, the lift read from the
  # simulator's own carriage pose. Own ROS_DOMAIN_ID and GZ_PARTITION.
  add_launch_test(test/test_g03_lift_guard_gazebo.py TARGET g03_lift_guard_gazebo TIMEOUT 1200)
endif()

ament_package()
