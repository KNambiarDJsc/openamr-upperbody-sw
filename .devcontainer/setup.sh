#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Dev container onCreateCommand: lays out ~/openamr_ws as README.md (Prerequisites) and CI do, installs the rosdep
# dependencies of the packages the model needs and builds them. Run from the repository root.
set -euo pipefail

repo="$(pwd)"
ws="${OPENAMR_WS:-$HOME/openamr_ws}"
pins="$repo/ros2/src/openamrobot_upperbody_description/config/upstream_pins.yaml"
packages="openamrobot_upperbody_description openamrobot_upperbody_control openamrobot_upperbody_gazebo openamrobot_upperbody_moveit openamrobot_upperbody_bringup"

pin() { python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["pins"][sys.argv[2]])' "$pins" "$1"; }

# Clone $1 into $2 at commit $3 (only that commit's history), unless $2 is already there.
checkout() {
  if [ ! -d "$2/.git" ]; then
    git init -q "$2"
    git -C "$2" remote add origin "$1"
  fi
  git -C "$2" fetch -q --depth 1 origin "$3"
  git -C "$2" -c advice.detachedHead=false checkout -q "$3"
}

mkdir -p "$ws/src/upstream"
ln -sfn "$repo" "$ws/src/openamr-upperbody-sw"
checkout https://github.com/openAMRobot/openamr-platform-sw.git "$ws/src/openamr-platform-sw" "$(pin openamr_platform_sw)"
checkout https://github.com/enactic/openarm_description.git "$ws/src/upstream/openarm_description" "$(pin openarm_description)"
checkout https://github.com/enactic/openarm_ros2.git "$ws/src/upstream/openarm_ros2" "$(pin openarm_ros2)"

set +u
source /opt/ros/jazzy/setup.bash
set -u
cd "$ws"
sudo apt-get update
rosdep update --rosdistro jazzy
# shellcheck disable=SC2086
rosdep install --rosdistro jazzy --ignore-src -y --from-paths $(colcon list --base-paths src --paths-only --packages-up-to $packages)
# shellcheck disable=SC2086
colcon build --symlink-install --packages-up-to $packages

line="source $ws/install/setup.bash"
grep -qxF "$line" "$HOME/.bashrc" || printf 'source /opt/ros/jazzy/setup.bash\n%s\n' "$line" >> "$HOME/.bashrc"
echo "Workspace built: $ws. In a new terminal: ros2 launch openamrobot_upperbody_bringup moveit.launch.py sim:=mock"
