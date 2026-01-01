## Intro

This folder mirrors the original XLeRobot project [[Link](https://github.com/Vector-Wangel/XLeRobot/tree/main)]. Some modifications are made for the SO101-yaw (6DoF arm) [[Link](https://makerworld.com/en/models/1913316-so101-arm-wrist-yaw-6dof#profileId-2088553)].

## Installation

Follow the [[steps](https://xlerobot.readthedocs.io/en/latest/software/index.html)] to install XLeRobot, that is you need to install `lerobot` first, and then move files to corresponding folders.

Check the README in the XLeVR folder to setup the VR control.

## Usages

Run the scripts in `software/examples` to start teleoperating via a PS5 console or VR (tested with Quest3).

More examples are in progress...

## Changes

### IK

The `RRKinematics` class in `software/src/model/rr_kinematics.py` is essentially a refactoring of the `SO101Kinematics` from the XLeRobot, with a detailed comment on the angle definitions and better namings.

### XLeVR

The `wrist_yaw_deg` is a newly added field to control the SO101-yaw (or any 6DoF) robot arm. The current code (`xlevr/inputs/vr_ws_server.py`) uses the absolute `target_position`. And the pose (xyz + wrist rotation angles) is expressed in the local frame (i.e., `origin_quaternion`), since this format is more convinent when the user relocate in space (see the controller keymap below).

### Controller Keymap: SO101_yaw

Below are the default keymaps used in `xlerobot_yaw_teleop_vr.py`.

**Left controller**

- thumbstick: not configured
- trigger: open / close left gripper
- grip (squeeze) button: freeze left arm
- X button: quit and move back to initial position
- Y button: move back to zero position

**Right controller**

- thumbstick: base motion control, forward / backward / turn left / turn right

- trigger: open / close gripper

- grip (squeeze) button: freeze right arm

- A button: toggle trigger state, i.e., open / close gripper

- B button: reset the origin pose of both controllers, press this button whenever you relocate / reorientate in space
