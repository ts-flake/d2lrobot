## 1. Intro

This repo is based on the XLeRobot [[Link](https://github.com/Vector-Wangel/XLeRobot/tree/main)]. We provide the codes to teleoperate and record dataset using the bimanual SO101-yaw (6DoF) platform [[Link](https://makerworld.com/en/models/1913316-so101-arm-wrist-yaw-6dof#profileId-2088553)].

## 2. Installation

> [!NOTE]
> 
>  Before you start:
> 
> - Have the physical platform (XLeRobot) ready
> 
> - Install the `lerobot`

### 2.1 Move files

The installation is minimal, you only need to copy/move some files. The folder structre is the same as `lerobot`, for example:

```
software/
    - examples
        - xlerobot_yaw  --> move to lerobot/examples
    - src/
        - model/rr_kinematics.py  --> move to lerobot/src/lerobot/model
        - robots/
            - xlerobot_yaw/  --> move to lerobot/src/lerobot/robots
        - ...
```

### 2.2 Modifiy codes

You still need to update a few lines of codes.

- In `.../teleoperators/xlerobot_yaw_vr/vr_monitor.py` around line 21, set the variable `XLEVR_PATH`.

- In `lerobot/src/lerobot/robots/utils.py` around line 65, add
  
  ```python
  elif config.type == "xlerobot_yaw":
      from .xlerobot_yaw import XLeRobotYaw
      return XLeRobotYaw(config)
  ```

- In `lerobot/src/lerobot/teleoperators/utils.py` around line 80, add
  
  ```python
  elif config.type == "xlerobot_yaw_vr":
      from .xlerobot_yaw_vr import XLeRobotYawVR
      return XLeRobotYawVR(config)
  ```

## 3. Usages

To teleoperate XLeRobot using a VR (Quest3), go to `lerobot/examples/xlerobot_yaw` and run:

```bash
python teleoperate_vr.py \
--robot.id xlerobot_yaw01 \
--teleop.fps 30 \
--display_data true
```

The full list of parameters can be found in...

- `.../robots/xlerobot_yaw/config_xlerobot_yaw.py`

- `.../teleoperators/xlerobot_yaw_vr/config_xlerobot_yaw_vr.py`

## 4. Changes

### 4.1 IK

The `RRKinematics` class in `software/src/model/rr_kinematics.py` is essentially a refactoring of the `SO101Kinematics` from the XLeRobot, with a detailed comment on the angle definitions and better namings.

### 4.2 XLeVR

The `wrist_yaw_deg` is a newly added field to `ControlGoal` for the SO101-yaw (or any 6DoF) robot arm.

Significant changes to `XLeVR/xlevr/inputs/vr_ws_server.py`:

- `VRControllerState` now records the prev/curr position/quaternion to faciliate delta position/quaternion computation

- `VRWebSocketServer` sends delta EE commands, in the **robot's frame**, i.e., forward (x), left (y), upward (z):
  
  - `target_position`: (dx, dy, dz)
  
  - `wrist_roll_deg`: x-axis rotation (angle directions follow the right hand rule)
  
  - `wrist_flex_deg`: pitch, y-axis rotation
  
  - `wrist_yaw_deg`: z-axis rotation
  
  Note that in the codes, the delta actions are expressed in the local/body frame (`origin_quaternion`) first and then converted to the robot's frame.

- The original squeeze-to-teleoperate logic is pushed to downstreams, all necessary information is stored in `metadata`. E.g., the user can decide the behavior of the controller when `metadata['buttons']['squeeze']` is true.
  
  ```python
  metadata = {
      ...,
      'trigger': float (0-1)
      'trigger_active': bool if trigger > 0.5
      'thumbstick': {'x': float (0-1), 'y': float (0-1)},
      'buttons': {
          'squeeze': bool,
           'x': bool,
           'y': bool,
           'a': bool,
           'b': bool
      }
  }
  ```

### 4.3 Controller Keymap: SO101_yaw

Below are the default keymaps used for SO101_yaw.

**Left controller**

- **thumbstick**: not configured
- **trigger**: open / close left gripper (toggled by pressing button A)
- **grip (squeeze) button**: freeze / unfreeze left arm (set by `grip_to_activate` in `XLeRobotYawVRConfig`)
- **X button**: quit and move back to initial position
- **Y button**: move back to zero position



**Right controller**

- **thumbstick**: base motion control, forward / backward / turn left / turn right

- **trigger**: open / close gripper

- **grip (squeeze) button**: freeze / unfreeze left arm (set by `grip_to_activate` in `XLeRobotYawVRConfig`)

- **A button**: toggle trigger state, i.e., open / close gripper

- **B button**: reset the origin pose of both controllers, press this button whenever you relocate / reorientate in space
