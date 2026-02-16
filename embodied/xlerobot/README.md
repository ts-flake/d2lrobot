## 1. Introduction

This repo is based on the XLeRobot [[Link](https://github.com/Vector-Wangel/XLeRobot/tree/main)]. Below are some key changes...

- LeRobot-styled teleoperation, i.e., using a **Teleoperator** and **Robot Processors**, thus this increases code reusability
- Better code readability with detailed comments and clear parameter names
- Easier ways to set configurations via `TeleoperatorConfig` or CLI
- **Improved VR control experience** with added keymaps for more actions such as 'back to zero/home', 'exit', and 'reset reference frame' (I found this is very useful when you drive the robot around and still be able to reposition yourself)
- **Smoothier base movements**. Using the `EMAJointAction` processor, the three-wheeled base now starts off gently and stops pointly
- (New) **Easier and unified keymap definitions** for different gamepads, see details in `.../teleoperators/xlerobot_gamepad/gamepad_utils.py`
- (Optional) Support SO101-yaw arm (SO101 + 1 dof wrist yaw) [[Hardware](https://makerworld.com/en/models/1913316-so101-arm-wrist-yaw-6dof#profileId-2088553)]

## 2. Installation

> [!IMPORTANT]
> Before you start:
> 
> - Have the physical platform (XLeRobot) ready
> - If you want to try XLeRobot-yaw, ensure you have reconfigured motor ids (using tools from [[Bambot](https://bambot.org/feetech.js?lang=en)])...
>   - left arm (7 dof) + head (2dof): id 1 to 9
>   - right arm (7 dof) + base (3 dof): id 1 to 10
> - Install the `lerobot`

### 2.1 Move files

The installation is minimal, you only need to copy/move some files. The folder structure is the same as `lerobot`, specifically:

```
software/
├── examples/               --> move to lerobot/examples/
│   ├── xlerobot/
│   └── xlerobot_yaw/
└── src/
    ├── model/              --> move to lerobot/src/lerobot/model
    │   └── rr_kinematics.py
    ├── robots/             --> move to lerobot/src/lerobot/robots
    │   ├── xlerobot/
    │   └── xlerobot_yaw/
    └── teleoperators/      --> move to lerobot/src/lerobot/teleoperators
        ├── xlerobot_vr/
        ├── xlerobot_yaw_vr/
        └── ...
```

### 2.2 Modifiy codes

You still need to update a few lines of codes.

- In `.../teleoperators/xlerobot_vr/vr_monitor.py` around line 21, set the variable `XLEVR_PATH`.

- In `lerobot/src/lerobot/robots/utils.py` around line 65, add
  
  ```python
  elif config.type == "xlerobot":
      from .xlerobot import XLeRobot
      return XLeRobot(config)
  elif config.type == "xlerobot_yaw":
      from .xlerobot_yaw import XLeRobotYaw
      return XLeRobotYaw(config)
  ```

- In `lerobot/src/lerobot/teleoperators/utils.py` around line 80, add
  
  ```python
  elif config.type == "xlerobot_vr":
      from .xlerobot_vr import XLeRobotVR
      return XLeRobotVR(config)
  elif config.type == "xlerobot_yaw_vr":
      from .xlerobot_yaw_vr import XLeRobotYawVR
      return XLeRobotYawVR(config)
  elif config.type == "xlerobot_gamepad":
      from .xlerobot_gamepad import XLeRobotGamepad
      return XLeRobotGamepad(config)
  elif config.type == "xlerobot_yaw_gamepad":
      from .xlerobot_yaw_gamepad import XLeRobotYawGamepad
      return XLeRobotYawGamepad(config)
  ```

## 3. Usages

To teleoperate XLeRobot using a VR (Quest3), go to `lerobot/examples/xlerobot` and run:

```bash
python teleoperate_vr.py \
--robot.id [your_robot_name] \
--teleop.fps 30 \
--teleop.grip_to_activate true \
--display_data true
```

To record dataset, go to `lerobot/examples/xlerobot` and run:

```bash
python record_vr.py \
--robot.id [your_robot_name] \
--teleop.fps 30 \
--teleop.grip_to_activate true \
--teleop.record_dataset true \
--dataset.repo_id [user/folder] \
--dataset.single_task [task_name] \
--dataset.num_episodes 50 \
--dataset.push_to_hub true \
--display_data true
```

Other mode of teleoperation, e.g., gamepad, can be done in the same way.



The full list of parameters can be found in (for examples)...

- `.../robots/xlerobot/config_xlerobot.py`
- `.../teleoperators/xlerobot_vr/config_xlerobot_vr.py`
- `lerobot/src/lerobot/scripts/lerobot_record.py:line142`

## 4. Changes

### 4.1 Inverse Kinematics

The `RRKinematics` class in `software/src/model/rr_kinematics.py` is essentially a refactoring of the `SO101Kinematics` from the XLeRobot, with a detailed comment on the angle definitions and better naming.

### 4.2 XLeVR

The `wrist_yaw_deg` is a newly added field to `ControlGoal` for the SO101-yaw (or any 6DoF) robot arm. For the SO101 arm, this value is ignored.

Significant changes to `XLeVR/xlevr/inputs/vr_ws_server.py`:

- `VRControllerState` now records the prev/curr-position/quaternion to facilitate delta position/quaternion computation

- `VRWebSocketServer` sends **delta EE** commands, in the **robot's frame**, i.e., forward (x), left (y), upward (z):
  
  - `target_position`: (dx, dy, dz)
  - `wrist_roll_deg`: x-axis rotation (angle directions follow the right-hand rule)
  - `wrist_flex_deg`: pitch, y-axis rotation
  - `wrist_yaw_deg`: z-axis rotation

- The **original squeeze-to-teleoperate logic is pushed to downstreams**, and all necessary information is stored in `metadata`. E.g., the user can decide the behavior of the controller when `metadata['buttons']['squeeze']` is true.
  
  > ***Note:*** In the codes, the delta actions are expressed in the local/body frame (`origin_quaternion`) first and then converted to the robot's frame.
  
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

### 4.3 Controller Keymaps: In Teleoperator

Below are the default keymaps for the Quest3 VR controller.

**Left controller**

- **thumbstick**: Reserved for recording actions, e.g., 'early exiting', 'stop recording', etc.
- **trigger**: Open/close left gripper (toggled by pressing button A)
- **grip (squeeze) button**: Freeze/unfreeze left arm (set by `grip_to_activate` in `XLeRobotYawVRConfig`)
- **X button**: Quit and move back to the initial position
- **Y button**: Move back to the zero position

**Right controller**

- **thumbstick**: Base motion control, e.g., 'forward', 'backward', 'turn left/right'
- **trigger**: Open/close gripper (toggled by pressing button A)
- **grip (squeeze) button**: Freeze/unfreeze right arm (set by `grip_to_activate` in `XLeRobotYawVRConfig`)
- **A button**: Toggle trigger state, i.e., open/close gripper
- **B button**: Reset the origin pose of both controllers. *Press this button whenever you relocate/reorientate in space*
