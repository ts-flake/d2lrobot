#!/usr/bin/env python3
"""
VR control for XLerobot robot
Uses handle_vr_input with delta action control
"""

# Standard library imports
import time
import copy
import asyncio
import logging
import threading
import traceback
from typing import Any
from pprint import pformat
from dataclasses import dataclass, field

# Third-party imports
import draccus
import numpy as np

# Local imports
from utils.vr_monitor import VRMonitor
from utils.utils import init_logging
from utils.teleop_utils import SimpleControl, SimpleBaseControl
from lerobot.robots.xlerobot_yaw import XLerobotYawConfig, XLerobotYaw
from lerobot.model.rr_kinematics import RRKinematics

# --------------- 电机关节映射 ---------------
LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_yaw": "left_arm_wrist_yaw",  # New joint
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_yaw": "right_arm_wrist_yaw",  # New joint
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_JOINT_MAP = {
    "head_yaw": "head_yaw",
    "head_pitch": "head_pitch",
}

# ------------- Controllers -------------
class SimpleHeadControl(SimpleControl):
    def __init__(
        self,
        robot: XLerobotYaw,
        name: str = "Head",
        bus_name: str = 'bus1',
        stepsize: float | dict[str, float] = 1.0,
        control_freq: int = 50,
        motor_map: dict[str, str] = HEAD_JOINT_MAP,
        logger: logging.Logger | None = None,
        use_thumbstick: bool = False
    ):
        super().__init__(name, robot, bus_name, stepsize, control_freq, motor_map, logger)
        self._motors = self._bus.motors
        self._use_thumbstick = use_thumbstick

        self._prev_vr_goal = None
    
    def _set_target_from_thumbstick(self, thumb: dict) -> None:
        thumb_x = thumb.get('x', 0)
        thumb_y = thumb.get('y', 0)

        stepsize_joint = self.stepsize.get('joint', 1.0)
        _dead_zone = 0.5
        
        # Head yaw
        if abs(thumb_x) > _dead_zone:
            if thumb_x > 0:
                _motor = self.motor_map['head_yaw']
                _value = self.target_pos[_motor] + stepsize_joint
                self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
            else:
                _motor = self.motor_map['head_yaw']
                _value = self.target_pos[_motor] - stepsize_joint
                self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        # Head pitch
        if abs(thumb_y) > _dead_zone:
            if thumb_y > 0:
                _motor = self.motor_map['head_pitch']
                _value = self.target_pos[_motor] + stepsize_joint
                self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
            else:
                _motor = self.motor_map['head_pitch']
                _value = self.target_pos[_motor] - stepsize_joint
                self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)

    def set_target_from_states(self, states: dict[str, Any]) -> None:
        """
        Handle VR input with delta action control - incremental position updates.
        """
        vr_goal = states.get('vr_goal', None)
        if vr_goal is None:
            self.logger.warning(f"[{self.name}] VR goal is None")
            return

        # Establish pose baseline for delta pose calculation
        if self._prev_vr_goal is None:
            if any(getattr(vr_goal, attr) is None for attr in ['target_position', 'wrist_yaw_deg', 'wrist_flex_deg']):
                self.logger.warning(f"[{self.name}] VR goal baseline is not set yet!")
                return

            self._prev_vr_goal = copy.deepcopy(vr_goal)
            self.logger.info(f"[{self.name}] VR goal baseline set to: {pformat(self._prev_vr_goal)}")
            return # Skip first frame to establish baseline
        
        # Cache the previous target_pos
        self.update_prev_target_pos()

        # Use thumbstick as input
        if self._use_thumbstick:
            thumb = vr_goal.metadata.get('thumbstick', {})
            if not thumb:
                return
            else:
                self._set_target_from_thumbstick(thumb)
        # Otherwise, use the VR headset's delta rotation
        else:
            _dead_zone = 0.8

            _weights = {
                'yaw': 1,
                'pitch': 1,
            }
            ang_scale = 4.0
            ang_limit = 1.0
            
            # Head yaw
            curr_vr_yaw = vr_goal.wrist_yaw_deg
            prev_vr_yaw = self._prev_vr_goal.wrist_yaw_deg
            if curr_vr_yaw is not None and prev_vr_yaw is not None:
                dyaw = curr_vr_yaw - prev_vr_yaw

                if abs(dyaw) > _dead_zone:
                    dyaw = dyaw * _weights['yaw']
                    dyaw = np.clip(dyaw * ang_scale, -ang_limit, ang_limit)
                    _motor = self.motor_map['head_yaw']
                    _value = self.target_pos[_motor] + dyaw
                    self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
            
            # Head pitch
            curr_vr_pitch = vr_goal.wrist_flex_deg
            prev_vr_pitch = self._prev_vr_goal.wrist_flex_deg
            if curr_vr_pitch is not None and prev_vr_pitch is not None:
                dpitch = curr_vr_pitch - prev_vr_pitch

                if abs(dpitch) > _dead_zone:
                    dpitch = dpitch * _weights['pitch']
                    dpitch = np.clip(dpitch * ang_scale, -ang_limit, ang_limit)
                    _motor = self.motor_map['head_pitch']
                    _value = self.target_pos[_motor] + dpitch
                    self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)

        # Update previous VR goal
        self._prev_vr_goal = copy.deepcopy(vr_goal)
        # Print target if any
        self.log_target_pos()


class SimpleArmControl(SimpleControl):
    def __init__(
        self,
        robot: XLerobotYaw,
        kinematics: RRKinematics,
        name: str = "arm",
        bus_name: str = 'bus1',
        stepsize: float | dict[str, float] = {'joint': 1.0, 'xy': 0.001},
        control_freq: int = 50,
        motor_map: dict[str, str] = {},
        logger: logging.Logger | None = None
    ):
        super().__init__(name, robot, bus_name, stepsize, control_freq, motor_map, logger)
        self._motors = self._bus.motors
        self.kinematics = kinematics

        self._target_xy = None
        self._target_pitch = None

        self._prev_vr_goal = None # ControlGoal object

    def move_to_initial_position(self, duration: float = 5.0) -> None:
        self._target_xy = None
        self._target_pitch = None
        super().move_to_initial_position(duration)
    
    def move_to_zero_position(self, duration: float = 5.0, offset: dict[str, float] = {}) -> None:
        self._target_xy = None
        self._target_pitch = None
        super().move_to_zero_position(duration, offset)

    def set_target_from_states(self, states: dict[str, Any]) -> None:
        """
        Handle VR input with delta action control - incremental position updates.
        """
        vr_goal = states.get('vr_goal', None) # ControlGoal object
        # TODO: gripper_state = states.get('gripper_state', None)

        if vr_goal is None:
            self.logger.warning(f"[{self.name}] VR goal is None")
            return
        
        # VR goal contains:
        # - target_position [x, y, z] in VR coordinate, [right, up, back (to the user)]
        # - wrist_roll_deg
        # - wrist_flex_deg
        # - wrist_yaw_deg
        # - gripper_closed
        
        # Establish pose baseline for delta pose calculation
        if self._prev_vr_goal is None:
            if any(getattr(vr_goal, attr) is None for attr in ['target_position', 'wrist_roll_deg', 'wrist_flex_deg', 'wrist_yaw_deg']):
                self.logger.warning(f"[{self.name}] VR goal baseline is not set yet!")
                return

            self._prev_vr_goal = copy.deepcopy(vr_goal)
            self.logger.info(f"[{self.name}] VR goal baseline set to: {pformat(self._prev_vr_goal)}")
            return # Skip first frame to establish baseline
        
        # Cache the previous target_pos
        self.update_prev_target_pos()

        # Calculate delta pose from previous frame
        # Adjust these parameters for control sensitivity
        # --------------------------
        _weights = {
            'roll': 500,
            'pitch': 1000,
            'yaw': 1000,
            'dx': 2000,
            'dy': 70,
            'dz': 70
        }
        _dead_zones = {
            'roll': 0.7,
            'pitch': 0.3,
            'yaw': 0.3,
            'dx': 0.001,
            'dy': 0.001,
            'dz': 0.001
        }
        pos_scale = 0.01
        ang_scale = 0.1
        pos_limit = 0.005
        ang_limit = 1.0
        # --------------------------
        if self._target_pitch is None:
            self._target_pitch = 0.
        
        if self._target_xy is None:
            names_values = self.get_motor_values_deg([
                self.motor_map['shoulder_lift'],
                self.motor_map['elbow_flex']
            ])
            jnt2 = names_values[self.motor_map['shoulder_lift']]
            jnt3 = names_values[self.motor_map['elbow_flex']]
            self._target_xy = self.kinematics.forward_kinematics(jnt2, jnt3)
        
        # Freeze the arm (no target update) when the squeeze (grip) button is pressed
        if not vr_goal.metadata.get('buttons', {}).get('squeeze', False):
            # Pitch
            curr_vr_wrist_flex = vr_goal.wrist_flex_deg
            prev_vr_wrist_flex = self._prev_vr_goal.wrist_flex_deg
            if curr_vr_wrist_flex is not None and prev_vr_wrist_flex is not None:
                dpitch = curr_vr_wrist_flex - prev_vr_wrist_flex

                if abs(dpitch) > _dead_zones['pitch']:
                    dpitch = dpitch * _weights['pitch']
                    dpitch = np.clip(dpitch * ang_scale, -ang_limit, ang_limit)

                    self._target_pitch = np.clip(self._target_pitch - dpitch, -90, 90)  # 注意: wrist_flex_deg 上转为负数, 而 pitch 上转为正
            
            # Wrist roll (direct control)
            curr_vr_wrist_roll = vr_goal.wrist_roll_deg
            prev_vr_wrist_roll = self._prev_vr_goal.wrist_roll_deg
            if curr_vr_wrist_roll is not None and prev_vr_wrist_roll is not None:
                droll = curr_vr_wrist_roll - prev_vr_wrist_roll

                if abs(droll) > _dead_zones['roll']:
                    droll = droll * _weights['roll']
                    droll = np.clip(droll * ang_scale, -ang_limit, ang_limit)

                    _motor = self.motor_map['wrist_roll']
                    _value = self.target_pos[_motor] + droll
                    self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
            
            # Wrist yaw (direct control)
            curr_vr_wrist_yaw = vr_goal.wrist_yaw_deg
            prev_vr_wrist_yaw = self._prev_vr_goal.wrist_yaw_deg
            if curr_vr_wrist_yaw is not None and prev_vr_wrist_yaw is not None:
                dyaw = curr_vr_wrist_yaw - prev_vr_wrist_yaw

                if abs(dyaw) > _dead_zones['yaw']:
                    dyaw = dyaw * _weights['yaw']
                    dyaw = np.clip(dyaw * ang_scale, -ang_limit, ang_limit)

                    _motor = self.motor_map['wrist_yaw']
                    _value = self.target_pos[_motor] + dyaw
                    self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)

            # Wrist position (xyz)
            curr_vr_pos = vr_goal.target_position # [x, y, z] in meters
            prev_vr_pos = self._prev_vr_goal.target_position
            if curr_vr_pos is not None and prev_vr_pos is not None:
                dx = curr_vr_pos[0] - prev_vr_pos[0]
                dy = curr_vr_pos[1] - prev_vr_pos[1]
                dz = curr_vr_pos[2] - prev_vr_pos[2]

                # VR's x motion directly controls shoulder_pan
                if abs(dx) > _dead_zones['dx']:
                    dx = dx * _weights['dx']
                    dx = np.clip(dx * ang_scale, -ang_limit, ang_limit)
                    _motor = self.motor_map['shoulder_pan']
                    _value = self.target_pos[_motor] + dx
                    self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
                
                # VR's y is upward
                if abs(dy) > _dead_zones['dy']:
                    dy = dy * _weights['dy']
                    dy = np.clip(dy * pos_scale, -pos_limit, pos_limit)
                    self._target_xy[1] += dy
                
                # VR's z is in arm's negative x direction
                if abs(dz) > _dead_zones['dz']:
                    dz = dz * _weights['dz']
                    dz = np.clip(dz * pos_scale, -pos_limit, pos_limit)
                    self._target_xy[0] += -dz

                # Solve IK to get angles in degrees
                self._target_xy = self.kinematics.apply_workspace_bound(*self._target_xy)[:2]
                jnt2, jnt3 = self.kinematics.inverse_kinematics(*self._target_xy)
                # TODO: convert angles (deg) to raw and then to normalized values
                jnt2_raw = self._deg_to_raw(self.motor_map['shoulder_lift'], jnt2)
                jnt3_raw = self._deg_to_raw(self.motor_map['elbow_flex'], jnt3)
                jnt2_norm = self._bus._normalize({2: jnt2_raw})[2]
                jnt3_norm =self._bus._normalize({3: jnt3_raw})[3]
                self.target_pos[self.motor_map['shoulder_lift']] = jnt2_norm
                self.target_pos[self.motor_map['elbow_flex']] = jnt3_norm

                # Wrist flex control: wrist_flex
                _motor = self.motor_map['wrist_flex']
                jnt4 = -jnt2 - jnt3 - self._target_pitch
                # TODO: convert degrees to raw and then to normalized values
                jnt4_raw = self._deg_to_raw(_motor, jnt4)
                jnt4_norm = self._bus._normalize({4: jnt4_raw})[4]
                self.target_pos[_motor] = jnt4_norm

            # Handle gripper state directly
            global GRIPPER_TOGGLE_CLOSE
            if vr_goal.metadata.get('trigger', 0) > 0.5 and not GRIPPER_TOGGLE_CLOSE:
                _motor = self.motor_map['gripper']
                _value = self.target_pos[_motor] + self.stepsize.get('joint', 1.0)
                self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
            elif vr_goal.metadata.get('trigger', 0) > 0.5 and GRIPPER_TOGGLE_CLOSE:
                _motor = self.motor_map['gripper']
                _value = self.target_pos[_motor] - self.stepsize.get('joint', 1.0)
                self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        else:
            self.logger.info(f"[{self.name}] 🧊 Squeeze/grip button is pressed, arm is frozen...")
        
        # Apply EMA to target position, higher alpha value means more smoothing
        self.apply_ema_to_target_pos(ema_alpha=0.9)
        # Update previous VR goal
        self._prev_vr_goal = copy.deepcopy(vr_goal)
        # Print target if any
        self.log_target_pos()


class BaseControl(SimpleBaseControl):
    def __init__(self, name: str, robot: XLerobotYaw, control_freq: int = 50, logger: logging.Logger | None = None):
        super().__init__(name, robot, control_freq, logger)
    
    def set_target_from_states(self, states: dict[str, Any]) -> None:
        vr_goal = states.get('vr_goal', None)
        if vr_goal is None: return

        thumb = vr_goal.metadata.get('thumbstick', {})
        if not thumb: return

        # Cache the previous target
        self.update_prev_base_target()

        pressed_keys = set()
        thumb_x = thumb.get('x', 0)
        thumb_y = thumb.get('y', 0)
        _dead_zone = 0.5
        if abs(thumb_x) > _dead_zone:
            if thumb_x > 0:
                pressed_keys.add(self.robot.teleop_keys['rotate_right'])
            else:
                pressed_keys.add(self.robot.teleop_keys['rotate_left'])
        if abs(thumb_y) > _dead_zone:
            if thumb_y > 0:
                pressed_keys.add(self.robot.teleop_keys['backward'])
            else:
                pressed_keys.add(self.robot.teleop_keys['forward'])
        
        self.base_target = self.robot._from_keyboard_to_base_action(list(pressed_keys))

        # Apply EMA to base target, higher alpha value means more smoothing
        self.apply_ema_to_base_target()
        # Print base target if any
        self.log_base_target()

# --------------- Main ---------------
@dataclass
class Config:
    id: str = "xlerobot_yaw01" # Name of the robot, used to retrieve the robot calibration file
    control_freq: int = 60 # Hz
    console_level: str = 'info' # Logging level
    stepsize: float | dict[str, float] = field(default_factory=lambda: {'joint': 0.8, 'xy': 0.001}) # Stepsize
    enable_left_arm_control: bool = True # Enable left arm control with VR controller; disable to use only right arm
    enable_head_control: bool = False # Enable head control with VR headset;
    use_thumbstick_for_head: bool = False # Use left VR controller's thumbstick for head control; False: use VR headset

@draccus.wrap()
def main(cfg: Config):
    logger = init_logging(cfg.console_level)
    logger.info("[MAIN] 🚀 Starting teleoperation...")
    
     # Connect to devices
    try:
        robot_config = XLerobotYawConfig()
        robot_config.id = cfg.id
        robot = XLerobotYaw(robot_config)
        robot.connect()
        logger.info(f"[MAIN] ✅ Successfully connected to devices: {robot.name}")
        if robot.is_calibrated:
            logger.info(f"[MAIN] ✅ Robot is calibrated and ready to use!")
        else:
            logger.info(f"[MAIN] ⚠️ Robot requires calibration")
    except Exception as e:
        robot.disconnect()
        logger.error(f"[MAIN] ❌ Failed to connect devices: {e}")
        traceback.print_exc()
        return

    # Initialize VR monitor
    try:
        logger.info("[MAIN] 🔧 Initializing VR monitor...")
        vr_monitor = VRMonitor()
        logger.info("[MAIN] 🚀 Starting VR monitoring...")
        vr_thread = threading.Thread(target=lambda: asyncio.run(vr_monitor.start_monitoring()), daemon=True)
        vr_thread.start()
        logger.info("[MAIN] ✅ VR system ready")
    except Exception as e:
        logger.error(f"[MAIN] ❌ VR monitor initialization failed: {e}")
        traceback.print_exc()
        return
    
    # Setting up controls
    if cfg.enable_left_arm_control:
        left_ctrl = SimpleArmControl(
            robot,
            RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
            name='larm',
            bus_name='bus1',
            stepsize=cfg.stepsize,
            control_freq=cfg.control_freq,
            motor_map=LEFT_JOINT_MAP,
            logger=logger
        )
    else:
        left_ctrl = None
    right_ctrl = SimpleArmControl(
        robot,
        RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
        name='rarm',
        bus_name='bus2',
        stepsize=cfg.stepsize,
        control_freq=cfg.control_freq,
        motor_map=RIGHT_JOINT_MAP,
        logger=logger
    )
    head_ctrl = SimpleHeadControl(
        robot,
        name='head',
        bus_name='bus1',
        stepsize=cfg.stepsize,
        control_freq=cfg.control_freq,
        motor_map=HEAD_JOINT_MAP,
        logger=logger,
        use_thumbstick=cfg.use_thumbstick_for_head
    )
    base_ctrl = BaseControl(
        name='base',
        robot=robot,
        control_freq=cfg.control_freq,
        logger=logger
    )

    def move_all_to_zero(duration: float = 3.0):
        offsets = (
            {
                LEFT_JOINT_MAP['shoulder_lift']: -80,
                LEFT_JOINT_MAP['elbow_flex']: 80,
                LEFT_JOINT_MAP['wrist_roll']: -50
            },
            {
                RIGHT_JOINT_MAP['shoulder_lift']: -80,
                RIGHT_JOINT_MAP['elbow_flex']: 80,
                RIGHT_JOINT_MAP['wrist_roll']: -50
            },
            {HEAD_JOINT_MAP['head_pitch']: 10}
        )
        for ctrl, offset in zip([left_ctrl, right_ctrl, head_ctrl], offsets):
            if ctrl is not None:
                ctrl.move_to_zero_position(duration=duration, offset=offset)
                ctrl._prev_vr_goal = None
        base_ctrl.reset_base_target()
    
    def move_all_to_init(duration: float = 3.0):
        for ctrl in [left_ctrl, right_ctrl, head_ctrl]:
            if ctrl is not None: ctrl.move_to_initial_position(duration=duration)
        base_ctrl.reset_base_target()

    # Main control loop
    move_all_to_zero(3)
    SAFE_EXIT =  False
    global GRIPPER_TOGGLE_CLOSE
    GRIPPER_TOGGLE_CLOSE = False
    LAST_TIME_TOGGLE_GRIPPER = time.time()
    LAST_TIME_RESET_VR_GOAL = time.time()

    try:
        while True:
            # Get VR controller data
            dual_goals = vr_monitor.get_latest_goal_nowait()

            # Wait for VR connection before proceeding
            if dual_goals is None:
                time.sleep(0.01)  # Wait 10ms for VR connection
                continue

            left_goal = dual_goals.get("left") if dual_goals else None
            right_goal = dual_goals.get("right") if dual_goals else None
            if cfg.use_thumbstick_for_head:
                head_goal = left_goal # Use left hand thumbstick for head control
            else:
                head_goal = dual_goals.get("headset") if dual_goals else None

            # :----- Check for reset or exit -----:
            # Left hand Y button: reset to zero position
            # Left hand X button: exit the program
            if left_goal is not None and left_goal.metadata.get('buttons', {}).get('y', False):
                logger.info("[MAIN] ♻️ Resetting to zero position...")
                move_all_to_zero(3)
                continue

            if left_goal is not None and left_goal.metadata.get('buttons', {}).get('x', False):
                logger.info("[MAIN] 👋 Exiting the program...")
                move_all_to_init(3)
                SAFE_EXIT = True
                break

            # :----- Toggle gripper close state -----:
            # Right hand A button: toggle gripper close state
            if right_goal is not None and right_goal.metadata.get('buttons', {}).get('a', False):
                if time.time() - LAST_TIME_TOGGLE_GRIPPER > 0.5:
                    GRIPPER_TOGGLE_CLOSE = not GRIPPER_TOGGLE_CLOSE
                    LAST_TIME_TOGGLE_GRIPPER = time.time()
                    logger.info(f"[MAIN] 🤏 {'Gripper trigger to close' if GRIPPER_TOGGLE_CLOSE else 'Gripper trigger to open'}")
            
            # :----- Reset VR goal baseline -----:
            # Right hand B button: reset VR goal baseline
            if right_goal is not None and right_goal.metadata.get('buttons', {}).get('b', False):
                if time.time() - LAST_TIME_RESET_VR_GOAL > 0.5:
                    # Reset VR server origin
                    vr_monitor.vr_server.left_controller.reset_origin()
                    vr_monitor.vr_server.right_controller.reset_origin()
                    # Reset controller baseline
                    for ctrl in [left_ctrl, right_ctrl, head_ctrl]:
                        if ctrl is not None:
                            ctrl._prev_vr_goal = None
                    LAST_TIME_RESET_VR_GOAL = time.time()
                    logger.info("[MAIN] 🎯 Reset VR goal baseline")

            # :----- Set target -----:
            right_ctrl.set_target_from_states({'vr_goal': right_goal})
            base_ctrl.set_target_from_states({'vr_goal': right_goal})
            if cfg.enable_left_arm_control:
                left_ctrl.set_target_from_states({'vr_goal': left_goal})
            if cfg.enable_head_control:
                head_ctrl.set_target_from_states({'vr_goal': head_goal})

            # :----- Get action -----:
            right_action = right_ctrl.get_action_dict()
            base_action = base_ctrl.get_action_dict()
            left_action = left_ctrl.get_action_dict() if cfg.enable_left_arm_control else {}
            head_action = head_ctrl.get_action_dict() if cfg.enable_head_control else {}
            
            # :----- Merge and send action to robot -----:
            action = {**left_action, **right_action, **head_action, **base_action}
            robot.send_action(action)

            time.sleep(1 / cfg.control_freq)

    except Exception as e:
        logger.error(f"[MAIN] Error in teleoperation loop: {e}")
        traceback.print_exc()

    finally:
        if not SAFE_EXIT: move_all_to_init(3)
        robot.disconnect()

if __name__ == "__main__":
    main()
