# To Run on the host
'''
PYTHONPATH=src python -m lerobot.robots.xlerobot_yaw.xlerobot_yaw_host --robot.id=my_xlerobot_yaw
'''

# To Run the teleop:
'''
PYTHONPATH=src python -m examples.xlerobot.5_xlerobot_yaw_teleop_ps5
'''

# Standard library imports
import time
import logging
import traceback
from dataclasses import dataclass, field

# Third-party imports
import numpy as np
import pygame
import draccus

# Local imports
from utils.utils import init_logging
from utils.teleop_utils import SimpleControl, SimpleBaseControl
from lerobot.robots.xlerobot_yaw import XLerobotYawConfig, XLerobotYaw
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.rr_kinematics import RRKinematics

# --------------- PS5 控制键映射 ---------------
# 控制按键映射 (semantic action -> controller mapping)
LEFT_KEYMAP: dict[str, str] = {
    # 左臂 XY 控制 (左摇杆; 未按下)
    'x+': 'left_stick_up',
    'x-': 'left_stick_down',
    'y+': 'left_stick_right',
    'y-': 'left_stick_left',
    # 左臂 shoulder_pan 和 pitch 控制 (LB 按下 + 左摇杆)
    'pitch+': 'lb_up',
    'pitch-': 'lb_down',
    'shoulder_pan+': 'lb_right',
    'shoulder_pan-': 'lb_left',
    # 左臂 wrist_yaw (new) 和 wrist_roll 控制 (LB 按下 + D-pad)
    'wrist_roll+': 'lb_dpad_up',
    'wrist_roll-': 'lb_dpad_down',
    'wrist_yaw+': 'lb_dpad_right',
    'wrist_yaw-': 'lb_dpad_left',
    # 左夹爪控制 (LT)
    'gripper+': 'lt',
    'gripper-': 'lb_lt'
}
RIGHT_KEYMAP = {
    # 右臂 XY 控制 (右摇杆; 未按下)
    'x+': 'right_stick_up',
    'x-': 'right_stick_down',
    'y+': 'right_stick_right',
    'y-': 'right_stick_left',
    # 右臂 shoulder_pan 和 pitch 控制 (RB 按下 + 左摇杆)
    'pitch+': 'rb_up',
    'pitch-': 'rb_down',
    'shoulder_pan+': 'rb_right',
    'shoulder_pan-': 'rb_left',
    # 右臂 wrist_yaw (new) 和 wrist_roll 控制 (RB 按下 + xots)
    'wrist_roll+': 'rb_xots_up',
    'wrist_roll-': 'rb_xots_down',
    'wrist_yaw+': 'rb_xots_right',
    'wrist_yaw-': 'rb_xots_left',
    # 右夹爪控制 (RT)
    'gripper+': 'rt',
    'gripper-': 'rb_rt'
}
HEAD_KEYMAP = {
    # 头部电机控制 (x, o, s, t)
    "head_yaw+": 'o',
    "head_yaw-": 's',
    "head_pitch+": 't',
    "head_pitch-": 'x'
}
BASE_KEYMAP = {
    # 底盘控制
    'forward': 'dpad_up',
    'backward': 'dpad_down',
    'left': 'right_stick_pressed_dpad_left',
    'right': 'right_stick_pressed_dpad_right',
    'rotate_left': 'dpad_left',
    'rotate_right': 'dpad_right',
    'speed_up': 'options' # 增加底盘速度, 最高 3 档
}

RESET_KEYMAP = {
    'zero': 'share', # 回到零点位置
    'exit': 'ps'
}

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

# ------------ Controllers ------------
class SimpleHeadControl(SimpleControl):
    def __init__(
        self,
        robot: XLerobotYaw,
        name: str = "Head",
        bus_name: str = 'bus1',
        stepsize: float | dict[str, float] = 1.0,
        control_freq: int = 50,
        motor_map: dict[str, str] = HEAD_JOINT_MAP,
        logger: logging.Logger | None = None
    ):
        super().__init__(name, robot, bus_name, stepsize, control_freq, motor_map, logger)
        self._motors = self._bus.motors
       
    def set_target_from_states(self, states: dict[str, bool]) -> None:
        """
        Set the target position from the semantic action states.
        """
        stepsize_joint = self.stepsize.get('joint', 1.0)

        # Cache the previous target
        self.update_prev_target_pos()

        if states.get('head_yaw+'):
            _motor = self.motor_map['head_yaw']
            _value = self.target_pos[_motor] + stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('head_yaw-'):
            _motor = self.motor_map['head_yaw']
            _value = self.target_pos[_motor] - stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('head_pitch+'):
            _motor = self.motor_map['head_pitch']
            _value = self.target_pos[_motor] + stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('head_pitch-'):
            _motor = self.motor_map['head_pitch']
            _value = self.target_pos[_motor] - stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        
        # Apply EMA to base target, higher alpha value means more smoothing
        self.apply_ema_to_target_pos(ema_alpha=0.9)
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

    def move_to_initial_position(self, duration: float = 5.0) -> None:
        self._target_xy = None
        self._target_pitch = None
        super().move_to_initial_position(duration)
    
    def move_to_zero_position(self, duration: float = 5.0, offset: dict[str, float] = {}) -> None:
        self._target_xy = None
        self._target_pitch = None
        super().move_to_zero_position(duration, offset)
    
    def set_target_from_states(self, states: dict[str, bool]) -> None:
        """
        Set the target position from the semantic action states.
        """
        stepsize_joint = self.stepsize.get('joint', 1.0)
        stepsize_xy = self.stepsize.get('xy', 0.001)

        # Cache the previous target
        self.update_prev_target_pos()
        
        # Direct joint control: shoulder_pan, wrist_roll, wrist_yaw
        if states.get('shoulder_pan+'):
            _motor = self.motor_map['shoulder_pan']
            _value = self.target_pos[_motor] + stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('shoulder_pan-'):
            _motor = self.motor_map['shoulder_pan']
            _value = self.target_pos[_motor] - stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('wrist_roll+'):
            _motor = self.motor_map['wrist_roll']
            _value = self.target_pos[_motor] + stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('wrist_roll-'):
            _motor = self.motor_map['wrist_roll']
            _value = self.target_pos[_motor] - stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('wrist_yaw+'):
            _motor = self.motor_map['wrist_yaw']
            _value = self.target_pos[_motor] + stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('wrist_yaw-'):
            _motor = self.motor_map['wrist_yaw']
            _value = self.target_pos[_motor] - stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        
        # Gripper control
        if states.get('gripper+'):
            _motor = self.motor_map['gripper']
            _value = self.target_pos[_motor] + stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)
        if states.get('gripper-'):
            _motor = self.motor_map['gripper']
            _value = self.target_pos[_motor] - stepsize_joint
            self.target_pos[_motor] = self._clip_norm_value(_value, self._motors[_motor].norm_mode)

        
        # Pitch: angle above horizontal x-axis, in degrees;
        # this indirectly controls wrist_flex

        # Get the current joint angles in degrees
        names_values = self.get_motor_values_deg([
            self.motor_map['shoulder_lift'],
            self.motor_map['elbow_flex'],
            self.motor_map['wrist_flex']
        ])
        jnt2 = names_values[self.motor_map['shoulder_lift']]
        jnt3 = names_values[self.motor_map['elbow_flex']]
        jnt4 = names_values[self.motor_map['wrist_flex']]

        if self._target_pitch is None:
            self._target_pitch = 0
        
        if states.get('pitch+'):
            _value = self._target_pitch + stepsize_joint
            self._target_pitch = np.clip(_value, -90, 90)
        if states.get('pitch-'):
            _value = self._target_pitch - stepsize_joint
            self._target_pitch = np.clip(_value, -90, 90)
        
        # XY plane (IK) control: shoulder_lift, elbow_flex
        if self._target_xy is None:
            self._target_xy = self.kinematics.forward_kinematics(jnt2, jnt3)
        
        if states.get('x+'):
            self._target_xy[0] += stepsize_xy
        if states.get('x-'):
            self._target_xy[0] -= stepsize_xy
        if states.get('y+'):
            self._target_xy[1] += stepsize_xy
        if states.get('y-'):
            self._target_xy[1] -= stepsize_xy
        if any([states.get('x+'), states.get('x-'), states.get('y+'), states.get('y-')]):
            self._target_xy = self.kinematics.apply_workspace_bound(*self._target_xy)[:2]
            self.logger.info(f"[{self.name}] Target XY position: ({self._target_xy[0]:.4f}, {self._target_xy[1]:.4f})")

            # Solve IK to get angles in degrees
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

        # Apply EMA to target position, higher alpha value means more smoothing
        self.apply_ema_to_target_pos(ema_alpha=0.9)
        # Print target if any
        self.log_target_pos()


class BaseControl(SimpleBaseControl):
    def __init__(
        self,
        name: str,
        robot: XLerobotYaw,
        joystick: pygame.joystick.Joystick,
        control_freq: int = 50,
        logger: logging.Logger | None = None
    ):
        super().__init__(name, robot, control_freq, logger)
        self.joystick = joystick
        self.last_speed_up_time = time.time()
    
    def set_target_from_states(self, states: dict[str, bool]) -> None:
        # Set base speed level
        if states.get('speed_up') and (time.time() - self.last_speed_up_time) > 0.2: # debounce, avoid multiple presses within 0.2s
            self.robot.set_base_speed_index((self.robot.base_speed_index + 1) % len(self.robot.base_speed_levels)) # cycle through speed levels
            self.logger.info(f"[{self.name}] Base speed index: {self.robot.base_speed_index}")
            self.joystick.rumble(0.7, 0.9, 500) # vibrate the controller for 500ms
            self.last_speed_up_time = time.time()
        
        # Cache the previous target
        self.update_prev_base_target()
        
        # Set base target
        ctrls = ['forward', 'backward', 'left', 'right', 'rotate_left', 'rotate_right']
        pressed_keys = set([self.robot.teleop_keys[k] for k in ctrls if states.get(k)]) # get the corresponding keyboard keys    
        self.base_target = self.robot._from_keyboard_to_base_action(list(pressed_keys))
        
        # Apply EMA to base target, higher alpha value means more smoothing
        self.apply_ema_to_base_target()
        # Print base target if any
        self.log_base_target()

# --------------- PS5 按键状态映射 ---------------
def get_ps5_states(joystick: pygame.joystick.Joystick, keymap: dict[str, str]) -> dict[str, bool]:
    """
    Map PS5 controller state to semantic action booleans using the provided keymap.
    Reference: https://www.pygame.org/docs/ref/joystick.html#playstation-5-controller-pygame-2-x
    *Note*: 't' and 's' are swapped. It is better to run the test code from pygame to get the correct mapping.
    - 'x': button 0
    - 'o': button 1
    - 't': button 2
    - 's': button 3
    """
    # Read axes, buttons, hats
    axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
    hat = joystick.get_hat(0) if joystick.get_numhats() > 0 else (0, 0)
    assert len(axes) == 6, "Expected 6 axes, got {}".format(len(axes))
    assert len(buttons) == 13, "Expected 13 buttons, got {}".format(len(buttons))

    # 获取摇杆按下状态
    left_stick_pressed = bool(buttons[11])
    right_stick_pressed = bool(buttons[12])
    lb_pressed = bool(buttons[4])
    rb_pressed = bool(buttons[5])
    
    # 构建 state 字典 (semantic action -> boolean)
    state = {}
    for action, control in keymap.items():
        # 夹爪控制
        if control == 'lt':
            state[action] = not lb_pressed and axes[2] > 0.5
        elif control == 'lb_lt':
            state[action] = lb_pressed and axes[2] > 0.5
        elif control == 'rt':
            state[action] = not rb_pressed and axes[5] > 0.5
        elif control == 'rb_rt':
            state[action] = rb_pressed and axes[5] > 0.5
        # 头部电机控制 (RB 未按下)
        elif control == 'x':
            state[action] = not rb_pressed and bool(buttons[0])
        elif control == 'o':
            state[action] = not rb_pressed and bool(buttons[1])
        # Note: t and s are swapped
        elif control == 't':
            state[action] = not rb_pressed and bool(buttons[2])
        elif control == 's':
            state[action] = not rb_pressed and bool(buttons[3])
        # 底盘控制 (LB 未按下 + 右摇杆未按下)
        elif control == 'dpad_up':
            state[action] = (not lb_pressed) and (not right_stick_pressed) and (hat[1] == 1)
        elif control == 'dpad_down':
            state[action] = (not lb_pressed) and (not right_stick_pressed) and (hat[1] == -1)
        elif control == 'dpad_left':
            state[action] = (not lb_pressed) and (not right_stick_pressed) and (hat[0] == -1)
        elif control == 'dpad_right':
            state[action] = (not lb_pressed) and (not right_stick_pressed) and (hat[0] == 1)
        # 底盘 lateral 控制 (LB 未按下 + 右摇杆按下)
        elif control == 'right_stick_pressed_dpad_left':
            state[action] = (not lb_pressed) and right_stick_pressed and (hat[0] == -1)
        elif control == 'right_stick_pressed_dpad_right':
            state[action] = (not lb_pressed) and right_stick_pressed and (hat[0] == 1)
        # 左臂 XY 控制 (左摇杆未按下 + LB 未按下)
        elif control == 'left_stick_up':
            state[action] = (not left_stick_pressed) and (not lb_pressed) and (axes[1] < -0.5)
        elif control == 'left_stick_down':
            state[action] = (not left_stick_pressed) and (not lb_pressed) and (axes[1] > 0.5)
        elif control == 'left_stick_left':
            state[action] = (not left_stick_pressed) and (not lb_pressed) and (axes[0] < -0.5)
        elif control == 'left_stick_right':
            state[action] = (not left_stick_pressed) and (not lb_pressed) and (axes[0] > 0.5)
        # 右臂 XY 控制 (右摇杆未按下 + RB 未按下)
        elif control == 'right_stick_up':
            state[action] = (not right_stick_pressed) and (not rb_pressed) and (axes[4] < -0.5)
        elif control == 'right_stick_down':
            state[action] = (not right_stick_pressed) and (not rb_pressed) and (axes[4] > 0.5)
        elif control == 'right_stick_left':
            state[action] = (not right_stick_pressed) and (not rb_pressed) and (axes[3] < -0.5)
        elif control == 'right_stick_right':
            state[action] = (not right_stick_pressed) and (not rb_pressed) and (axes[3] > 0.5)
        # 左臂 shoulder_pan 和 pitch 控制 (LB 按下)
        elif control == 'lb_right':
            state[action] = lb_pressed and (axes[0] > 0.5)
        elif control == 'lb_left':
            state[action] = lb_pressed and (axes[0] < -0.5)
        elif control == 'lb_up':
            state[action] = lb_pressed and (axes[1] < -0.5)
        elif control == 'lb_down':
            state[action] = lb_pressed and (axes[1] > 0.5)
        # 右臂 shoulder_pan 和 pitch 控制 (RB 按下)
        elif control == 'rb_right':
            state[action] = rb_pressed and (axes[3] > 0.5)
        elif control == 'rb_left':
            state[action] = rb_pressed and (axes[3] < -0.5)
        elif control == 'rb_up':
            state[action] = rb_pressed and (axes[4] < -0.5)
        elif control == 'rb_down':
            state[action] = rb_pressed and (axes[4] > 0.5)
        # 左臂 wrist_yaw 和 wrist_roll 控制 (LB 按下 + D-pad)
        elif control == 'lb_dpad_left':
            state[action] = lb_pressed and (hat[0] == -1)
        elif control == 'lb_dpad_right':
            state[action] = lb_pressed and (hat[0] == 1)
        elif control == 'lb_dpad_down':
            state[action] = lb_pressed and (hat[1] == -1)
        elif control == 'lb_dpad_up':
            state[action] = lb_pressed and (hat[1] == 1)
        # 右臂 wrist_yaw 和 wrist_roll 控制 (RB 按下 + xots)
        elif control == 'rb_xots_left':
            state[action] = rb_pressed and bool(buttons[3])
        elif control == 'rb_xots_right':
            state[action] = rb_pressed and bool(buttons[1])
        elif control == 'rb_xots_down':
            state[action] = rb_pressed and bool(buttons[0])
        elif control == 'rb_xots_up':
            state[action] = rb_pressed and bool(buttons[2])
        # 底盘速度控制 (增加底盘速度)
        elif control == 'options':
            state[action] = bool(buttons[9])
        # 重置, 返回零点位置
        elif control == 'share':
            state[action] = bool(buttons[8])
        # 退出, 并返回初始位置
        elif control == 'ps':
            state[action] = bool(buttons[10])
        else:
            state[action] = False
    return state

# --------------- Main ---------------
@dataclass
class Config:
    id: str = "xlerobot_yaw01" # Name of the robot, used to retrieve the robot calibration file
    control_freq: int = 60 # Hz
    console_level: str = 'info' # Logging level
    stepsize: float | dict[str, float] = field(default_factory=lambda: {'joint': 0.8, 'xy': 0.001}) # Stepsize

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
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        # init_rerun(session_name="xlerobot_yaw_teleop_ps5")
        logger.info(f"[MAIN] ✅ Successfully connected to devices: {robot.name, joystick.get_name()}")
        if robot.is_calibrated:
            logger.info(f"[MAIN] ✅ Robot is calibrated and ready to use!")
        else:
            logger.info(f"[MAIN] ⚠️ Robot requires calibration")
    except Exception as e:
        robot.disconnect()
        logger.error(f"[MAIN] ❌ Failed to connect devices: {e}")
        traceback.print_exc()
        return

    # Setting up controls
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
        logger=logger
    )
    base_ctrl = BaseControl(
        name='base',
        robot=robot,
        joystick=joystick,
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
            ctrl.move_to_zero_position(duration=duration, offset=offset)
        base_ctrl.reset_base_target()
    
    def move_all_to_init(duration: float = 3.0):
        for ctrl in [left_ctrl, right_ctrl, head_ctrl]:
            ctrl.move_to_initial_position(duration=duration)
        base_ctrl.reset_base_target()
        
    # Main control loop
    move_all_to_zero(3)
    SAFE_EXIT =  False

    try:
        while True:
            pygame.event.pump()
            reset_states = get_ps5_states(joystick, RESET_KEYMAP)

            # Check for reset or exit
            if reset_states['zero']:
                move_all_to_zero(3)
                continue
            if reset_states['exit']:
                logger.info("[MAIN] 👋 Exiting the program, returning to initial position...")
                move_all_to_init(3)
                SAFE_EXIT = True
                break

            # :----- Get PS5 state -----:
            left_states = get_ps5_states(joystick, LEFT_KEYMAP) # left arm
            right_states = get_ps5_states(joystick, RIGHT_KEYMAP) # right arm
            head_states = get_ps5_states(joystick, HEAD_KEYMAP) # head
            base_states = get_ps5_states(joystick, BASE_KEYMAP) # base
            
            # :----- Set target -----:
            left_ctrl.set_target_from_states(left_states)
            right_ctrl.set_target_from_states(right_states)
            head_ctrl.set_target_from_states(head_states)
            base_ctrl.set_target_from_states(base_states)

            # :----- Get action dict -----:
            left_action = left_ctrl.get_action_dict()
            right_action = right_ctrl.get_action_dict()
            head_action = head_ctrl.get_action_dict()
            base_action = base_ctrl.get_action_dict()

            # :----- Merge and send action to robot -----:
            action = {**left_action, **right_action, **head_action, **base_action}
            robot.send_action(action)
            
            # :----- Log data -----:
            # obs = robot.get_observation()
            # log_rerun_data(obs, action)
            
            time.sleep(1 / cfg.control_freq)
    except:
        logger.error("[MAIN] ❌ Error in teleoperation loop")
        traceback.print_exc()

    finally:
        if not SAFE_EXIT: move_all_to_init(3)
        robot.disconnect()

if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()