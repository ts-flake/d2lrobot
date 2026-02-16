#!/usr/bin/env python

import time
import logging
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.utils import log_say
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_xlerobot_yaw import XLeRobotYawConfig

logger = logging.getLogger(__name__)


class XLeRobotYaw(Robot):
    """
    XLeRobotYaw 基于 XLeRobot V0.3 改装.
    原始的 XLeRobot 包含一个移动底盘 (3 omniwheels) 和两个 SO101 机械臂.
    (新增) 改装后的机械臂 SO101Yaw 多加一个 wrist yaw 关节, 一共 6+1 DoFs.

    *注意*: 默认电机旋转顺时针为正.
    Note: The default motor rotation is clockwise.

    ## 机器人组装位置示意 (top view):

                  <forward>
        +⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯+
        |          [base]           |        
        | [left arm]    [right arm] |
        +⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯+
    
    , where [base] is
               
            (back wheel)
                 /\
                /__\
     (right wheel) (left wheel)
        
    ## 机器人坐标系定义 (Frame definition):
    机械臂 (Arm):
    - x: forward direction
    - y: left direction
    - z: follows the right-hand rule

    底盘:
    - x: forward direction
    - y: left direction
    - z: follows the right-hand rule
    """
    config_class = XLeRobotYawConfig
    name = "xlerobot_yaw"

    # Chassis parameters
    _wheel_radius: float = 0.05
    _base_radius: float = 0.125
    _wheel_max_raw: int = 3000
    _wheel_mounting_angles: list[float] = [240, 0, 120] # left, back, right in clockwise order, in degrees

    def __init__(self, config: XLeRobotYawConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys
        self._step_per_deg = (1<<12) / 360.0 # 12-bit encoder

        # Define three speed levels and a current index
        self.base_speed_levels = config.base_speed_levels
        self._base_speed_index = 0  # start from 0 (slow)

        # Compute base jacobian matrix
        angles = np.deg2rad(self._wheel_mounting_angles)
        betas = angles + np.pi / 2 # wheel driving directions,❗注意: 电机旋转顺时针为正
        self._base_jacobian = np.array([[self._base_radius, np.cos(beta), np.sin(beta)] for beta in betas]) # _base_radius 取正号, 因为电机旋转方向和底盘旋转方向相同
        self._base_jacobian_inv = np.linalg.inv(self._base_jacobian)

        # Set up motor buses and calibration
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        ## Left arm + head
        if self.calibration.get("left_arm_shoulder_pan") is not None:
            calibration1 = {
                "left_arm_shoulder_pan": self.calibration.get("left_arm_shoulder_pan"),
                "left_arm_shoulder_lift": self.calibration.get("left_arm_shoulder_lift"),
                "left_arm_elbow_flex": self.calibration.get("left_arm_elbow_flex"), 
                "left_arm_wrist_flex": self.calibration.get("left_arm_wrist_flex"),
                "left_arm_wrist_yaw": self.calibration.get("left_arm_wrist_yaw"),  # 新增的关节
                "left_arm_wrist_roll": self.calibration.get("left_arm_wrist_roll"),
                "left_arm_gripper": self.calibration.get("left_arm_gripper"),
                "head_yaw": self.calibration.get("head_yaw"),
                "head_pitch": self.calibration.get("head_pitch"),
            }
        else:
            calibration1 = self.calibration
        self.bus1 = FeetechMotorsBus(
            port=self.config.port1,
            motors={
                # left arm
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_yaw": Motor(5, "sts3215", norm_mode_body),  # 新增的关节
                "left_arm_wrist_roll": Motor(6, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
                # head
                "head_yaw": Motor(8, "sts3215", norm_mode_body),  # 相机 yaw
                "head_pitch": Motor(9, "sts3215", norm_mode_body),  # 相机 pitch
            },
            calibration= calibration1,
        )
        ## Right arm + base
        if self.calibration.get("right_arm_shoulder_pan") is not None:
            calibration2 = {
                "right_arm_shoulder_pan": self.calibration.get("right_arm_shoulder_pan"),
                "right_arm_shoulder_lift": self.calibration.get("right_arm_shoulder_lift"),
                "right_arm_elbow_flex": self.calibration.get("right_arm_elbow_flex"),
                "right_arm_wrist_flex": self.calibration.get("right_arm_wrist_flex"),
                "right_arm_wrist_yaw": self.calibration.get("right_arm_wrist_yaw"),  # 新增的关节
                "right_arm_wrist_roll": self.calibration.get("right_arm_wrist_roll"),
                "right_arm_gripper": self.calibration.get("right_arm_gripper"),
                "base_left_wheel": self.calibration.get("base_left_wheel"),
                "base_back_wheel": self.calibration.get("base_back_wheel"),
                "base_right_wheel": self.calibration.get("base_right_wheel"),
            }
        else:
            calibration2 = self.calibration
        self.bus2= FeetechMotorsBus(
            port=self.config.port2,
            motors={
                # right arm
                "right_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "right_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "right_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "right_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "right_arm_wrist_yaw": Motor(5, "sts3215", norm_mode_body),
                "right_arm_wrist_roll": Motor(6, "sts3215", norm_mode_body),  # 新增的关节
                "right_arm_gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
                # base
                "base_left_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=calibration2,
        )

        # Record motors: list[str] and cameras: dict[str, CameraConfig]
        self.left_arm_motors = [motor for motor in self.bus1.motors if motor.startswith("left_arm")]
        self.right_arm_motors = [motor for motor in self.bus2.motors if motor.startswith("right_arm")]
        self.head_motors = [motor for motor in self.bus1.motors if motor.startswith("head")]
        self.base_motors = [motor for motor in self.bus2.motors if motor.startswith("base")]
        self.cameras = make_cameras_from_configs(config.cameras)
    
    @property
    def base_speed_index(self) -> int:
        return self._base_speed_index
    
    def set_base_speed_index(self, value: int) -> None:
        value = int(value) % len(self.base_speed_levels)
        self._base_speed_index = value
        log_say(f"Base speed index set to {value}")
    
    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_yaw.pos",  # 新增的关节
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_yaw.pos",  # 新增的关节
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
                "head_yaw.pos",
                "head_pitch.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus1.is_connected and self.bus2.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus1.connect()
        self.bus2.connect()
        
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus1.is_calibrated and self.bus2.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # 校准文件存在，询问用户是否使用现有校准文件或重新校准
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}," 
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus1.write_calibration({motor: self.calibration[motor] for motor in self.left_arm_motors + self.head_motors})
                self.bus2.write_calibration({motor: self.calibration[motor] for motor in self.right_arm_motors + self.base_motors})
                return
        
        # 重新校准
        logger.info(f"\nRunning calibration of {self}")
        
        ## 校准左臂和头部
        left_motors = self.left_arm_motors + self.head_motors
        self.bus1.disable_torque()
        for name in left_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
        input(
            "Move \033[32mleft arm\033[0m and \033[32mhead\033[0m motors "
            "to the \033[1mmiddle\033[0m of their range of motion and press ENTER...."
        )
        homing_offsets = self.bus1.set_half_turn_homings(left_motors)
        
        print(
            "Move all \033[32mleft arm\033[0m and \033[32mhead\033[0m joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus1.record_ranges_of_motion(left_motors)
        
        calibration_left = {}
        for name, motor in self.bus1.motors.items():
            calibration_left[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0, # 电机方向不反转
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )
        self.bus1.write_calibration(calibration_left)
        
        ## 校准右臂和基座
        right_motors = self.right_arm_motors + self.base_motors
        self.bus2.disable_torque(self.right_arm_motors)
        for name in self.right_arm_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)
        
        input(
            "Move \033[32mright arm\033[0m motors "
            "to the \033[1mmiddle\033[0m of their range of motion and press ENTER...."
        )
        
        homing_offsets = self.bus2.set_half_turn_homings(self.right_arm_motors)
        homing_offsets.update(dict.fromkeys(self.base_motors, 0))
        
        full_turn_motor = [
            motor for motor in right_motors if any(keyword in motor for keyword in ["wheel"])
        ]
        unknown_range_motors = [motor for motor in right_motors if motor not in full_turn_motor]
        print(
            f"Move all \033[32mright arm\033[0m joints \033[3mexcept\033[0m '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus2.record_ranges_of_motion(unknown_range_motors)
        for name in full_turn_motor:
            range_mins[name] = 0
            range_maxes[name] = 4095
        
        calibration_right = {}
        for name, motor in self.bus2.motors.items():
            calibration_right[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0, # 电机方向不反转
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )
        self.bus2.write_calibration(calibration_right)

        # 合并校准文件,保存到本地
        self.calibration = {**calibration_left, **calibration_right}
        self._save_calibration()
        print(f"\033[92mCalibration saved to\033[0m {self.calibration_fpath}")

    def configure(self) -> None:
        # Set-up arm actuators (position mode)
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration        
        self.bus1.disable_torque()
        self.bus2.disable_torque()
        self.bus2.configure_motors()
        self.bus2.configure_motors()
        
        for name in self.left_arm_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus1.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus1.write("I_Coefficient", name, 0)
            self.bus1.write("D_Coefficient", name, 32)
        
        for name in self.head_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus1.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus1.write("I_Coefficient", name, 0)
            self.bus1.write("D_Coefficient", name, 32)

            if "gripper" in name:
                    self.bus1.write(
                        "Max_Torque_Limit", name, 500
                    )  # 50% of the max torque limit to avoid burnout
                    self.bus1.write("Protection_Current", name, 250)  # 50% of max current to avoid burnout
                    self.bus1.write("Overload_Torque", name, 25)  # 25% torque when overloaded
        
        for name in self.right_arm_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus2.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus2.write("I_Coefficient", name, 0)
            self.bus2.write("D_Coefficient", name, 32)

            if "gripper" in name:
                    self.bus2.write(
                        "Max_Torque_Limit", name, 500
                    )  # 50% of the max torque limit to avoid burnout
                    self.bus2.write("Protection_Current", name, 250)  # 50% of max current to avoid burnout
                    self.bus2.write("Overload_Torque", name, 25)  # 25% torque when overloaded
        
        for name in self.base_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        
        self.bus1.enable_torque()
        self.bus2.enable_torque()
        
    def setup_motors(self) -> None:
        # Set up left arm motors
        for motor in chain(reversed(self.left_arm_motors), reversed(self.head_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus1.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus1.motors[motor].id}")
        
        # Set up right arm motors
        for motor in chain(reversed(self.right_arm_motors), reversed(self.base_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus2.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus2.motors[motor].id}")
    
    def _raw_to_deg(self, motor: str, value: int) -> float:
        _bus = self.bus1 if motor in self.left_arm_motors + self.head_motors else self.bus2
        max_res = _bus.model_resolution_table[_bus._id_to_model(_bus.motors[motor].id)] - 1
        _value = (value - int(max_res / 2)) * 360 / max_res
        if _bus.apply_drive_mode and _bus.calibration[motor].drive_mode:
            _value = - _value
        return _value

    def _deg_to_raw(self, motor: str, value: float) -> int:
        _bus = self.bus1 if motor in self.left_arm_motors + self.head_motors else self.bus2
        if _bus.apply_drive_mode and _bus.calibration[motor].drive_mode:
            value = - value
        max_res = _bus.model_resolution_table[_bus._id_to_model(_bus.motors[motor].id)] - 1
        _value = (value * max_res / 360) + int(max_res / 2)
        return _value
    
    def _normalize(self, motor: str, value: int) -> float:
        _bus = self.bus1 if motor in self.left_arm_motors + self.head_motors else self.bus2
        _id = _bus.motors[motor].id
        return _bus._normalize({_id: value})[_id]

    def _unnormalize(self, motor: str, value: int) -> float:
        _bus = self.bus1 if motor in self.left_arm_motors + self.head_motors else self.bus2
        _id = _bus.motors[motor].id
        return _bus._unnormalize({_id: value})[_id]

    def _clip_norm_value(self, motor: str,value: float) -> float:
        norm_mode = (
            self.bus1.motors[motor].norm_mode
            if motor in self.left_arm_motors + self.head_motors
            else self.bus2.motors[motor].norm_mode)
        if norm_mode == MotorNormMode.RANGE_M100_100:
            return max(-100.0, min(100.0, value))
        elif norm_mode == MotorNormMode.RANGE_0_100:
            return max(0.0, min(100.0, value))
        elif norm_mode == MotorNormMode.DEGREES:
            return max(-180.0, min(180.0, value))
        else:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

    def _degps_to_raw(self, degps: float) -> int:
        return int(round(degps * self._step_per_deg))

    def _raw_to_degps(self, raw_speed: int) -> float:
        return raw_speed / self._step_per_deg

    def _body_to_wheel_raw(
        self,
        x_vel: float,
        y_vel: float,
        theta_vel: float
    ) -> dict[str, int]:
        """
        Convert desired body-frame velocities into wheel raw commands.
        `x_vel`, `y_vel`, `theta_vel` are in m/s and deg/s respectively.

        If any value exceeds the maximum allowed raw command (ticks) per wheel,
        all commands are scaled down proportionally to ensure correct driving direction.

        Formula from https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=531.29
        """
        _theta_vel = np.deg2rad(theta_vel)
        body_twist = np.array([_theta_vel, x_vel, y_vel])

        # Forward mapping: wheel linear speed = J * body twist
        wheel_linear_speeds = self._base_jacobian @ body_twist
        wheel_degps = np.rad2deg(wheel_linear_speeds / self._wheel_radius)

        # Scaling
        max_raw = max([abs(degps) * self._step_per_deg for degps in wheel_degps])
        if max_raw > self._wheel_max_raw:
            wheel_degps = wheel_degps * (self._wheel_max_raw / max_raw)

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(degps) for degps in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed
    ) -> dict[str, float]:
        """
        Convert wheel raw command feedback back into body twist.
        `left_wheel_speed`, `back_wheel_speed`, `right_wheel_speed` are in ticks.

        Formula from https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=531.29
        """
        wheel_degps = [self._raw_to_degps(speed) for speed in [left_wheel_speed, back_wheel_speed, right_wheel_speed]]
        wheel_radps = np.deg2rad(wheel_degps)
        wheel_linear_speeds = wheel_radps * self._wheel_radius

        # Backward mapping: body twist = J⁻¹ * wheel linear speed
        body_twist = self._base_jacobian_inv @ wheel_linear_speeds
        theta_vel, x_vel, y_vel = body_twist

        return {
            "x.vel": x_vel, # m/s
            "y.vel": y_vel, # m/s
            "theta.vel": np.rad2deg(theta_vel), # deg/s
        }
    
    def _from_keyboard_to_base_action(self, pressed_keys: list[str]) -> dict[str, float]:
        # Set speed level
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.set_base_speed_index(self.base_speed_index + 1)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.set_base_speed_index(self.base_speed_index - 1)
        
        speed_setting = self.base_speed_levels[self.base_speed_index]
        xy_speed = speed_setting["xy"]  # m/s
        theta_speed = speed_setting["theta"]  # deg/s

        # Set speed command
        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed
            
        return {
            "x.vel": x_cmd, 
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read actuators position for arm and vel for base
        start = time.perf_counter()
        left_arm_pos = self.bus1.sync_read("Present_Position", self.left_arm_motors, normalize=True)
        right_arm_pos = self.bus2.sync_read("Present_Position", self.right_arm_motors, normalize=True)
        head_pos = self.bus1.sync_read("Present_Position", self.head_motors, normalize=True)
        base_wheel_vel = self.bus2.sync_read("Present_Velocity", self.base_motors, normalize=True)
        
        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_arm_pos.items()}
        head_state = {f"{k}.pos": v for k, v in head_pos.items()}
        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )
        
        obs_dict = {**left_arm_state, **right_arm_state, **head_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command xlerobot_yaw to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        right_arm_pos = {k: v for k, v in action.items() if k.startswith("right_arm_") and k.endswith(".pos")}
        head_pos = {k: v for k, v in action.items() if k.startswith("head_") and k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}
        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("y.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )
        
        if self.config.max_relative_target is not None and not self.config.skip_goal_position_check:
            # Read present positions for left arm, right arm, and head
            present_pos_left = self.bus1.sync_read("Present_Position", self.left_arm_motors)
            present_pos_right = self.bus2.sync_read("Present_Position", self.right_arm_motors)
            present_pos_head = self.bus1.sync_read("Present_Position", self.head_motors)

            # Combine all present positions
            present_pos = {**present_pos_left, **present_pos_right, **present_pos_head}

            # Ensure safe goal position for each arm and head
            goal_present_pos = {
                key: (g_pos, present_pos[key.removesuffix(".pos")])
                for key, g_pos in chain(left_arm_pos.items(), right_arm_pos.items(), head_pos.items())
            }
            safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            # Update the action with the safe goal positions
            left_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in left_arm_pos}
            right_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in right_arm_pos}
            head_pos = {k: v for k, v in safe_goal_pos.items() if k in head_pos}
        
        left_arm_pos_raw = {k.removesuffix(".pos"): v for k, v in left_arm_pos.items()}
        right_arm_pos_raw = {k.removesuffix(".pos"): v for k, v in right_arm_pos.items()}
        head_pos_raw = {k.removesuffix(".pos"): v for k, v in head_pos.items()}
        
        # Only sync_write if there are motors to write to
        # dict[str, value]: motor name -> value
        if left_arm_pos_raw:
            self.bus1.sync_write("Goal_Position", left_arm_pos_raw)
        if right_arm_pos_raw:
            self.bus2.sync_write("Goal_Position", right_arm_pos_raw)
        if head_pos_raw:
            self.bus1.sync_write("Goal_Position", head_pos_raw)
        if base_wheel_goal_vel:
            self.bus2.sync_write("Goal_Velocity", base_wheel_goal_vel)
        return {
            **left_arm_pos,
            **right_arm_pos,
            **head_pos,
            **base_goal_vel,
        }

    def stop_base(self):
        self.bus2.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus1.disconnect(self.config.disable_torque_on_disconnect)
        self.bus2.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
