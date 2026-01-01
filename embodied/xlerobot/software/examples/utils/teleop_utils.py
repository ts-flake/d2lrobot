import abc
import logging
from pprint import pformat
from typing import Any, Callable


import time
import numpy as np

from lerobot.robots.robot import Robot
from lerobot.motors.motors_bus import MotorNormMode

from .control_utils import get_trajectory_fn, TrajectoryMode


class SimpleControl(abc.ABC):
    def __init__(
        self,
        name: str,
        robot: Robot,
        bus_name: str = 'bus',
        stepsize: float | dict[str, float] = {'joint': 1.0, 'xy': 0.01},
        control_freq: int = 50,
        motor_map: dict[str, str] = {},
        logger: logging.Logger | None = None
    ):
        self.name = name.upper()
        self.robot = robot
        self._bus = getattr(robot, bus_name)
        assert type(stepsize) in [dict, float], f"Stepsize must be a dict or a float, got {type(stepsize)}"
        self.stepsize = stepsize if isinstance(stepsize, dict) else {'joint': stepsize, 'xy': stepsize}
        self.control_freq = control_freq # Hz
        self.motor_map = motor_map
        self.logger = logger if logger is not None else logging.getLogger(self.name)

        self.initial_pos = self.get_motor_values() # set initial position to current motor values
        self.target_pos = {motor: 0.0 for motor in self.motor_map.values()} # initialize target position to zero
        self._prev_target_pos = self.target_pos.copy()
        self.logger.debug(f"[{self.name}] Initial position:\n{pformat(self.initial_pos, indent=4)}")
    
    def update_prev_target_pos(self) -> None:
        self._prev_target_pos = self.target_pos.copy()
    
    def apply_ema_to_target_pos(self, ema_alpha: float | dict[str, float]) -> None:
        if isinstance(ema_alpha, float):
            ema_alpha = {motor: ema_alpha for motor in self.motor_map.values()}
        for motor in self.motor_map.values():
            _alpha = ema_alpha.get(motor, 1.0)
            self.target_pos[motor] = self.target_pos[motor] * _alpha + self._prev_target_pos[motor] * (1 - _alpha)
    
    def log_target_pos(self) -> None:
        _print_target_pos = {k:v for k,v in self.target_pos.items() if self.target_pos[k] != self._prev_target_pos[k]}
        if _print_target_pos:
            self.logger.info(f"[{self.name}] Target position:\n{pformat(_print_target_pos, indent=4)}")

    def get_motor_values(self, motors: str | list[str] | None = None, normalize: bool = True) -> dict[str, float]:
        if motors is None:
            motors = list(self.motor_map.values())
        return self._bus.sync_read("Present_Position", motors, normalize=normalize)
    
    def _raw_to_deg(self, motor: str,value: int) -> float:
        max_res = self._bus.model_resolution_table[self._bus._id_to_model(self._bus.motors[motor].id)] - 1
        _value = (value - int(max_res / 2)) * 360 / max_res
        if self._bus.apply_drive_mode and self._bus.calibration[motor].drive_mode:
            _value = - _value
        return _value

    def _deg_to_raw(self, motor: str, value: float) -> int:
        if self._bus.apply_drive_mode and self._bus.calibration[motor].drive_mode:
            value = - value
        max_res = self._bus.model_resolution_table[self._bus._id_to_model(self._bus.motors[motor].id)] - 1
        _value = (value * max_res / 360) + int(max_res / 2)
        return _value

    def get_motor_values_deg(self, motors: str | list[str] | None = None) -> dict[str, float]:
        # TODO: We use the half-turn raw value as the 0 degree position. The actual motor position
        # depends on the calibration process, i.e., homing offset.
        motor_values = self.get_motor_values(motors, normalize=False)
        for motor, value in motor_values.items():
            motor_values[motor] = self._raw_to_deg(motor, value)
        return motor_values
    
    def _clip_norm_value(self, value: float, norm_mode: MotorNormMode = MotorNormMode.RANGE_M100_100) -> float:
        if norm_mode == MotorNormMode.RANGE_M100_100:
            return max(-100.0, min(100.0, value))
        elif norm_mode == MotorNormMode.RANGE_0_100:
            return max(0.0, min(100.0, value))
        elif norm_mode == MotorNormMode.DEGREES:
            return max(-180.0, min(180.0, value))
        else:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")
    
    @abc.abstractmethod
    def set_target_from_states(self, states: dict[str, Any]) -> None:
        """
        This method is used to set the target position from the semantic action states.
        """
        pass

    def get_action_dict(self) -> dict[str, float]:
        return {f"{motor}.pos": value for motor, value in self.target_pos.items()}

    def move_to_initial_position(self, duration: float = 3.0) -> None:
        self.logger.info(f"[{self.name}] Moving to initial position in {duration} seconds...")
        sampler = self.get_action_sampler(
            target_pos=self.initial_pos,
            duration=duration,
            traj_mode=TrajectoryMode.MIN_JERK
        )
        t = 0.0
        dt = 1.0 / self.control_freq
        while t < duration:
            pos, _, _ = sampler(t)
            self.target_pos.update({motor: value for motor, value in zip(self.motor_map.values(), pos)})
            action = self.get_action_dict()
            self.robot.send_action(action)
            time.sleep(dt)
            t += dt

    def move_to_zero_position(self, duration: float = 3.0, offset: dict[str, float] = {}) -> None:
        self.logger.info(f"[{self.name}] Moving to zero position in {duration} seconds...")
        sampler = self.get_action_sampler(
            target_pos={motor: offset.get(motor, 0.0) for motor in self.motor_map.values()},
            duration=duration,
            traj_mode=TrajectoryMode.MIN_JERK
        )
        t = 0.0
        dt = 1.0 / self.control_freq
        while t < duration:
            pos, _, _ = sampler(t)
            self.target_pos.update({motor: value for motor, value in zip(self.motor_map.values(), pos)})
            action = self.get_action_dict()
            self.robot.send_action(action)
            time.sleep(dt)
            t += dt

    def get_action_sampler(
        self,
        target_pos: dict[str, float] | None = None,
        duration: float = 1.0,
        traj_mode: TrajectoryMode = TrajectoryMode.MIN_JERK,
        **kwargs: Any
    ) -> Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if target_pos is not None:
            if type(target_pos) != dict:
                raise TypeError(f"Target position must be a dict, got {type(target_pos)}")
            if not all(motor in self.motor_map.values() for motor in target_pos):
                raise ValueError(f"Target position keys must be in {list(self.motor_map.values())}")
            self.target_pos.update(target_pos)
        start_pos = np.array(list(self.get_motor_values().values()))
        end_pos = np.array([self.target_pos[motor] for motor in self.motor_map.values()])
        traj_fn = get_trajectory_fn(traj_mode)
        sampler = traj_fn(start_pos, end_pos, duration=duration, **kwargs)
        return sampler


class SimpleBaseControl(abc.ABC):
    def __init__(self, name: str, robot: Robot, control_freq: int = 50, logger: logging.Logger | None = None):
        self.name = name.upper()
        self.robot = robot
        self.control_freq = control_freq # Hz
        self.logger = logger if logger is not None else logging.getLogger(self.name)

        self.base_target = dict.fromkeys(['x.vel', 'y.vel', 'theta.vel'], 0.0)
        self._prev_base_target = self.base_target.copy()

    def reset_base_target(self) -> None:
        self.base_target = dict.fromkeys(['x.vel', 'y.vel', 'theta.vel'], 0.0)
        self._prev_base_target = self.base_target.copy()
    
    def get_action_dict(self) -> dict[str, float]:
        return self.base_target

    def update_prev_base_target(self) -> None:
        self._prev_base_target = self.base_target.copy()

    def apply_ema_to_base_target(self, speed_up_time: float = 5.0, stop_time: float = 0.5) -> None:
        # Start slowly and stop quickly, smooth out base speed change while in motion
        calc_alpha = lambda a0, t, hz: 1.0 - a0 ** (1 / (t * hz))
        if all(v == 0.0 for v in self._prev_base_target.values()): # previous is at rest
            ema_alpha = calc_alpha(0.01, speed_up_time, self.control_freq) # speed up
        else:
            ema_alpha = calc_alpha(0.01, stop_time, self.control_freq) # stop
        self.base_target = {k: self.base_target[k] * ema_alpha + self._prev_base_target[k] * (1 - ema_alpha) for k in self.base_target}
        self.base_target = {k: 0.0 if abs(v) < 3e-3 else v for k, v in self.base_target.items()} # ignore small values
    
    def log_base_target(self) -> None:
        if not all(v == 0.0 for v in self.base_target.values()):
            self.logger.info(f"[{self.name}] Velocity target:\n{pformat(self.base_target, indent=4)}")

    @abc.abstractmethod
    def set_target_from_states(self, states: dict[str, Any]) -> None:
        """
        This method is used to set the target position from the semantic action states.
        """
        pass
