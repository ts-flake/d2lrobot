import time
import copy
import logging
import traceback

import numpy as np

from ..teleoperator import Teleoperator
from .config_xlerobot_gamepad import XLeRobotGamepadConfig
from .gamepad_utils import ALL_KEYMAP, get_gamepad_states, PS5Gamepad, print_decode_keymap

logger = logging.getLogger(__name__)

class XLeRobotGamepad(Teleoperator):
    config_class = XLeRobotGamepadConfig
    name = "xlerobot_gamepad"

    def __init__(self, config: XLeRobotGamepadConfig, gamepad: PS5Gamepad | None = None):
        super().__init__(config)
        self.config = config
        self.gamepad = gamepad

        self.safe_exit = False
        self.last_time_base_speed_up = time.time()

        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "exit_teleop": False,
            "back_robot_to_zero": False,
        }
    
    @property
    def is_connected(self) -> bool:
        return self.gamepad is not None and self.gamepad.is_connected()

    def connect(self) -> None:
        if self.gamepad is not None:
            self.gamepad.connect()
            print_decode_keymap(ALL_KEYMAP)
        
    
    @property
    def is_calibrated(self) -> None:
        pass
    
    def calibrate(self) -> None:
        pass
    
    def configure(self) -> None:
        pass
    
    @property
    def action_features(self) -> dict[str, type]:
        return {
            "left_arm_ee_delta.x": float,
            "left_arm_ee_delta.y": float,
            "left_arm_ee_delta.z": float,
            "left_arm_ee_delta.roll": float,
            "left_arm_ee_delta.pitch": float,
            "left_arm_gripper.pos": float,
            "right_arm_ee_delta.x": float,
            "right_arm_ee_delta.y": float,
            "right_arm_ee_delta.z": float,
            "right_arm_ee_delta.roll": float,
            "right_arm_ee_delta.pitch": float,
            "right_arm_gripper.pos": float,
            "head.pitch": float,
            "head.yaw": float,
            "base_action": list[str],
        }
    
    def get_action(self) -> dict[str, float]:
        action = dict.fromkeys(self.action_features.keys(), 0.0)
        action["base_action"] = []

        states = get_gamepad_states(self.gamepad, ALL_KEYMAP)

        # :----- Check for record control events -----:
        if self.config.record_dataset:
            if states.get('exit_early', False):
                self.events["exit_early"] = True
                logger.info("🚫 Exiting the loop...")
                return action
            if states.get('rerecord_episode', False):
                self.events["exit_early"] = True
                self.events["rerecord_episode"] = True
                logger.info("🔄 Exiting the loop and re-recording the last episode...")
                return action
            if states.get('exit_early', False):
                self.events["exit_early"] = True
                self.events["stop_recording"] = True
                logger.info("🛑 Stopping data recording...")
                return action

        # :----- Check for reset or exit -----:
        if states.get('back_robot_to_zero', False):
            self.events["back_robot_to_zero"] = True
            logger.info("🔄 Resetting to zero position...")
            return action
        if states.get('exit_teleop', False):
            self.events["exit_teleop"] = True
            self.events["exit_early"] = True
            self.safe_exit = True
            logger.info("👋 Exiting the teleop loop...")
            return action
        
        # :----- Set action -----:
        stepsize = self.config.stepsize
        stepsize_pos = stepsize.get('pos', 0.01)
        stepsize_ang = stepsize.get('ang', 1.0)
        stepsize_gripper = stepsize.get('gripper', 2.0)

        stepsize_factors = self.config.stepsize_factors
        stepsize_factors_arm = stepsize_factors.get('arm', {})
        stepsize_factors_head = stepsize_factors.get('head', {})

        # Head control
        if self.config.enable_head_control:
            dpitch = stepsize_ang * stepsize_factors_head.get('pitch', 1)
            dyaw = stepsize_ang * stepsize_factors_head.get('yaw', 1)
            if states.get('head.pitch+', False) or states.get('head.pitch-', False):
                action["head.pitch"] = (dpitch if states.get('head.pitch+', False) else -dpitch)
            if states.get('head.yaw+', False) or states.get('head.yaw-', False):
                action["head.yaw"] = (dyaw if states.get('head.yaw+', False) else -dyaw)

        def _set_action_arm(prefix):
            dx, dz = stepsize_pos, stepsize_pos
            dy = stepsize_ang * stepsize_factors_arm.get('y', 1) # y-axis motion directly controls shoulder_pan
            if states.get(f"{prefix}.x+", False) or states.get(f"{prefix}.x-", False):
                action[f"{prefix}_ee_delta.x"] = (dx if states.get(f"{prefix}.x+", False) else -dx)
            if states.get(f"{prefix}.shoulder_pan+", False) or states.get(f"{prefix}.shoulder_pan-", False):
                action[f"{prefix}_ee_delta.y"] = (dy if states.get(f"{prefix}.shoulder_pan+", False) else -dy)
            if states.get(f"{prefix}.z+", False) or states.get(f"{prefix}.z-", False):
                action[f"{prefix}_ee_delta.z"] = (dz if states.get(f"{prefix}.z+", False) else -dz)
            
            droll = stepsize_ang * stepsize_factors_arm.get('roll', 1)
            dpitch = stepsize_ang * stepsize_factors_arm.get('pitch', 1)
            if states.get(f"{prefix}.wrist_roll+", False) or states.get(f"{prefix}.wrist_roll-", False):
                action[f"{prefix}_ee_delta.roll"] = (droll if states.get(f"{prefix}.wrist_roll+", False) else -droll)
            if states.get(f"{prefix}.pitch+", False) or states.get(f"{prefix}.pitch-", False):
                action[f"{prefix}_ee_delta.pitch"] = (dpitch if states.get(f"{prefix}.pitch+", False) else -dpitch)
            
        def _set_action_gripper(prefix):
            if states.get(f"{prefix}.gripper+", False) or states.get(f"{prefix}.gripper-", False):
                action[f"{prefix}_gripper.pos"] = (stepsize_gripper if states.get(f"{prefix}.gripper+", False) else -stepsize_gripper)
        
        if self.config.enable_left_arm_control:
            # Left hand
            _set_action_arm("left_arm")
        
        # Right hand
        _set_action_arm("right_arm")
        
        # Gripper (always active)
        _set_action_gripper("left_arm")
        _set_action_gripper("right_arm")

        # Base (always active)
        if states.get('base.forward', False):
            action["base_action"].append('forward')
        if states.get('base.backward', False):
            action["base_action"].append('backward')
        if states.get('base.left', False):
            action["base_action"].append('left')
        if states.get('base.right', False):
            action["base_action"].append('right')
        if states.get('base.rotate_left', False):
            action["base_action"].append('rotate_left')
        if states.get('base.rotate_right', False):
            action["base_action"].append('rotate_right')
        if states.get('base.speed_up', False) and (time.time() - self.last_time_base_speed_up) > 0.2: # debounce, avoid multiple presses within 0.2s
            action["base_action"].append('speed_up')
            self.last_time_base_speed_up = time.time()
        
        return action

    @property
    def feedback_features(self) -> dict[str, type]:
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError
    
    def disconnect(self) -> None:
        if self.gamepad is not None:
            self.gamepad.disconnect()