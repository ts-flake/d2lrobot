import time
import copy
import asyncio
import logging
import threading
import traceback
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from .config_xlerobot_vr import XLeRobotVRConfig
from .vr_monitor import VRMonitor

logger = logging.getLogger(__name__)

class XLeRobotVR(Teleoperator):
    config_class = XLeRobotVRConfig
    name = "xlerobot_vr"

    def __init__(self, config: XLeRobotVRConfig):
        super().__init__(config)
        self.config = config
        self.vr_monitor: VRMonitor | None = None
        self._vr_thread: threading.Thread | None = None

        self._prev_left_active = False
        self._prev_right_active = False
        self.safe_exit = False
        self.last_time_toggle_gripper = time.time()
        self.last_time_reset_vr_goal = time.time()

        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "exit_teleop": False,
            "back_robot_to_zero": False,
            "gripper_toggle_close": False
        }
    
    @property
    def is_connected(self) -> bool:
        return self.vr_monitor is not None and self.vr_monitor.is_running

    def connect(self) -> None:
        # Initialize VR monitor
        try:
            vr_monitor = VRMonitor()
            vr_thread = threading.Thread(target=lambda: asyncio.run(vr_monitor.start_monitoring()), daemon=True)
            vr_thread.start()
            self._vr_thread = vr_thread
        except Exception as e:
            logger.error(f"❌ VR monitor initialization failed: {e}")
            traceback.print_exc()
            return
        self.vr_monitor = vr_monitor
        # self.calibrate() # Calibration is done in the VR monitor
    
    @property
    def is_calibrated(self) -> bool:
        return self.is_connected and self.vr_monitor.is_calibrated
    
    def calibrate(self) -> None:
        self.vr_monitor.calibrate()
    
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

        dual_goals = self.vr_monitor.get_latest_goal_nowait()
        if dual_goals is None:
            return action
        
        left_goal = dual_goals.get("left") if dual_goals else None
        right_goal = dual_goals.get("right") if dual_goals else None
        base_goal = right_goal
        if self.config.use_thumbstick_for_head:
            headset_goal = left_goal
        else:
            headset_goal = dual_goals.get("headset") if dual_goals else None

        # :----- Check for record control events -----:
        if self.config.record_dataset and left_goal is not None and left_goal.metadata.get('thumbstick', {}):
            thumb = left_goal.metadata['thumbstick']
            thumb_x = thumb.get('x', 0.0)
            thumb_y = thumb.get('y', 0.0)
            if thumb_x > 0.5: # right
                self.events["exit_early"] = True
                logger.info("🚫 Exiting the loop...")
                return action
            if thumb_x < -0.5: # left
                self.events["exit_early"] = True
                self.events["rerecord_episode"] = True
                logger.info("🔄 Exiting the loop and re-recording the last episode...")
                return action
            if thumb_y < -0.5: # up
                self.events["exit_early"] = True
                self.events["stop_recording"] = True
                logger.info("🛑 Stopping data recording...")
                return action

        # :----- Check for reset or exit -----:
        # Left hand Y button: reset to zero position
        # Left hand X button: exit the program
        if left_goal is not None and left_goal.metadata.get('buttons', {}).get('y', False):
            self.events["back_robot_to_zero"] = True
            logger.info("🔄 Resetting to zero position...")
            return action
        if left_goal is not None and left_goal.metadata.get('buttons', {}).get('x', False):
            self.events["exit_teleop"] = True
            self.events["exit_early"] = True
            self.safe_exit = True
            logger.info("👋 Exiting the teleop loop...")
            return action
        
        # :----- Check for gripper toggle -----:
        # Right hand A button: toggle gripper close state
        if right_goal is not None and right_goal.metadata.get('buttons', {}).get('a', False):
            if time.time() - self.last_time_toggle_gripper > 0.5:
                self.events["gripper_toggle_close"] = not self.events["gripper_toggle_close"]
                self.last_time_toggle_gripper = time.time()
        
        # :----- Check for reset VR goal baseline -----:
        # Right hand B button: reset VR goal baseline
        if right_goal is not None and right_goal.metadata.get('buttons', {}).get('b', False):
            if time.time() - self.last_time_reset_vr_goal > 0.5:
                self.calibrate()
                self.last_time_reset_vr_goal = time.time()
                logger.info("🎯 Reset VR goal baseline")
        
        # :----- Set action -----:
        stepsize = self.config.stepsize
        stepsize_pos = stepsize.get('pos', 0.01)
        stepsize_ang = stepsize.get('ang', 1.0)
        stepsize_gripper = stepsize.get('gripper', 2.0)
        
        dead_zones = self.config.vr2robot_dead_zones
        dead_zones_head = dead_zones.get('head', {})
        dead_zones_arm = dead_zones.get('arm', {})

        factors = self.config.vr2robot_factors
        factors_head = factors.get('head', {})
        factors_arm = factors.get('arm', {})

        limits = self.config.vr2robot_limits
        limits_head = limits.get('head', {})
        limits_arm = limits.get('arm', {})

        # Activate pose tracking
        left_active = left_goal is not None and left_goal.metadata.get('buttons', {}).get('squeeze', False)
        right_active = right_goal is not None and right_goal.metadata.get('buttons', {}).get('squeeze', False)
        if not self.config.grip_to_activate: # Reverse: press the grip button to deactivate
            left_active = not left_active
            right_active = not right_active
        
        if not left_active and self._prev_left_active:
            logger.info("🧊 Left controller (head, left arm) is deactivated")
        if not right_active and self._prev_right_active:
            logger.info("🧊 Right controller (right arm) is deactivated")
        self._prev_left_active = left_active
        self._prev_right_active = right_active

        # Head control
        if left_active and self.config.enable_head_control:
            # Use left hand thumbstick for head control
            if self.config.use_thumbstick_for_head:
                thumb = headset_goal.metadata.get('thumbstick', {})
                thumb_x = thumb.get('x', 0.0)
                thumb_y = thumb.get('y', 0.0)
                
                if abs(thumb_x) > 0.5:
                    sign = 1 if thumb_x < 0 else -1
                    action["head.yaw"] = sign * stepsize_ang
                if abs(thumb_y) > 0.5:
                    sign = 1 if thumb_y < 0 else -1
                    action["head.pitch"] = sign * stepsize_ang
            else: # Use VR headset for head control
                dyaw = headset_goal.wrist_yaw_deg
                dpitch = headset_goal.wrist_flex_deg
                if abs(dyaw) > dead_zones_head.get('yaw', 0.5):
                    dyaw *= factors_head.get('yaw', 10)
                    _limit = limits_head.get('ang', 1.0)
                    dyaw = np.clip(dyaw, -_limit, _limit)
                    action["head.yaw"] = dyaw
                if abs(dpitch) > dead_zones_head.get('pitch', 0.5):
                    dpitch *= factors_head.get('pitch', 10)
                    _limit = limits_head.get('ang', 1.0)
                    dpitch = np.clip(dpitch, -_limit, _limit)
                    action["head.pitch"] = dpitch
        
        def _set_action_arm(goal, prefix):
            dx, dy, dz = goal.target_position
            if abs(dx) > dead_zones_arm.get('dx', 0.001):
                dx *= factors_arm.get('dx', 1)
                _limit = limits_arm.get('pos', 0.005)
                dx = np.clip(dx, -_limit, _limit)
                action[f"{prefix}_ee_delta.x"] = dx
            if abs(dy) > dead_zones_arm.get('dy', 0.001):
                dy *= factors_arm.get('dy', 1)
                _limit = limits_arm.get('ang', 1.0) # y-axis motion directly controls shoulder_pan
                dy = np.clip(dy, -_limit, _limit)
                action[f"{prefix}_ee_delta.y"] = dy
            if abs(dz) > dead_zones_arm.get('dz', 0.001):
                dz *= factors_arm.get('dz', 1)
                _limit = limits_arm.get('pos', 0.005)
                dz = np.clip(dz, -_limit, _limit)
                action[f"{prefix}_ee_delta.z"] = dz
            
            droll = goal.wrist_roll_deg
            dpitch = goal.wrist_flex_deg
            if abs(droll) > dead_zones_arm.get('roll', 0.3):
                droll *= factors_arm.get('roll', 100)
                _limit = limits_arm.get('ang', 1.0)
                droll = np.clip(droll, -_limit, _limit)
                action[f"{prefix}_ee_delta.roll"] = droll
            if abs(dpitch) > dead_zones_arm.get('pitch', 0.3):
                dpitch *= factors_arm.get('pitch', 100)
                _limit = limits_arm.get('ang', 1.0)
                dpitch = np.clip(dpitch, -_limit, _limit)
                action[f"{prefix}_ee_delta.pitch"] = dpitch
            
        def _set_action_gripper(goal, prefix):
            if goal.metadata.get('trigger', 0) > 0.5:
                if not self.events["gripper_toggle_close"]:
                    action[f"{prefix}_gripper.pos"] = stepsize_gripper
                else:
                    action[f"{prefix}_gripper.pos"] = -stepsize_gripper
        
        if left_active and self.config.enable_left_arm_control:
            # Left hand
            _set_action_arm(left_goal, "left_arm")
        if right_active:
            # Right hand
            _set_action_arm(right_goal, "right_arm")
        
        # Gripper (always active)
        _set_action_gripper(left_goal, "left_arm")
        _set_action_gripper(right_goal, "right_arm")

        # Base (always active)
        if base_goal is not None:
            thumb = base_goal.metadata.get('thumbstick', {})
            thumb_x = thumb.get('x', 0.0)
            thumb_y = thumb.get('y', 0.0)
            if abs(thumb_x) > 0.5:
                if thumb_x > 0:
                    action["base_action"].append('rotate_right')
                else:
                    action["base_action"].append('rotate_left')
            if abs(thumb_y) > 0.5:
                if thumb_y > 0:
                    action["base_action"].append('backward')
                else:
                    action["base_action"].append('forward')
        
        return action

    @property
    def feedback_features(self) -> dict[str, type]:
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError
    
    def disconnect(self) -> None:
        if self.vr_monitor is not None:
            # Signal the monitor to stop by setting is_running = False
            # The monitor's finally block in start_monitoring() will call stop_monitoring()
            # which is async and will be properly awaited within its own event loop
            self.vr_monitor.is_running = False
            
            # Wait for the VR thread to finish (with timeout)
            if self._vr_thread is not None and self._vr_thread.is_alive():
                self._vr_thread.join(timeout=5.0)
                if self._vr_thread.is_alive():
                    logger.warning("VR monitor thread did not stop within timeout")
        
        self.vr_monitor = None
        self._vr_thread = None