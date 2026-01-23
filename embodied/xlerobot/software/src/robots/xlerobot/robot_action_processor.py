from dataclasses import dataclass, field
from typing import Any
from pprint import pformat
import logging

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.model.rr_kinematics import RRKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.rotation import Rotation
from lerobot.robots.xlerobot import XLeRobot


@ProcessorStepRegistry.register("analytical_inverse_kinematics_delta_to_joints")
@dataclass
class AnalyticalInverseKinematicsDeltaToJoints(RobotActionProcessorStep):
    """
    Computes the target arm joint positions from the delta commands using analytical inverse kinematics (IK).
    The analytical IK is only partial, i.e., for a EE delta commmand (dx, dy, dz, droll, dpitch, dyaw),
    the kinematics model solves for the 'shoulder_lift' and 'elbow_flex' from (dx, dz),
    and uses (dx, droll, dpitch, dyaw) to directly compute 'shoulder_pan', 'wrist_roll', 'wrist_flex', and 'wrist_yaw'.

    Attributes:
        kinematics_left: The kinematic model for the left arm.
        kinematics_right: The kinematic model for the right arm.
        robot: The robot instance.
        motor_names: The names of the motors to control.
    """
    kinematics_left: RRKinematics
    kinematics_right: RRKinematics
    robot: XLeRobot
    motor_names: list[str]

    def __post_init__(self):
        self._target_pitch_left = None
        self._target_pitch_right = None
        self._target_xz_left = None
        self._target_xz_right = None

    def reset(self):
        self._target_pitch_left = None
        self._target_pitch_right = None
        self._target_xz_left = None
        self._target_xz_right = None

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_dict = {
            k.removesuffix(".pos"):float(v)
            for k, v in observation.items()
            if isinstance(k, str)
            and k.endswith(".pos")
            and k.removesuffix(".pos") in self.motor_names
        }

        if q_dict is None:
            raise ValueError("Joints observation is require for computing robot kinematics")
        
        # Head yaw
        dyaw_head = action.pop("head.yaw")
        if dyaw_head is not None and q_dict.get("head_yaw") is not None:
            q_dict["head_yaw"] -= dyaw_head

        # Head pitch
        dpitch_head = action.pop("head.pitch")
        if dpitch_head is not None and q_dict.get("head_pitch") is not None:
            q_dict["head_pitch"] += dpitch_head

        # Wrist roll (direct control)
        droll_left = action.pop("left_arm_ee_delta.roll")
        droll_right = action.pop("right_arm_ee_delta.roll")
        if droll_left is not None and q_dict.get("left_arm_wrist_roll") is not None:
            q_dict["left_arm_wrist_roll"] -= droll_left
        if droll_right is not None and q_dict.get("right_arm_wrist_roll") is not None:
            q_dict["right_arm_wrist_roll"] -= droll_right
        
        # Wrist yaw (direct control)
        dyaw_left = action.pop("left_arm_ee_delta.yaw", None)
        dyaw_right = action.pop("right_arm_ee_delta.yaw", None)
        if dyaw_left is not None and q_dict.get("left_arm_wrist_yaw") is not None:
            q_dict["left_arm_wrist_yaw"] -= dyaw_left
        if dyaw_right is not None and q_dict.get("right_arm_wrist_yaw") is not None:
            q_dict["right_arm_wrist_yaw"] -= dyaw_right
        
        # Wrist position (xyz in robot frame)
        # dy directly controls shoulder_pan
        dy_left = action.pop("left_arm_ee_delta.y")
        dy_right = action.pop("right_arm_ee_delta.y")
        if dy_left is not None and q_dict.get("left_arm_shoulder_pan") is not None:
            q_dict["left_arm_shoulder_pan"] -= dy_left
        if dy_right is not None and q_dict.get("right_arm_shoulder_pan") is not None:
            q_dict["right_arm_shoulder_pan"] -= dy_right
        
        # Initialize target_pitch and target_xz
        if self._target_pitch_left is None:
            self._target_pitch_left = 0.0
        if self._target_pitch_right is None:
            self._target_pitch_right = 0.0
        if (
            self._target_xz_left is None
            and q_dict.get("left_arm_shoulder_lift") is not None
            and q_dict.get("left_arm_elbow_flex") is not None
        ):
            jnt2_raw = self.robot._unnormalize("left_arm_shoulder_lift", q_dict["left_arm_shoulder_lift"])
            jnt3_raw = self.robot._unnormalize("left_arm_elbow_flex", q_dict["left_arm_elbow_flex"])
            jnt2 = self.robot._raw_to_deg("left_arm_shoulder_lift", jnt2_raw)
            jnt3 = self.robot._raw_to_deg("left_arm_elbow_flex", jnt3_raw)
            self._target_xz_left = self.kinematics_left.forward_kinematics(jnt2, jnt3)
        if (
            self._target_xz_right is None
            and q_dict.get("right_arm_shoulder_lift") is not None
            and q_dict.get("right_arm_elbow_flex") is not None
        ):
            jnt2_raw = self.robot._unnormalize("right_arm_shoulder_lift", q_dict["right_arm_shoulder_lift"])
            jnt3_raw = self.robot._unnormalize("right_arm_elbow_flex", q_dict["right_arm_elbow_flex"])
            jnt2 = self.robot._raw_to_deg("right_arm_shoulder_lift", jnt2_raw)
            jnt3 = self.robot._raw_to_deg("right_arm_elbow_flex", jnt3_raw)
            self._target_xz_right = self.kinematics_right.forward_kinematics(jnt2, jnt3)
        
        # Wrist pitch
        dpitch_left = action.pop("left_arm_ee_delta.pitch")
        dpitch_right = action.pop("right_arm_ee_delta.pitch")
        if dpitch_left is not None: self._target_pitch_left -= dpitch_left
        if dpitch_right is not None: self._target_pitch_right -= dpitch_right

        # Solve IK to get shoulder_lift and elbow_flex
        # Left arm
        dx_left = action.pop("left_arm_ee_delta.x")
        dz_left = action.pop("left_arm_ee_delta.z")
        if (
            dx_left is not None
            and dz_left is not None
            and q_dict.get("left_arm_shoulder_lift") is not None
            and q_dict.get("left_arm_elbow_flex") is not None
        ):
            self._target_xz_left[0] += dx_left
            self._target_xz_left[1] += dz_left
            self._target_xz_left = self.kinematics_left.apply_workspace_bound(*self._target_xz_left)[:2]
            jnt2, jnt3 = self.kinematics_left.inverse_kinematics(*self._target_xz_left)
            jnt2_raw = self.robot._deg_to_raw("left_arm_shoulder_lift", jnt2)
            jnt3_raw = self.robot._deg_to_raw("left_arm_elbow_flex", jnt3)
            jnt2_norm = self.robot._normalize("left_arm_shoulder_lift", jnt2_raw)
            jnt3_norm = self.robot._normalize("left_arm_elbow_flex", jnt3_raw)
            q_dict["left_arm_shoulder_lift"] = jnt2_norm
            q_dict["left_arm_elbow_flex"] = jnt3_norm

            # Wrist flex
            if (
                q_dict.get("left_arm_wrist_flex") is not None
                and self._target_pitch_left is not None
            ):
                jnt4 = -jnt2 - jnt3 - self._target_pitch_left
                jnt4_raw = self.robot._deg_to_raw("left_arm_wrist_flex", jnt4)
                jnt4_norm = self.robot._normalize("left_arm_wrist_flex", jnt4_raw)
                q_dict["left_arm_wrist_flex"] = jnt4_norm

        # Right arm
        dx_right = action.pop("right_arm_ee_delta.x")
        dz_right = action.pop("right_arm_ee_delta.z")
        if (
            dx_right is not None
            and dz_right is not None
            and q_dict.get("right_arm_shoulder_lift") is not None
            and q_dict.get("right_arm_elbow_flex") is not None
        ):
            self._target_xz_right[0] += dx_right
            self._target_xz_right[1] += dz_right
            self._target_xz_right = self.kinematics_right.apply_workspace_bound(*self._target_xz_right)[:2]
            jnt2, jnt3 = self.kinematics_right.inverse_kinematics(*self._target_xz_right)
            jnt2_raw = self.robot._deg_to_raw("right_arm_shoulder_lift", jnt2)
            jnt3_raw = self.robot._deg_to_raw("right_arm_elbow_flex", jnt3)
            jnt2_norm = self.robot._normalize("right_arm_shoulder_lift", jnt2_raw)
            jnt3_norm = self.robot._normalize("right_arm_elbow_flex", jnt3_raw)
            q_dict["right_arm_shoulder_lift"] = jnt2_norm
            q_dict["right_arm_elbow_flex"] = jnt3_norm
        
            # Wrist flex
            if (
                q_dict.get("right_arm_wrist_flex") is not None
                and self._target_pitch_right is not None
            ):
                jnt4 = -jnt2 - jnt3 - self._target_pitch_right
                jnt4_raw = self.robot._deg_to_raw("right_arm_wrist_flex", jnt4)
                jnt4_norm = self.robot._normalize("right_arm_wrist_flex", jnt4_raw)
                q_dict["right_arm_wrist_flex"] = jnt4_norm
            
        # Gripper
        gripper_left = action.pop("left_arm_gripper.pos")
        gripper_right = action.pop("right_arm_gripper.pos")
        if gripper_left is not None and q_dict.get("left_arm_gripper") is not None:
            q_dict["left_arm_gripper"] += gripper_left
        if gripper_right is not None and q_dict.get("right_arm_gripper") is not None:
            q_dict["right_arm_gripper"] += gripper_right
        
        # New action
        action.update({f"{k}.pos":v for k, v in q_dict.items()})
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        no_left_wrist_yaw = 'left_arm_ee_delta.yaw' not in features[PipelineFeatureType.ACTION]
        no_right_wrist_yaw = 'right_arm_ee_delta.yaw' not in features[PipelineFeatureType.ACTION]
        
        for feat in [
            "left_arm_ee_delta.x",
            "left_arm_ee_delta.y",
            "left_arm_ee_delta.z",
            "left_arm_ee_delta.roll",
            "left_arm_ee_delta.pitch",
            "left_arm_ee_delta.yaw",
            "left_arm_gripper.pos",
            "right_arm_ee_delta.x",
            "right_arm_ee_delta.y",
            "right_arm_ee_delta.z",
            "right_arm_ee_delta.roll",
            "right_arm_ee_delta.pitch",
            "right_arm_ee_delta.yaw",
            "right_arm_gripper.pos",
            "head.yaw",
            "head.pitch",
        ]:
            features[PipelineFeatureType.ACTION].pop(feat, None)
        
        for feat in [
            "left_arm_shoulder_pan",
            "left_arm_shoulder_lift",
            "left_arm_elbow_flex",
            "left_arm_wrist_flex",
            "left_arm_wrist_yaw",
            "left_arm_wrist_roll",
            "left_arm_gripper",
            "right_arm_shoulder_pan",
            "right_arm_shoulder_lift",
            "right_arm_elbow_flex",
            "right_arm_wrist_flex",
            "right_arm_wrist_yaw",
            "right_arm_wrist_roll",
            "right_arm_gripper",
            "head_yaw",
            "head_pitch",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        if no_left_wrist_yaw:
            features[PipelineFeatureType.ACTION].pop("left_arm_wrist_yaw", None)
        if no_right_wrist_yaw:
            features[PipelineFeatureType.ACTION].pop("right_arm_wrist_yaw", None)
        return features


@ProcessorStepRegistry.register("base_joint_action")
@dataclass
class BaseJointAction(RobotActionProcessorStep):
    """
    Converts the generic base commands to the actual base actions (e.g., 'x.vel', 'y.vel' and 'theta.vel').

    Attributes:
        robot: The robot instance.
    """
    robot: XLeRobot

    def action(self, action: RobotAction) -> RobotAction:
        base_action = action.pop("base_action")
        pressed_keys = set()
        for act in base_action:
            pressed_keys.add(self.robot.teleop_keys[act])
        action.update(self.robot._from_keyboard_to_base_action(list(pressed_keys)))
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("base_action", None)
        for feat in [
            "x.vel",
            "y.vel",
            "theta.vel",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("joint_clip_norm_value")
@dataclass
class JointClipNormValue(RobotActionProcessorStep):
    robot: XLeRobot

    def action(self, action: RobotAction) -> RobotAction:
        for k, v in action.items():
            if isinstance(k, str) and k.endswith(".pos"):
                action[k] = self.robot._clip_norm_value(k.removesuffix(".pos"), v)
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ema_joint_action")
@dataclass
class EMAJointAction(RobotActionProcessorStep):
    """
    Applies Exponential Moving Average (EMA) to the joint actions to smooth the control signals.
    
    Attributes:
        fps: The frame rate of the robot.
        ema_alpha: The alpha value for the EMA.
        base_speed_up_time: The time to speed up the base speed.
        base_speed_down_time: The time to slow down the base speed.
    """
    fps: int = 30 # Hz
    ema_alpha: float = 0.9
    base_speed_up_time: float = 3.0
    base_speed_down_time: float = 0.5

    def __post_init__(self):
        self.reset()
    
    def reset(self):
        self._prev_action = None
        self._speed_up_cnt = {}
        self._speed_down_cnt = {}

    def action(self, action: RobotAction) -> RobotAction:
        curr_action = action.copy()
        if self._prev_action is None:
            self._prev_action = curr_action
        
        calc_alpha = lambda a0, t, hz: 1.0 - a0 ** (1 / (t * hz))

        new_action = {}
        for k, v in curr_action.items():
            if not k.endswith('.vel'):
                new_action[k] = v * self.ema_alpha + self._prev_action[k] * (1 - self.ema_alpha)
            else: # Base speed EMA
                # Speed up is triggered when the robot moves from rest
                # And continues until the number of steps is completed
                if (abs(self._prev_action[k]) < 1e-3 and abs(v) > 1e-3) or self._speed_up_cnt.get(k, 0) > 0:
                    if k not in self._speed_up_cnt or self._speed_up_cnt[k] == 0:
                        self._speed_up_cnt[k] = int(self.fps * self.base_speed_up_time)
                    self._speed_up_cnt[k] -= 1
                    ema_alpha = calc_alpha(0.01, self.base_speed_up_time, self.fps)
                # Speed down is triggered when the robot is required to stop from motion
                # And continues until the number of steps is completed
                elif (abs(self._prev_action[k]) > 1e-3 and abs(v) < 1e-3) or self._speed_down_cnt.get(k, 0) > 0:
                    if k not in self._speed_down_cnt or self._speed_down_cnt[k] == 0:
                        self._speed_down_cnt[k] = int(self.fps * self.base_speed_down_time)
                    self._speed_down_cnt[k] -= 1
                    ema_alpha = calc_alpha(0.01, self.base_speed_down_time, self.fps)
                else:
                    ema_alpha = self.ema_alpha
                new_action[k] = v * ema_alpha + self._prev_action[k] * (1 - ema_alpha)
        self._prev_action.update(new_action)
        return new_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("safe_goal_position")
@dataclass
class SafeGoalPosition(RobotActionProcessorStep):
    """
    Ensures the goal position is safe by clamping the relative target magnitude.

    Attributes:
        max_relative_target: The maximum relative target magnitude for each motor.
        motor_names: The names of the motors to control.
    """
    max_relative_target: float | dict[str, float]
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing safe goal position")

        present_pos = {
            k.removesuffix(".pos"):float(v)
            for k, v in observation.items()
            if isinstance(k, str)
            and k.endswith(".pos")
            and k.removesuffix(".pos") in self.motor_names
        }

        if present_pos is None:
            raise ValueError("Joints observation is require for computing safe goal position")
        
        goal_present_pos = {
            k: (goal_pos, present_pos[k.removesuffix(".pos")])
            for k, goal_pos in action.items()
            if k.removesuffix(".pos") in self.motor_names
        }
        safe_pos = ensure_safe_goal_position(goal_present_pos, self.max_relative_target)
        action.update(safe_pos)
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("round_off_action")
@dataclass
class RoundOffAction(RobotActionProcessorStep):
    """
    Round off the action values to avoid shakiness.

    Attributes:
        decimal_places: The number of decimal places to round off the action values.
        min_abs_value: The minimum absolute value below which the action value is set to 0.
        filters: A list of suffixes to filter the action keys (e.g., ".pos", ".vel").
    """
    decimal_places: int = 2
    min_abs_value: float = 0.01
    filters: list[str] = field(default_factory=lambda: [".pos", ".vel"])

    def action(self, action: RobotAction) -> RobotAction:
        for k, v in action.items():
            if isinstance(k, str) and any(k.endswith(f) for f in self.filters):
                action[k] = round(float(v), self.decimal_places)
                if abs(action[k]) < self.min_abs_value:
                    action[k] = 0.0
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("log_action")
@dataclass
class LogAction(RobotActionProcessorStep):
    logger: logging.Logger

    def __post_init__(self):
        self._prev_action = None

    def reset(self):
        self._prev_action = None

    def action(self, action: RobotAction) -> RobotAction:
        is_first = False
        curr_action = action.copy()
        if self._prev_action is None:
            self._prev_action = curr_action
            is_first = True
        
        log_action = {}
        for k, v in curr_action.items():
            if v != self._prev_action[k] or is_first:
                log_action[k] = round(float(v), 3)
        self._prev_action = curr_action

        if log_action:
            self.logger.info(f"Target action:\n{pformat(log_action, indent=4)}")
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
