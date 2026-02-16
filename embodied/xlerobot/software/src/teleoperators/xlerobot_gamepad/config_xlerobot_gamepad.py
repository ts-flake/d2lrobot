#!/usr/bin/env python

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("xlerobot_gamepad")
@dataclass
class XLeRobotGamepadConfig(TeleoperatorConfig):
    fps: int = 30 # Hz
    console_level: str = 'info' # Logging level
    stepsize: float | dict[str, float] = field(default_factory=lambda: {'ang': 1.0, 'pos': 0.001, 'gripper': 5.0}) # Stepsize
    enable_left_arm_control: bool = True # Enable left arm control; disable to use only right arm
    enable_head_control: bool = False # Enable head control;
    record_dataset: bool = False # Use the left thumbstick for record control

    # Zero position offset in degrees
    zero_position_offset: dict[str, float] = field(default_factory=lambda: {
        'left_arm_shoulder_lift': -80,
        'left_arm_elbow_flex': 80,
        'left_arm_wrist_roll': -50,
        'right_arm_shoulder_lift': -80,
        'right_arm_elbow_flex': 80,
        'right_arm_wrist_roll': -50,
        'head_pitch': 55
    })

    stepsize_factors: dict[str, float] = field(default_factory=lambda: {
        'arm': {
            'roll': 3,
            'pitch': 0.5,
            'yaw': 3,
        },
        'head': {
            'pitch': 3
        }
    })
    