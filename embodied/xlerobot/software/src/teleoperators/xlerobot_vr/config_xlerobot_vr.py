#!/usr/bin/env python

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("xlerobot_vr")
@dataclass
class XLeRobotVRConfig(TeleoperatorConfig):
    fps: int = 30 # Hz
    console_level: str = 'info' # Logging level
    stepsize: float | dict[str, float] = field(default_factory=lambda: {'ang': 2.0, 'pos': 0.001, 'gripper': 5.0}) # Stepsize
    enable_left_arm_control: bool = True # Enable left arm control with VR controller; disable to use only right arm
    enable_head_control: bool = False # Enable head control with VR headset;
    use_thumbstick_for_head: bool = False # Use left VR controller's thumbstick for head control; False: use VR headset
    grip_to_activate: bool = True # Press the grip button to activate tracking; Reverse: press the grip button to deactivate
    record_dataset: bool = False # Use the left thumbstick for record control, if this is True, the `use_thumbstick_for_head` will be False

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

    # VR to robot dead zones
    vr2robot_dead_zones: dict[str, dict[str, float]] = field(default_factory=lambda: {
        'arm': {
            'roll': 0.5,
            'pitch': 0.4,
            'dx': 0.001,
            'dy': 0.001,
            'dz': 0.001
        },
        'head': {
            'yaw': 0.4,
            'pitch': 0.2
        }
    })

    # VR to robot multiplicative factors
    vr2robot_factors: dict[str, dict[str, float]] = field(default_factory=lambda: {
        'arm': {
            'roll': 100,
            'pitch': 2,
            'dx': 1.5,
            'dy': 1000,
            'dz': 1.5
            },
        'head': {
            'yaw': 200,
            'pitch': 300
        }
    })

    # VR to robot limits
    vr2robot_limits: dict[str, dict[str, float]] = field(default_factory=lambda: {
        'arm': {'pos': 0.005, 'ang': 5.0}, 'head': {'pos': 0.005, 'ang': 3.0}
    })

    def __post_init__(self):
        if self.record_dataset:
            self.use_thumbstick_for_head = False

    