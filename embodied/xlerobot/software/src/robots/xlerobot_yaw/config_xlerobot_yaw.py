#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from ..config import RobotConfig


def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, Cv2Rotation
    return {
        # "left_wrist": OpenCVCameraConfig(
        #     index_or_path="/dev/video7", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        # "right_wrist": OpenCVCameraConfig(
        #     index_or_path="/dev/video6", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),  
        # "head": OpenCVCameraConfig(
        #     index_or_path="/dev/video4", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),                     
        # "head": RealSenseCameraConfig(
        #     serial_number_or_name="125322060037",  # Replace with camera SN
        #     fps=30,
        #     width=1280,
        #     height=720,
        #     color_mode=ColorMode.BGR, # Request BGR output
        #     rotation=Cv2Rotation.NO_ROTATION,
        #     use_depth=True
        # ),
    }


@RobotConfig.register_subclass("xlerobot_yaw")
@dataclass
class XLeRobotYawConfig(RobotConfig):
    
    port1: str = "/dev/ttyACM0"  # port to connect to the bus (so101_yaw + head camera)
    port2: str = "/dev/ttyACM1"  # port to connect to the bus (so101_yaw + 3 omniwheels)
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: float | None = 10.0

    # Shift the `ensure_safe_goal_position` to pipeline and skip the redundant check in the `send_action` function
    skip_goal_position_check: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # Base control keyboard keys
    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # movement
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            # speed control
            "speed_up": "n",
            "speed_down": "m",
            # quit teleop
            "quit": "b",
        }
    )

    # Base control speed levels
    base_speed_levels: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
    )
