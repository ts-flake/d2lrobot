# !/usr/bin/env python

import time
import logging
import draccus
from dataclasses import dataclass

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.rr_kinematics import RRKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.xlerobot_yaw import XLeRobotYaw, XLeRobotYawConfig
from lerobot.robots.xlerobot.robot_action_processor import (
    AnalyticalInverseKinematicsDeltaToJoints,
    BaseJointAction,
    JointClipNormValue,
    SafeGoalPosition,
    EMAJointAction,
    RoundOffAction,
    LogAction
)
from lerobot.processor import make_default_processors
from lerobot.robots.xlerobot.utils.action_utils import move_robot_to_position, move_robot_to_zero_position
from lerobot.teleoperators.xlerobot_yaw_vr import XLeRobotYawVR, XLeRobotYawVRConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record_loop
from lerobot.utils.utils import log_say
from lerobot.utils.color_logger import init_color_logging

@dataclass
class Config:
    robot: XLeRobotYawConfig
    teleop: XLeRobotYawVRConfig
    dataset: DatasetRecordConfig
    display_data: bool = False

@draccus.wrap()
def main(cfg: Config):
    # Initialize logging
    init_color_logging()
    logger = logging.getLogger(__name__)

    # Initialize the robot and teleoperator
    robot_config = cfg.robot
    teleop_config = cfg.teleop
    dataset_config = cfg.dataset

    FPS = teleop_config.fps
    ZERO_POSITION_OFFSET = teleop_config.zero_position_offset
    NUM_EPISODES = dataset_config.num_episodes
    EPISODE_TIME_SEC = dataset_config.episode_time_s
    RESET_TIME_SEC = dataset_config.reset_time_s
    TASK_DESCRIPTION = dataset_config.single_task
    HF_REPO_ID = dataset_config.repo_id
    DISPLAY_DATA = cfg.display_data

    # Initialize the robot and teleoperator
    robot = XLeRobotYaw(robot_config)
    teleop_device = XLeRobotYawVR(teleop_config)

    # Build pipeline to convert vr action to joint action
    vr_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            AnalyticalInverseKinematicsDeltaToJoints(
                kinematics_left=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
                kinematics_right=RRKinematics(use_degrees=True, offsets=[90, 90], reversed=[True, False]),
                robot=robot,
                motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors,
            ),
            BaseJointAction(robot=robot),
            JointClipNormValue(robot=robot),
            SafeGoalPosition(
                max_relative_target=robot.config.max_relative_target,
                motor_names=robot.left_arm_motors + robot.right_arm_motors + robot.head_motors
            ),
            EMAJointAction(fps=FPS),
            RoundOffAction(),
            LogAction(logger),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    _,robot_action_processor, robot_observation_processor = make_default_processors()

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=combine_feature_dicts(
            # Run the feature contract of the pipelines
            # This tells you how the features would look like after the pipeline steps
            aggregate_pipeline_dataset_features(
                pipeline=robot_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=dataset_config.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=dataset_config.video,
            ),
        ),
        robot_type=robot.name,
        use_videos=dataset_config.video,
        image_writer_threads=dataset_config.num_image_writer_threads_per_camera,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer
    init_rerun(session_name="xlerobot_yaw_teleop_vr")

    if not robot.is_connected:
        logger.error("❌ Robot is not connected!")
        exit(1)

    while not teleop_device.is_connected:
        busy_wait(0.1) # Wait for VR to connect

    while not teleop_device.is_calibrated:
        busy_wait(0.1) # Wait for VR to calibrate

    # Record the initial position
    initial_obs = robot.get_observation()
    initial_pos = {k.removesuffix(".pos"): v for k, v in initial_obs.items() if k.endswith(".pos")}

    # Move to zero position
    move_robot_to_zero_position(robot, fps=FPS, duration=5.0, end_offset=ZERO_POSITION_OFFSET)

    # :----- Main teleop loop -----:
    logger.info("Starting teleop loop. Move your VR controllers to teleoperate the robot...")
    events = teleop_device.events

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"] and not events["exit_teleop"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop_device,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=DISPLAY_DATA,
            teleop_action_processor=vr_to_robot_joints_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Reset the environment without recording if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            log_say("Resetting to zero position...")
            move_robot_to_zero_position(robot, fps=FPS, duration=5.0, end_offset=ZERO_POSITION_OFFSET)
            vr_to_robot_joints_processor.reset()
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop_device,
                dataset=None, # No recording
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=DISPLAY_DATA,
                teleop_action_processor=vr_to_robot_joints_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            log_say("Resetting to zero position...")
            move_robot_to_zero_position(robot, fps=FPS, duration=5.0, end_offset=ZERO_POSITION_OFFSET)
            vr_to_robot_joints_processor.reset()
            busy_wait(5.0)
            dataset.clear_episode_buffer()
            continue

        # Save episode
        dataset.save_episode()
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    log_say("Exiting the teleop loop, returning to initial position...")
    move_robot_to_position(robot, end_pos=initial_pos, fps=FPS, duration=5.0)

    robot.disconnect()
    teleop_device.disconnect()

    dataset.finalize()
    if dataset_config.push_to_hub:
        dataset.push_to_hub()

if __name__ == "__main__":
    main()