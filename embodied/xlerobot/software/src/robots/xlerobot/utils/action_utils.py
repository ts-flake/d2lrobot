#! /usr/bin/env python3
from typing import Callable, Any
import logging
import numpy as np
import time

from lerobot.robots.robot import Robot
from .control_utils import TrajectoryMode, get_trajectory_fn

logger = logging.getLogger(__name__)

def get_action_sampler(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    duration: float = 1.0,
    traj_mode: TrajectoryMode = TrajectoryMode.MIN_JERK,
    **kwargs: Any
) -> Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    traj_fn = get_trajectory_fn(traj_mode)
    sampler = traj_fn(start_pos, end_pos, duration=duration, **kwargs)
    return sampler

def move_robot_to_position(
    robot: Robot,
    end_pos: dict[str, float],
    fps: int = 30,
    duration: float = 3.0,
    end_offset: dict[str, float] = {}
) -> None:
    logger.info(f"[{robot.name}] Moving to target position in {duration} seconds...")
    
    # Use the current joint positions as the start position
    left_arm_pos = robot.bus1.sync_read("Present_Position", robot.left_arm_motors)
    right_arm_pos = robot.bus2.sync_read("Present_Position", robot.right_arm_motors)
    head_pos = robot.bus1.sync_read("Present_Position", robot.head_motors)
    merged_pos = {**left_arm_pos, **right_arm_pos, **head_pos}

    start_pos = np.array(list(merged_pos.values()))
    end_pos = np.array(
        [
            end_pos.get(k, v) + end_offset.get(k, 0.0)
            for k, v in merged_pos.items()
        ]
    )
    assert len(start_pos) == len(end_pos), "Start and end positions must have the same length"
    sampler = get_action_sampler(
        start_pos=start_pos,
        end_pos=end_pos,
        duration=duration,
        traj_mode=TrajectoryMode.MIN_JERK
    )
    t = 0.0
    dt = 1.0 / fps
    while t < duration:
        pos, _, _ = sampler(t)
        action = {f"{motor}.pos": value for motor, value in zip(merged_pos.keys(), pos)}
        robot.send_action(action)
        time.sleep(dt)
        t += dt

def move_robot_to_zero_position(
    robot: Robot,
    fps: int = 30,
    duration: float = 3.0,
    end_offset: dict[str, float] = {}
) -> None:
    motors = robot.left_arm_motors + robot.right_arm_motors + robot.head_motors
    end_pos = dict.fromkeys(motors, 0.0)
    move_robot_to_position(robot, end_pos=end_pos, fps=fps, duration=duration, end_offset=end_offset)