import numpy as np
from enum import Enum
from typing import Callable, Any


class TrajectoryMode(Enum):
    LINEAR = 0
    MIN_JERK = 1

def get_trajectory_fn(mode: TrajectoryMode):
    if mode == TrajectoryMode.LINEAR:
        return linear_traj_fn
    elif mode == TrajectoryMode.MIN_JERK:
        return minimal_jerk_traj_fn
    else:
        raise ValueError(f"Unsupported trajectory mode: {mode}")

def _check_shape(array, shape):
    assert array.shape == shape, f"Expected array of shape {shape}, got {array.shape}"

def linear_traj_fn(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    t0: float = 0.0,
    duration: float = 1.0,
    **kwargs: Any
) -> Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Linear interpolation between two d-dimensional points.
    """
    assert duration > 0.0, "duration must be positive"
    start_pos = np.asarray(start_pos)
    end_pos = np.asarray(end_pos)
    _shape = (start_pos.shape[0],)
    for array in [start_pos, end_pos]:
        _check_shape(array, _shape)

    def _sampler(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # pos, vel, acc
        t = np.clip(t - t0, 0.0, duration)
        tau = t / duration
        delta_pos = end_pos - start_pos
        pos = start_pos + delta_pos * tau
        vel = delta_pos / duration
        acc = np.zeros_like(pos)
        return pos, vel, acc
    return _sampler

def minimal_jerk_traj_fn(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_vel: np.ndarray | None = None,
    end_vel: np.ndarray | None = None,
    start_acc: np.ndarray | None = None,
    end_acc: np.ndarray | None = None,
    t0: float = 0.0,
    duration: float = 1.0,
    **kwargs: Any
    ) -> Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate a minimal jerk trajectory between two d-dimensional points.

    Args:
        start_pos: Start position. Shape (d,)
        end_pos: End position. Shape (d,)
        start_vel: Start velocity. Shape (d,)
        end_vel: End velocity. Shape (d,)
        start_acc: Start acceleration. Shape (d,)
        end_acc: End acceleration. Shape (d,)
        t0: Start time. Scalar
        duration: Duration of the trajectory in seconds

    Returns:
        sampler: Sampler function that takes a time t and returns the position, velocity, and acceleration.
    """
    assert duration > 0.0, "duration must be positive"
    start_pos = np.asarray(start_pos)
    end_pos = np.asarray(end_pos)
    _shape = (start_pos.shape[0],)

    zero_vel_acc = start_vel is None and start_acc is None and end_vel is None and end_acc is None
    start_vel = np.zeros_like(start_pos) if start_vel is None else np.asarray(start_vel)
    start_acc = np.zeros_like(start_pos) if start_acc is None else np.asarray(start_acc)
    end_vel = np.zeros_like(start_pos) if end_vel is None else np.asarray(end_vel)
    end_acc = np.zeros_like(start_pos) if end_acc is None else np.asarray(end_acc)
    for array in [start_pos, end_pos, start_vel, start_acc, end_vel, end_acc]:
        _check_shape(array, _shape)
    
    # Compute trajectory
    if zero_vel_acc:
        def _sampler(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # pos, vel, acc
            t = np.clip(t - t0, 0.0, duration)
            tau = t / duration
            pos = start_pos + (end_pos - start_pos)*(10*tau**3 - 15*tau**4 + 6*tau**5)
            vel = (end_pos - start_pos)*(30*tau**2 - 60*tau**3 + 30*tau**4)
            acc = (end_pos - start_pos)*(60*tau - 180*tau**2 + 120*tau**3)
            return pos, vel, acc
    else:
        c0 = start_pos
        c1 = start_vel
        c2 = start_acc / 2.0
        A = np.array([
            [duration**3, duration**4, duration**5],
            [3*duration**2, 4*duration**3, 5*duration**4],
            [6*duration, 12*duration**2, 20*duration**3],
        ])
        B = np.array([
            end_pos - c0 - c1*duration - c2*duration**2,
            end_vel - c1 - 2*c2*duration,
            end_acc - 2*c2,
        ])
        c3, c4, c5 = np.linalg.solve(A, B)
        
        def _sampler(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # pos, vel, acc
            t = np.clip(t - t0, 0.0, duration)
            pos = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
            vel = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
            acc = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
            return pos, vel, acc
    return _sampler

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    start_pos = np.array([0, 0])
    end_pos = np.array([1, 1])
    start_vel = np.array([1, 0])
    end_vel = np.array([0, -1])
    duration = 1.0
    control_freq = 10

    traj_fn = get_trajectory_fn(TrajectoryMode.MIN_JERK)
    sampler = traj_fn(start_pos, end_pos, start_vel=start_vel, end_vel=end_vel, duration=duration)
    times = np.linspace(0, duration, int(duration * control_freq) + 1)
    trajectory = np.array([sampler(t)[0] for t in times])

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(times, trajectory[:,0], 'o-')
    axs[0].set_title('x-time plot')
    axs[1].plot(times, trajectory[:,1], 'o-')
    axs[1].set_title('y-time plot')
    axs[2].plot(trajectory[:,0], trajectory[:,1], 'o-')
    axs[2].set_title('xy-plot')
    plt.suptitle(f'{traj_fn.__name__}', fontsize=16)
    plt.show()
        
