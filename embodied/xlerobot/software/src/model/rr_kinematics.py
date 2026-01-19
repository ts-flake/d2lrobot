import logging

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class RRKinematics:
    """
    A simple 2-link RR robot model. The class solves the inverse kinematics analytically.
    All public methods use degrees for input/output if `use_degrees` is True.

    The joint angles are defined as follows:
    - jnt1: range [-pi, pi]
    - jnt2: range [0, pi]
      
    \...jnt2↴
     O⎯⎯⎯L2⎯⎯⎯ee
      \   
       L1  y
        \  ▲
         \.⏐..jnt1↰
           O⎯⎯⎯⎯⎯▶x
    """
    def __init__(
        self,
        l1 : float = 0.1159,
        l2 : float = 0.1350,
        use_degrees : bool = True,
        offsets: list[float] = [0.0, 0.0],
        reversed: list[bool] = [False, False]
    ):
        self._l1 = l1  # length of the first link (upper arm)
        self._l2 = l2  # length of the second link (lower arm)
        self.use_degrees = use_degrees  # whether to use degrees for input/output
        self.offsets = np.deg2rad(offsets) if use_degrees else offsets # offsets are defined when `reversed` is False
        self.reversed = reversed # whether to reverse the direction of joint rotation

    @property
    def l1(self) -> float:
        return self._l1

    @property
    def l2(self) -> float:
        return self._l2

    def _convert_to_default_joint_angles(self, jnt1 : float, jnt2 : float) -> tuple[float, float]:
        # Inputs in radians
        jnt1 = self.offsets[0] + (1 if not self.reversed[0] else -1) * jnt1
        jnt2 = self.offsets[1] + (1 if not self.reversed[1] else -1) * jnt2
        if jnt2 < 0 or jnt2 > np.pi:
            jnt2_old = jnt2
            jnt2 = np.clip(jnt2, 0, np.pi)
            logger.warning(f"Joint 2 angle is out of range [0, pi]. Clamped from {jnt2_old:.2f} to {jnt2:.2f}.")
        return jnt1, jnt2

    def _convert_to_actual_joint_angles(self, jnt1 : float, jnt2 : float) -> tuple[float, float]:
        # !Inputs in radians
        jnt1 = (1 if not self.reversed[0] else -1) * (jnt1 - self.offsets[0])
        jnt2 = (1 if not self.reversed[1] else -1) * (jnt2 - self.offsets[1])
        return jnt1, jnt2
    
    def set_arm_lengths(self, l1 : float, l2 : float):
        self._l1 = l1
        self._l2 = l2

    def forward_kinematics(self, jnt1 : float, jnt2 : float) -> np.ndarray:
        jnt1 = np.deg2rad(jnt1) if self.use_degrees else jnt1
        jnt2 = np.deg2rad(jnt2) if self.use_degrees else jnt2

        jnt1, jnt2 = self._convert_to_default_joint_angles(jnt1, jnt2)

        x = self.l2 * np.cos(jnt1 - jnt2) + self.l1 * np.cos(jnt1)
        y = self.l2 * np.sin(jnt1 - jnt2) + self.l1 * np.sin(jnt1)
        return np.array([x, y])
    
    def apply_workspace_bound(self, x: float, y: float) -> np.ndarray:
        """
        Apply the workspace bound to the target point.
        Return the scaled target point and the scaled radius.
        """
        max_r = self.l1 + self.l2
        min_r = abs(self.l1- self.l2)
        r = np.hypot(x, y)
        if r > max_r:
            scale_factor = max_r / r
            x *= scale_factor
            y *= scale_factor
            r = max_r
        if r < min_r:
            scale_factor = min_r / r
            x *= scale_factor
            y *= scale_factor
            r = min_r
        return np.array([x, y, r])
    
    def inverse_kinematics(self, x : float, y : float) -> np.ndarray:
        # Bound the target point to the workspace
        x, y, r = self.apply_workspace_bound(x, y)

        # IK to find the joint angles
        _val = (r**2 - self.l1**2 - self.l2**2) / (-2 * self.l1 * self.l2)
        _val = max(-1, min(1, _val)) # ensure the value is within the range of [-1, 1]
        gamma = np.arccos(_val)
        alpha = np.arctan2(y, x)
        beta = np.arctan2(self.l2 * np.sin(gamma), self.l1  - self.l2 * np.cos(gamma))
        jnt1 = alpha + beta
        jnt2 = np.pi - gamma

        jnt1, jnt2 = self._convert_to_actual_joint_angles(jnt1, jnt2) # in radians
        jnt1 = np.rad2deg(jnt1) if self.use_degrees else jnt1
        jnt2 = np.rad2deg(jnt2) if self.use_degrees else jnt2
        return jnt1, jnt2

    def draw_links(self, ax : plt.Axes, jnt1 : float, jnt2 : float, alpha : float = 1):
        jnt1 = np.deg2rad(jnt1) if self.use_degrees else jnt1
        jnt2 = np.deg2rad(jnt2) if self.use_degrees else jnt2
        _jnt1, _jnt2 = self._convert_to_default_joint_angles(jnt1, jnt2)

        x1, y1 = self.l1 * np.cos(_jnt1), self.l1 * np.sin(_jnt1)
        x2, y2 = self.forward_kinematics(jnt1, jnt2)

        xs = np.array([0, x1, x2])
        ys = np.array([0, y1, y2])
        ax.plot(xs, ys, 'k-', linewidth=6, alpha=alpha)
        ax.plot(xs[0], ys[0], 'ro', markersize=12)
        ax.plot(xs[1], ys[1], 'ro', markersize=12, alpha=alpha)
        ax.plot(xs[2], ys[2], 'bs', markersize=12, alpha=alpha)


if __name__ == "__main__":
    # Define 2-link robot
    l1, l2 = 0.1159, 0.1350
    rr = RRKinematics(l1=l1, l2=l2, use_degrees=False, offsets=[np.pi/2, np.pi/2], reversed=[True, False])
    
    # Setup plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    N = 10
    alphas = np.linspace(0.2, 1.0, N) # past states are faded out

    # FK
    jnt1s = np.linspace(0, np.pi, N)
    jnt2s = np.linspace(0, np.pi/2, N)
    for i in range(N):
        rr.draw_links(axs[0], jnt1s[i], jnt2s[i], alpha=alphas[i])
    axs[0].axis('equal')
    axs[0].set_title(f'RR Kinematics', fontsize=16)

    # IK
    start_pos = [l2, l1]
    end_pos = [l2+0.2, l1]
    traj = np.linspace(start_pos, end_pos, N).T
    for i in range(N):
        jnt1, jnt2 = rr.inverse_kinematics(traj[0, i], traj[1, i])
        print(f'jnt1: {jnt1}, jnt2: {jnt2}')
        rr.draw_links(axs[1], jnt1, jnt2, alpha=alphas[i])
    axs[1].plot(traj[0, :], traj[1, :], 'k--', linewidth=3)
    axs[1].axis('equal')
    axs[1].set_title(f'RR IK Trajectory', fontsize=16)
    plt.show()