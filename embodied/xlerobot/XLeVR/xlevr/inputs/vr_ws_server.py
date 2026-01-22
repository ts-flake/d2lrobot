"""
VR WebSocket server for receiving controller data from web browsers.
Adapted from the original vr_robot_teleop.py script.

Original code: telegrip
Modified for XLeRobot:
- add y-axis rotation
- use delta position and rotation in local frame

Important functions to look at if you want to modify the code:
- process_controller_data
- process_single_controller
"""

import asyncio
from dataclasses import dataclass
import json
import ssl
import websockets
from pprint import pformat
import logging
from typing import Dict, Optional, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.utils.utils import move_cursor_up
from .base import BaseInputProvider, ControlGoal, ControlMode
from ..config import XLeVRConfig

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('vr_ws_server')

@dataclass
class VRControllerState:
    """State tracking for a VR controller."""
    
    hand: str
    grip_active: bool = False
    trigger_active: bool = False
    origin_position: np.ndarray | None = None # [x, y, z] in meters
    origin_quaternion: np.ndarray | None = None # [x, y, z, w]
    prev_position: np.ndarray | None = None
    prev_quaternion: np.ndarray | None = None
    curr_position: np.ndarray | None = None
    curr_quaternion: np.ndarray | None = None
    delta_position: np.ndarray | None = None
    delta_quaternion: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        return self.origin_position is not None and self.origin_quaternion is not None
    
    def calibrate_from_dict(self, position: Dict, quaternion: Dict):
        position = np.array([position['x'], position['y'], position['z']])
        quaternion = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
        self.calibrate(position, quaternion)
    
    def calibrate(self, position: np.ndarray, quaternion: np.ndarray):
        self.origin_position = self.prev_position = self.curr_position = position
        self.origin_quaternion = self.prev_quaternion = self.curr_quaternion = quaternion
        logger.info(f"🎯 [{self.hand}] controller calibrated:\n{pformat(dict(position=position, quaternion=quaternion), indent=4)}")
    
    @staticmethod
    def as_transformation_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        xmat = np.eye(4)
        xmat[:3, :3] = R.from_quat(quaternion, scalar_first=False).as_matrix()
        xmat[:3, 3] = position
        return xmat
    
    @staticmethod
    def as_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
        return R.from_quat(quaternion, scalar_first=False).as_matrix()
    
    @staticmethod
    def as_rotvec(quaternion: np.ndarray, degrees: bool = True) -> np.ndarray:
        rotvec = R.from_quat(quaternion, scalar_first=False).as_rotvec()
        return rotvec * (180.0 / np.pi) if degrees else rotvec
    
    def reset(self, keep_grip: bool = False, keep_trigger: bool = True):
        """Reset controller state."""
        if not keep_grip:
            self.grip_active = False
        if not keep_trigger:
            self.trigger_active = False
        self.origin_position = None
        self.origin_quaternion = None
        self.prev_position = None
        self.prev_quaternion = None
        self.curr_position = None
        self.curr_quaternion = None
        self.delta_position = None
        self.delta_quaternion = None

def euler_to_quat(euler_deg: np.ndarray) -> np.ndarray:
    return R.from_euler('xyz', euler_deg, degrees=True).as_quat(scalar_first=False)

def update_controller_state(controller: VRControllerState, position: np.ndarray, quaternion: np.ndarray) -> None:
    controller.prev_position = controller.curr_position
    controller.prev_quaternion = controller.curr_quaternion
    controller.curr_position = position.copy()
    controller.curr_quaternion = quaternion.copy()

def calculate_delta_state(controller: VRControllerState, in_local: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the delta pose (position and quaternion) of the controller.
    If in_local is True, the delta pose is in the local coordinate of the controller (origin frame).
    If in_local is False, the delta pose is in the global coordinate.
    """
    # In world frame
    delta_position = controller.curr_position - controller.prev_position
    prev_rot = controller.as_rotation_matrix(controller.prev_quaternion)
    curr_rot = controller.as_rotation_matrix(controller.curr_quaternion)
    delta_rot = curr_rot @ prev_rot.T
    
    # In local frame
    if in_local:
        wRb = controller.as_rotation_matrix(controller.origin_quaternion)
        delta_position = wRb.T @ delta_position
        delta_rot = wRb.T @ delta_rot @ wRb
    return delta_position, R.from_matrix(delta_rot).as_quat(scalar_first=False)

class VRWebSocketServer(BaseInputProvider):
    """WebSocket server for VR controller input."""
    
    def __init__(self, command_queue: asyncio.Queue, config: XLeVRConfig, print_only: bool = False, debug: bool = False):
        super().__init__(command_queue)
        self.config = config
        self.clients: Set = set()
        self.server = None
        self.print_only = print_only  # New flag for print-only mode
        self.debug = debug  # New flag for logging debug information
        self._need_calibration = False
        
        # Controller states
        self.left_controller = VRControllerState("left")
        self.right_controller = VRControllerState("right")
        self.headset_controller = VRControllerState("headset")
        
    
    def setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Setup SSL context for WebSocket server."""
        # Automatically generate SSL certificates if they don't exist
        if not self.config.ssl_files_exist:
            logger.info("SSL certificates not found for WebSocket server, attempting to generate them...")
            if not self.config.ensure_ssl_certificates():
                logger.error("Failed to generate SSL certificates for WebSocket server")
                return None
        
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(certfile=self.config.certfile, keyfile=self.config.keyfile)
            logger.info("SSL certificate and key loaded successfully for WebSocket server")
            return ssl_context
        except ssl.SSLError as e:
            logger.error(f"Error loading SSL cert/key: {e}")
            return None
    
    async def start(self):
        """Start the WebSocket server."""
        if not self.config.enable_vr:
            logger.info("VR WebSocket server disabled in configuration")
            return
        
        ssl_context = self.setup_ssl()
        if ssl_context is None:
            logger.error("Failed to setup SSL for WebSocket server")
            return
        
        host = self.config.host_ip
        port = self.config.websocket_port
        
        try:
            self.server = await websockets.serve(
                self.websocket_handler, 
                host, 
                port, 
                ssl=ssl_context
            )
            self.is_running = True
            logger.info(f"VR WebSocket server running on wss://{host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def stop(self):
        """Stop the WebSocket server."""
        self.is_running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("VR WebSocket server stopped")
    
    async def websocket_handler(self, websocket, path=None):
        """Handle WebSocket connections from VR controllers."""
        client_address = websocket.remote_address
        logger.info(f"VR client connected: {client_address}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_controller_data(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message}")
                except Exception as e:
                    logger.error(f"Error processing VR data: {e}")
                    # Add more context for debugging
                    logger.error(f"Data that caused error: {data}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
        
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"VR client {client_address} disconnected normally")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"VR client {client_address} disconnected with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with VR client {client_address}: {e}")
        finally:
            self.clients.discard(websocket)
            # Handle grip releases when client disconnects
            await self.handle_grip_release('left')
            await self.handle_grip_release('right')
            logger.info(f"VR client {client_address} cleanup complete")
    
    def print_headset_and_controllers_data(self, data: Dict):
        # 检查是否有摇杆或按钮操作，只在有操作时打印
        has_thumbstick_or_button_activity = False
        thumbstick_info = []
        button_info = []
        
        # 检查左右手柄的摇杆和按钮状态
        for hand in ['leftController', 'rightController']:
            if hand in data:
                controller_data = data[hand]
                hand_name = hand.replace('Controller', '').upper()
                
                # 检查摇杆
                if 'thumbstick' in controller_data:
                    thumbstick = controller_data['thumbstick']
                    x = thumbstick.get('x', 0)
                    y = thumbstick.get('y', 0)
                    # 只在摇杆有实际输入时打印（阈值0.1）
                    if abs(x) > 0.1 or abs(y) > 0.1:
                        has_thumbstick_or_button_activity = True
                        thumbstick_info.append(f"[{hand_name}] x={x:.2f}, y={y:.2f}")
                
                # 检查按钮
                if 'buttons' in controller_data:
                    buttons = controller_data['buttons']
                    pressed_buttons = []
                    for button_name, is_pressed in buttons.items():
                        if is_pressed:
                            has_thumbstick_or_button_activity = True
                            pressed_buttons.append(button_name)
                    
                    if pressed_buttons:
                        button_info.append(f"[{hand_name}] {', '.join(pressed_buttons)}")
        
        # 只在有操作时打印
        if has_thumbstick_or_button_activity:
            print(f"[VR_WS] Activity detected: Thumbstick - {pformat(thumbstick_info)}; Buttons - {pformat(button_info)}")
        
        if 'headset' in data:
            headset_data = data['headset']
            if headset_data and headset_data.get('position'):
                pos = headset_data['position']
                rot = headset_data.get('rotation', {})
                quat = headset_data.get('quaternion', {})
                
                print(
                    f"[VR_WS] Headset - Position: [{pos.get('x', 0):.3f}, {pos.get('y', 0):.3f}, {pos.get('z', 0):.3f}], "
                    f"Rotation: [{rot.get('x', 0):.1f}, {rot.get('y', 0):.1f}, {rot.get('z', 0):.1f}]"
                )

    def _do_calibration(self, data: Dict):
        def _calibrate(ctrl, data):
            position = data.get('position', None)
            quaternion = data.get('quaternion', None)
            euler = data.get('rotation', None)
            if quaternion is None and euler is not None:
                quaternion_array = euler_to_quat(np.array([euler['x'], euler['y'], euler['z']]))
                quaternion = {
                    'x': quaternion_array[0],
                    'y': quaternion_array[1],
                    'z': quaternion_array[2],
                    'w': quaternion_array[3]
                }
            ctrl.reset()
            ctrl.calibrate_from_dict(position, quaternion)

        all_calibrated = True
        if 'headset' in data:
            _calibrate(self.headset_controller, data['headset'])
            all_calibrated &= self.headset_controller.is_calibrated
        if 'leftController' in data:
            _calibrate(self.left_controller, data['leftController'])
            all_calibrated &= self.left_controller.is_calibrated
        if 'rightController' in data:
            _calibrate(self.right_controller, data['rightController'])
            all_calibrated &= self.right_controller.is_calibrated
        
        if all_calibrated:
            logger.info("✅ VR calibration done successfully")
            self._need_calibration = False
        else:
            logger.warning("❌ VR calibration failed, retry...")
            move_cursor_up(1)
            self._need_calibration = True
    
    @property
    def is_calibrated(self) -> bool:
        return not self._need_calibration
    
    def calibrate(self):
        self._need_calibration = True

    async def process_controller_data(self, data: Dict):
        """Process incoming VR controller data."""

        if self.debug:
            self.print_headset_and_controllers_data(data)
        
        if self._need_calibration and 'rightController' in data and data['rightController'].get('buttons', {}).get('b', False):
            self._do_calibration(data)
            return

        if 'headset' in data and self.headset_controller.is_calibrated:
            await self.process_single_controller(self.headset_controller, data['headset'])
        
        if 'leftController' in data and self.left_controller.is_calibrated:
            await self.process_single_controller(self.left_controller, data['leftController'])
        
        if 'rightController' in data and self.right_controller.is_calibrated:
            await self.process_single_controller(self.right_controller, data['rightController'])
    
    async def process_single_controller(self, controller: VRControllerState, data: Dict):
        """Process data for a single controller."""
        position = data.get('position', {})
        euler = data.get('rotation', {}) # xyz (extrinsic) in degrees
        quaternion = data.get('quaternion', {})  # Get quaternion data directly
        grip_active = data.get('gripActive', False) # Not used anymore
        trigger = data.get('trigger', 0)
        thumbstick = data.get('thumbstick', {})
        buttons = data.get('buttons', {}) # Get buttons data
        hand = controller.hand # Name of the controller
        
        # Convert to numpy arrays
        position = (
            np.array([position['x'], position['y'], position['z']])
            if position and all(k in position for k in ['x', 'y', 'z'])
            else None
        )
        euler = (
            np.array([euler['x'], euler['y'], euler['z']])
            if euler and all(k in euler for k in ['x', 'y', 'z'])
            else None
        )
        quaternion = (
            np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
            if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w'])
            else None
        )
        if quaternion is None and euler is not None:
            quaternion = euler_to_quat(euler)
        
        # Transform from the controller frame to the robot frame
        # Controller frame (seen in VR): x (right), y (up), z (back)
        # Robot frame: x (forward), y (left), z (up)
        bRc = np.array(
            [
                [0, -1, 0],
                [0,  0, 1],
                [-1, 0, 0]
            ]
        ).T

        # Headset control
        if hand == 'headset' and position is not None and quaternion is not None:
            update_controller_state(controller, position, quaternion)
            delta_position, delta_quaternion = calculate_delta_state(controller, in_local=True)
            controller.delta_position = delta_position
            controller.delta_quaternion = delta_quaternion
            delta_rotvec = controller.as_rotvec(delta_quaternion, degrees=True)
            
            # Target position and rotation in robot frame
            target_position = bRc @ delta_position * self.config.vr_to_robot_scale
            target_rpy = bRc @ delta_rotvec
            goal = ControlGoal(
                arm=hand,
                mode=ControlMode.POSITION_CONTROL,
                target_position=target_position,
                wrist_flex_deg=target_rpy[1],
                wrist_yaw_deg=target_rpy[2],
                metadata={
                    "source": "vr_headset",
                    "delta_position": True,
                    "vr_to_robot_scale": self.config.vr_to_robot_scale,
                    "local_frame": True,
                    "controller_state": controller, # unscaled, and in controller frame
                    "controller_to_robot_transform": bRc,
                }
            )
            await self.send_goal(goal)
            return # Skip gripper control for headset

        # Handle trigger for gripper control
        trigger_active = trigger > 0.5
        if trigger_active != controller.trigger_active:
            controller.trigger_active = trigger_active
            
            # # Send gripper control goal - do not specify mode to avoid interfering with position control
            # # Reverse behavior: gripper open by default, closes when trigger pressed
            # gripper_goal = ControlGoal(
            #     arm=hand,
            #     gripper_closed=not trigger_active,  # Inverted: closed when trigger NOT active
            #     metadata={
            #         "source": "vr_trigger",
            #         "trigger": trigger,
            #         "trigger_active": trigger_active,
            #         "thumbstick": thumbstick,
            #         "buttons": buttons
            #     }
            # )
            # await self.send_goal(gripper_goal)
            logger.info(f"🤏 {hand.upper()} gripper {'ACTIVE' if trigger_active else 'INACTIVE'}")
        
        # Always send goal regardless of grip_active,
        # the behavior is controlled by downstream process
        update_controller_state(controller, position, quaternion)
        delta_position, delta_quaternion = calculate_delta_state(controller, in_local=True)
        controller.delta_position = delta_position
        controller.delta_quaternion = delta_quaternion
        delta_rotvec = controller.as_rotvec(delta_quaternion, degrees=True)
        
        # Target position and rotation in robot frame
        target_position = bRc @ delta_position * self.config.vr_to_robot_scale
        target_rpy = bRc @ delta_rotvec
        goal = ControlGoal(
            arm=hand,
            mode=ControlMode.POSITION_CONTROL,
            target_position=target_position,
            wrist_roll_deg=target_rpy[0],
            wrist_flex_deg=target_rpy[1],
            wrist_yaw_deg=target_rpy[2],
            metadata={
                "source": "vr_controller",
                "delta_position": True,
                "vr_to_robot_scale": self.config.vr_to_robot_scale,
                "local_frame": True,
                "controller_state": controller, # unscaled, and in controller frame
                "controller_to_robot_transform": bRc,
                "trigger": trigger,
                "trigger_active": trigger_active,
                "thumbstick": thumbstick,
                "buttons": buttons
            }
        )
        await self.send_goal(goal)
    
    async def handle_grip_release(self, hand: str):
        """Handle grip release for a controller."""
        if hand == 'left':
            controller = self.left_controller
        elif hand == 'right':
            controller = self.right_controller
        else:
            return
        
        if controller.grip_active:
            controller.reset(keep_grip=True)
            
            # Send idle goal to stop arm control
            goal = ControlGoal(
                arm=hand,
                mode=ControlMode.IDLE,
                metadata={
                    "source": "vr_grip_release",
                    "trigger": 0.0,
                    "trigger_active": False,
                    "thumbstick": {},
                    "buttons": {}
                }
            )
            await self.send_goal(goal)
            
            logger.info(f"🔓 {hand.upper()} grip released - arm control stopped")

    async def send_goal(self, goal: ControlGoal):
        """Send a control goal to the command queue or print it if in print-only mode."""
        if self.print_only:
            # Print the ControlGoal in a formatted way
            print(f"\n🎮 ControlGoal:")
            print(f"   Arm: {goal.arm}")
            print(f"   Mode: {goal.mode}")
            if goal.target_position is not None:
                print(f"   Target Position: [{goal.target_position[0]:.3f}, {goal.target_position[1]:.3f}, {goal.target_position[2]:.3f}]")
            if goal.wrist_roll_deg is not None:
                print(f"   Wrist Roll: {goal.wrist_roll_deg:.1f}°")
            if goal.wrist_flex_deg is not None:
                print(f"   Wrist Flex: {goal.wrist_flex_deg:.1f}°")
            if goal.wrist_yaw_deg is not None:
                print(f"   Wrist Yaw: {goal.wrist_yaw_deg:.1f}°")
            if goal.gripper_closed is not None:
                print(f"   Gripper: {'CLOSED' if goal.gripper_closed else 'OPEN'}")
            if goal.metadata:
                print(f"   Metadata: {pformat(goal.metadata, indent=4)}")
            print()
        else:
            # Use the parent class method to send to queue
            await super().send_goal(goal) 