import logging
import pygame

logger = logging.getLogger(__name__)


# :----- Naming conventions -----:
# - (l/r)s: left/right stick
# - (l/r)b: left/right bumper
# - (l/r)t: left/right trigger
# - dpad: d-pad
# - abxy: a/b/x/y buttons, for PS this is x/o/square/triangle
# - start: the tiny button on the top right
# - back: the tiny button on the top left
# -logo: logo button on the middle
#
# Note:
# 1. For sticks, 'ls' or 'rs' means left/right stick being pressed.
# 2. To specify the direction, add '_up/_down/_left/_right' to the key name, e.g., 'ls_up', 'dpad_up'.
# 3. To combine conditions, use '&' to connect them (no space around the '&').
#    E.g., 'ls&ls_up' means left stick being pressed and pushing up.
# 4. To negate a condition, add '!' before the condition, e.g., '!ls' means left stick not being pressed.



LEFT_KEYMAP: dict[str, str] = {
    # 左臂 XZ 控制 (左摇杆; 未按下)
    'left_arm.x+': '!ls&!rs&!lb&ls_up',
    'left_arm.x-': '!ls&!rs&!lb&ls_down',
    'left_arm.z+': '!ls&!rs&!lb&ls_right',
    'left_arm.z-': '!ls&!rs&!lb&ls_left',
    # 左臂 shoulder_pan 和 pitch 控制 (LB 按下 + 左摇杆)
    'left_arm.pitch+': 'lb&ls_down',
    'left_arm.pitch-': 'lb&ls_up',
    'left_arm.shoulder_pan+': 'lb&ls_left',
    'left_arm.shoulder_pan-': 'lb&ls_right',
    # 左臂 wrist_yaw (new) 和 wrist_roll 控制 (LB 按下 + D-pad)
    'left_arm.wrist_roll+': 'lb&dpad_up',
    'left_arm.wrist_roll-': 'lb&dpad_down',
    'left_arm.wrist_yaw+': 'lb&dpad_left',
    'left_arm.wrist_yaw-': 'lb&dpad_right',
    # 左夹爪控制 (LT)
    'left_arm.gripper+': '!lb&lt',
    'left_arm.gripper-': 'lb&lt'
}
RIGHT_KEYMAP = {
    # 右臂 XZ 控制 (右摇杆; 未按下)
    'right_arm.x+': '!ls&!rs&!rb&rs_up',
    'right_arm.x-': '!ls&!rs&!rb&rs_down',
    'right_arm.z+': '!ls&!rs&!rb&rs_right',
    'right_arm.z-': '!ls&!rs&!rb&rs_left',
    # 右臂 shoulder_pan 和 pitch 控制 (RB 按下 + 左摇杆)
    'right_arm.pitch+': 'rb&rs_down',
    'right_arm.pitch-': 'rb&rs_up',
    'right_arm.shoulder_pan+': 'rb&rs_left',
    'right_arm.shoulder_pan-': 'rb&rs_right',
    # 右臂 wrist_yaw (new) 和 wrist_roll 控制 (RB 按下 + abxy)
    'right_arm.wrist_roll+': 'rb&y',
    'right_arm.wrist_roll-': 'rb&a',
    'right_arm.wrist_yaw+': 'rb&x',
    'right_arm.wrist_yaw-': 'rb&b',
    # 右夹爪控制 (RT)
    'right_arm.gripper+': '!rb&rt',
    'right_arm.gripper-': 'rb&rt'
}
HEAD_KEYMAP = {
    # 头部电机控制 Xbox: (x, y, a, b); PS5: (s, t, x, o)
    "head.yaw+": '!rb&x',
    "head.yaw-": '!rb&b',
    "head.pitch+": '!rb&a',
    "head.pitch-": '!rb&y'
}
BASE_KEYMAP = {
    # 底盘控制
    'base.forward': '!lb&!rs&dpad_up',
    'base.backward': '!lb&!rs&dpad_down',
    'base.left': 'rs&!lb&dpad_left',
    'base.right': 'rs&!lb&dpad_right',
    'base.rotate_left': '!lb&!rs&dpad_left',
    'base.rotate_right': '!lb&!rs&dpad_right',
    'base.speed_up': 'back' # Xbox: back; PS5: create/options button; the tiny button on the left
}

RESET_KEYMAP = {
    'back_robot_to_zero': 'start', # Xbox: start; PS5: share button; the tiny button on the right
    'exit_teleop': 'logo'
}

RECORD_KEYMAP = {
    'exit_early': 'rs&ls_right',
    'rerecord_episode': 'rs&ls_left',
    'stop_recording': 'rs&ls_up',
}

ALL_KEYMAP = {
    **LEFT_KEYMAP,
    **RIGHT_KEYMAP,
    **HEAD_KEYMAP,
    **BASE_KEYMAP,
    **RESET_KEYMAP,
    **RECORD_KEYMAP,
}

def print_decode_keymap(keymap: dict[str, str]):
    words = {
        'ls': 'left stick press',
        'rs': 'right stick press',
        'lb': 'left bumper',
        'rb': 'right bumper',
        'lt': 'left trigger',
        'rt': 'right trigger',
        'start': 'start button',
        'back': 'back button',
        'logo': 'logo button',
        'dpad': 'd-pad',
        '!': 'not ',
        '_': ' '
    }
    print("\033[92m")
    print("*-----------------------------*")
    print("*      Control Key Map        *")
    print("*-----------------------------*")
    print("*   [Action] -> Key Mapping   *")
    print("*-----------------------------*")
    print("\033[0m")
    for action, control in keymap.items():
        conditions = control.split('&')
        for i, condition in enumerate(conditions):
            condition = condition.replace('ls_', 'left stick ')
            condition = condition.replace('rs_', 'right stick ')
            for c, n in words.items():
                condition = condition.replace(c, n)
            conditions[i] = condition
        _conn = ' \033[4mand\033[0m '
        print(f"\033[94m[{action:^10}]\033[0m {_conn.join(conditions):^15}")
        

def decode_key(gamepad, code: str) -> bool:
    conditions = code.split('&')
    state = True
    for condition in conditions:
        negate = condition.startswith('!')
        condition = condition.lstrip('!')
        if condition in ['ls', 'rs', 'lb', 'rb', 'lt', 'rt', 'start', 'back', 'logo', 'a', 'b', 'x', 'y']:
            state_i = gamepad.get_button(condition)
        elif condition.startswith('ls') or condition.startswith('rs'):
            stick = gamepad.get_left_stick() if condition.startswith('ls') else gamepad.get_right_stick()
            if condition.endswith('_up'):
                state_i= stick[1] < -0.5
            elif condition.endswith('_down'):
                state_i= stick[1] > 0.5
            elif condition.endswith('_left'):
                state_i= stick[0] < -0.5
            elif condition.endswith('_right'):
                state_i= stick[0] > 0.5
        elif condition.startswith('dpad'):
            dpad = gamepad.get_dpad()
            if condition.endswith('_up'):
                state_i= dpad[1] == 1
            elif condition.endswith('_down'):
                state_i= dpad[1] == -1
            elif condition.endswith('_left'):
                state_i= dpad[0] == -1
            elif condition.endswith('_right'):
                state_i= dpad[0] == 1
        else:
            raise ValueError(f"Invalid key: {condition}")
        if negate:
            state_i = not state_i
        state &= state_i
    return state

def get_gamepad_states(gamepad, keymap: dict[str, str]) -> dict[str, bool]:
    gamepad.update()
    states = dict.fromkeys(keymap.keys(), False)
    for action, control in keymap.items():
        states[action] = decode_key(gamepad, control)
    return states


class PS5Gamepad:
    """
    Reference: https://www.pygame.org/docs/ref/joystick.html#playstation-5-controller-pygame-2-x
    *Note*: 't' and 's' are swapped. It is better to run the jstest-gtk to get the correct mapping.
    - 'a/x': button 0
    - 'b/o': button 1
    - 'y/t': button 2
    - 'x/s': button 3
    """
    def __init__(self, id: int = 0):
        self.joystick = None
        self.id = id
        self.reset()

    def is_connected(self) -> bool:
        return self.joystick is not None and self.joystick.get_init()
    
    def connect(self) -> None:
        # Initialize
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            logger.error("No gamepad detected. Please connect a gamepad and try again.")
            return
        self.joystick = pygame.joystick.Joystick(self.id)
        self.joystick.init()
        logger.info(f"Initialized gamepad: {self.joystick.get_name()}")
    
    def disconnect(self):
        if self.is_connected():
            self.joystick.quit()
        pygame.quit()
        self.joystick = None   
    
    def update(self):
        pygame.event.pump()
        self.axes = [self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())]
        self.buttons = [bool(self.joystick.get_button(i)) for i in range(self.joystick.get_numbuttons())]
        self.hat = self.joystick.get_hat(0) if self.joystick.get_numhats() > 0 else (0, 0)
        assert len(self.axes) == 6, "Expected 6 axes, got {}".format(len(self.axes))
        assert len(self.buttons) == 13, "Expected 13 buttons, got {}".format(len(self.buttons))
    
    def reset(self):
        self.axes = [0.0] * 6
        self.buttons = [False] * 13
        self.hat = (0, 0)

    def get_button(self, name: str) -> bool:
        if name == 'ls':
            return self.buttons[11]
        elif name == 'rs':
            return self.buttons[12]
        elif name == 'lb':
            return self.buttons[4]
        elif name == 'rb':
            return self.buttons[5]
        elif name == 'lt':
            return self.axes[2] > 0.5
        elif name == 'rt':
            return self.axes[5] > 0.5
        elif name == 'start':
            return self.buttons[8]
        elif name == 'back':
            return self.buttons[9]
        elif name == 'logo':
            return self.buttons[10]
        elif name == 'a':
            return self.buttons[0]
        elif name == 'b':
            return self.buttons[1]
        elif name == 'x':
            return self.buttons[3]
        elif name == 'y':
            return self.buttons[2]
        else:
            raise ValueError(f"Invalid button: {name}")
        return False

    def get_left_stick(self) -> tuple[float, float]:
        return (self.axes[0], self.axes[1]) # (x, y)

    def get_right_stick(self) -> tuple[float, float]:
        return (self.axes[3], self.axes[4]) # (x, y)

    def get_dpad(self) -> tuple[int, int]:
        return (self.hat[0], self.hat[1]) # (x, y)


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.DEBUG)

    print_decode_keymap(ALL_KEYMAP)
    gamepad = PS5Gamepad()
    gamepad.connect()
    while gamepad.is_connected():
        states = get_gamepad_states(gamepad, ALL_KEYMAP)
        _states = {k: v for k, v in states.items() if v}
        print("Actions:", _states)
        print('\033[1A\033[K', end='')  # Move cursor up one line and clear the line
        time.sleep(0.1)