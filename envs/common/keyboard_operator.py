import time
from sshkeyboard import listen_keyboard
from publisher import DataPublisher
import random
import sys
import numpy as np
pi = np.pi


data_publisher = DataPublisher(
    'udp://localhost:9871', encoding="msgpack", broadcast=True)


keyboard_operator_cmd = np.zeros(3) # vel_x, vel_y, yaw_orientation

delta_cmd = np.array([0.05, 0.05, 0.05])
reset = False
push = False
def press(key):
    global keyboard_operator_cmd
    global reset
    global push
    if key == 'i':
        keyboard_operator_cmd[0] += delta_cmd[0]
    elif key == 'k':
        keyboard_operator_cmd[0] -= delta_cmd[0]
    elif key == 'j':
        keyboard_operator_cmd[1] += delta_cmd[1]
    elif key == 'l':
        keyboard_operator_cmd[1] -= delta_cmd[1]
    elif key == 'u':
        keyboard_operator_cmd[2] += delta_cmd[2]
    elif key == 'o':
        keyboard_operator_cmd[2] -= delta_cmd[2]
    elif key == '0':
        keyboard_operator_cmd[:] = 0
    elif key == '9':
        reset = True
    elif key=='p':
        push = True
    keyboard_operator_cmd = np.where(np.abs(keyboard_operator_cmd) < 1e-5, 0, keyboard_operator_cmd)
    keyboard_operator_cmd.round(5)
    print(keyboard_operator_cmd)
    data = {
        "cmd": keyboard_operator_cmd,
        "reset": reset,
        "push": push,
    }
    data_publisher.publish(data)
    reset=False
    push=False
    # elif key == 't':

listen_keyboard(on_press=press,delay_second_char=0.005,delay_other_chars=0.005,sleep=0.0001)
