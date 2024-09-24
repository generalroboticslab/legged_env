import numpy as np
from publisher import DataPublisher
import pygame


lin_vel_x_max = 0.3
lin_vel_y_max = 0.25
ang_vel_z_max = 0.2


keyboard_operator_cmd = np.zeros(3) # vel_x, vel_y, yaw_orientation
reset = False

data_publisher = DataPublisher(
    'udp://localhost:9871', encoding="msgpack", broadcast=True)

pygame.init()
pygame.joystick.init()

joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
for joystick in joysticks:
    joystick.init()
    print(joystick.get_name())

running = True
while running:

    updated = False


    for event in pygame.event.get():
        # print(event)
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.JOYBUTTONDOWN:
            # print("Button pressed:", event.button)
            if event.button == 4:
                updated = True
                reset = True
                # print("reset")
        elif event.type == pygame.JOYAXISMOTION:
            # print("Axis moved:", event.axis, event.value)
            if event.axis==1:
                keyboard_operator_cmd[0] = np.round(-event.value * lin_vel_x_max,decimals=2)
                updated = True
            elif event.axis==0:
                keyboard_operator_cmd[2] = np.round(-event.value * ang_vel_z_max,decimals=2)
                updated = True
            elif event.axis==3:
                keyboard_operator_cmd[1] = np.round(-event.value * lin_vel_y_max,decimals=2)
                updated = True

        # print(gamepad_states)
    if updated:

        data = {
            "cmd": keyboard_operator_cmd,
            "reset": reset,
        }
        print(data)

        data_publisher.publish(data)
        reset=False

pygame.quit()