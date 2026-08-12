#!/usr/bin/env python3
"""Step-through hardware test for the L-ONE arm.

Exercises each configured channel one at a time, pausing before every
move so you can watch and confirm which physical joint responds. Reuses
the raw-REPL bridge from virtual_gripper.py.

Only MOTOR1 (Base), PWM1 (Upper Arm), PWM2 (Lower Arm), and PWM3
(Gripper) are wired on this build -- rc_config leaves MOTOR2/PWM4
unconfigured, so they're skipped here.

Run: python test_motors.py [port]
"""

import sys
import time

from virtual_gripper import CyberBrickLink, find_default_port

MOTOR_SPEED = 900   # MotorsController.set_speed range: -2048..2048
SERVO_SPEED = 50    # ServosController.set_speed range: -100..100


def wait(msg):
    input(f"{msg} [Enter to run, Ctrl+C to abort] ")


def test_motor(link, idx, name):
    wait(f"MOTOR{idx} ({name}): forward then reverse")
    print("  forward...")
    link.set_motor_speed(idx, MOTOR_SPEED)
    time.sleep(0.6)
    link.stop_motor(idx)
    time.sleep(0.3)
    print("  reverse...")
    link.set_motor_speed(idx, -MOTOR_SPEED)
    time.sleep(0.6)
    link.stop_motor(idx)
    print("  done.\n")


def test_speed_servo(link, idx, name):
    wait(f"Servo {idx} ({name}): one direction then the other")
    print("  +speed...")
    link.set_servo_speed(idx, SERVO_SPEED)
    time.sleep(0.6)
    link.stop_servo(idx)
    time.sleep(0.3)
    print("  -speed...")
    link.set_servo_speed(idx, -SERVO_SPEED)
    time.sleep(0.6)
    link.stop_servo(idx)
    print("  done.\n")


def test_angle_servo(link, idx, name):
    wait(f"Servo {idx} ({name}): sweep 30 -> 150 -> 90 degrees")
    for angle in (30, 150, 90):
        print(f"  -> {angle} deg")
        link.set_servo_angle(idx, angle)
        time.sleep(0.8)
    print("  done.\n")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_default_port()
    if not port:
        print("No serial port found. Pass one explicitly: python test_motors.py /dev/cu.usbmodem101")
        return

    print(f"Connecting on {port}...")
    link = CyberBrickLink(port)
    print("Connected. Board's normal RC loop is paused until this script exits.\n")

    try:
        test_motor(link, 1, "Base")
        test_speed_servo(link, 1, "Upper Arm")
        test_speed_servo(link, 2, "Lower Arm")
        test_angle_servo(link, 3, "Gripper")
        print("All channels tested.")
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        print("Stopping everything and resetting board (remote control restored)...")
        link.close()


if __name__ == "__main__":
    main()
