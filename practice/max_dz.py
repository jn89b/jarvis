import numpy as np


def compute_ascent_descent_rate(current_vel: float, pitch_cmd: float) -> float:
    return current_vel * np.sin(pitch_cmd)


current_z: float = 70
max_z: float = 73
current_vel = 25
pitch_commands = np.arange(np.deg2rad(-20), np.deg2rad(20), np.deg2rad(1))

dt = 0.5
for pitch_cmd in pitch_commands:
    ascent_descent_rate = compute_ascent_descent_rate(
        current_vel, pitch_cmd)
    # print(
    #     f"pitch_cmd: {np.rad2deg(pitch_cmd)} ascent_descent_rate: {ascent_descent_rate}")
    projected_z = current_z + ascent_descent_rate * dt

    if projected_z > max_z:
        print("projected_z is greater than max_z at pitch_cmd: ",
              np.rad2deg(pitch_cmd))
