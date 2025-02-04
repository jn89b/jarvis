from typing import Dict
import time
import random

from ray.rllib.env import EnvContext

from your_constants import NUM_CHICKENS, NUM_DIRECTIONS
from your_rllib_environment import YourEnvironment


def is_all_done(done: Dict) -> bool:
    for key, val in done.items():
        if not val:
            return False
    return True


env = YourEnvironment(
    config=None
)
env.reset()

action_dict = {'robot_1_high': random.choice(range(NUM_CHICKENS)),
               'robot_2_high': random.choice(range(NUM_CHICKENS))}

obs, rew, done, info = env.step(action_dict)
env.render()

while not is_all_done(done):
    action_dict = {}
    assert 'robot_1_low' in obs or 'robot_2_low' in obs
    if 'robot_1_low' in obs and not done['robot_1_low']:
        action_dict['robot_1_low'] = random.choice(range(NUM_DIRECTIONS))
    if 'robot_2_low' in obs and not done['robot_2_low']:
        action_dict['robot_2_low'] = random.choice(range(NUM_DIRECTIONS))
    obs, rew, done, info = env.step(action_dict)
    print("Reward: ", rew)
    print("action_dict: ", action_dict)
    print("done: ", done)
    print("obs: ", obs)
    time.sleep(.1)
    env.render()
