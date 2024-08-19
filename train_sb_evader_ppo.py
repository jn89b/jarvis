from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
from jarvis.envs.battle_env_single import BattleEnv

# Assuming BattleEnv is imported correctly

# Create the environment
def create_env():
    return BattleEnv(spawn_own_space=False, spawn_own_agents=False)

# Wrap the environment with DummyVecEnv to handle vectorized environments
env = DummyVecEnv([create_env])

# # Normalize observations and rewards
# env = VecNormalize(env, norm_obs=False, norm_reward=False)

# Optionally, you can normalize the actions if you define a custom policy or use a wrapper
# Set up a callback for saving the model at intervals
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                         name_prefix='ppo_battle_env')

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=1000000, callback=checkpoint_callback)

# Save the final model
model.save("ppo_battle_env_final")

# Remember to save VecNormalize statistics when saving the model
env.save("ppo_battle_env_vecnormalize.pkl")

# To load the model later:
# model = PPO.load("ppo_battle_env_final", env=env)
# env = VecNormalize.load("ppo_battle_env_vecnormalize.pkl", env)


