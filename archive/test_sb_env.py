from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from jarvis.envs.battle_env_single import BattleEnv

from stable_baselines3.common.callbacks import BaseCallback

class PrintNormalizedCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PrintNormalizedCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Retrieve the environment
        env = self.training_env
        # Get the current step's observation, reward, done, and info
        
        # Check if the environment is wrapped with VecNormalize
        # if isinstance(env, VecNormalize):
        #     # Get the normalized rewards directly from VecNormalize's `ret_rms`
        #     # which stores the running mean and variance of the returns
        #     for env_idx in range(env.num_envs):
        #         normalized_reward = self.locals['rewards'][env_idx]
        #         print(f"Normalized Reward: {normalized_reward}")
        # else:
        #     # If not using VecNormalize, just print the rewards directly
        #     for env_idx in range(env.num_envs):
        #         reward = self.locals['rewards'][env_idx]
        #         print(f"Reward: {reward}")
        
        # for env_idx in range(env.num_envs):
        #     obs = env.get_attr('observation', env_idx)
        #     reward = env.get_attr('reward', env_idx)
        #     done = env.get_attr('done', env_idx)
        #     # info = env.get_attr('info', env_idx)
            
        #     # Print the observation, reward, done status, and info for the current step
        #     print(f"Observation: {obs}")
        #     print(f"Reward: {reward}")
        #     print(f"Done: {done}")
        #     # print(f"Info: {info}")
        
        return True


test_env = BattleEnv()
check_env(test_env)

# Create the environment
def create_env():
    return BattleEnv(spawn_own_space=False, spawn_own_agents=False)

env = DummyVecEnv([create_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Add the callback
callback = PrintNormalizedCallback(verbose=1)

# Train the model with the callback
model.learn(total_timesteps=10000, callback=callback)
