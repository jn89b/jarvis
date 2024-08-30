from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from jarvis.envs.simple_2d_env import BattleEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from jarvis.utils.seed_function import set_global_seed
from jarvis.config.env_config_2d import NUM_PURSUERS

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

class SaveVecNormalizeCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, name_prefix, vec_normalize_env, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(save_freq, save_path, name_prefix, verbose)
        self.vec_normalize_env = vec_normalize_env

    def _on_step(self) -> bool:
        result = super(SaveVecNormalizeCallback, self)._on_step()

        if self.num_timesteps % self.save_freq == 0:
            # Save the VecNormalize statistics
            if self.vec_normalize_env is not None and self.model.get_env() is self.vec_normalize_env:
                vec_normalize_path = f"{self.save_path}/{self.name_prefix}_vecnormalize_{self.num_timesteps}.pkl"
                self.vec_normalize_env.save(vec_normalize_path)
                if self.verbose > 0:
                    print(f"Saved VecNormalize to {vec_normalize_path}")

        return result


class PrintNormalizedCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PrintNormalizedCallback, self).__init__(verbose)
        #print every 1000 steps
        self.step_freq = 3000
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if the environment is wrapped with VecNormalize
        # if isinstance(self.training_env, VecNormalize):
        #     for env_idx in range(self.training_env.num_envs):
        #         normalized_reward = self.locals['rewards'][env_idx]
        #         print(f"Normalized Reward: {normalized_reward}")
        # else:
        #     for env_idx in range(self.training_env.num_envs):
        #         reward = self.locals['rewards'][env_idx]
        #         print(f"Reward: {reward}")
        #         #get attributes
        #         #obs = env.get_attr('observation', env_idx)
        
        # Access the rewards from the training environment
        # rewards = self.locals['rewards']
        # # Log the mean reward
        # self.logger.record('train/mean_reward', sum(rewards) / len(rewards))
        
        # dones = self.locals['dones']
        
        # # Track the length of the episodes
        # for i, done in enumerate(dones):
        #     if done:
        #         # Retrieve the current step count (which should represent the episode length)
        #         current_step = self.training_env.get_attr('current_step', i)[0]
        #         self.episode_lengths.append(current_step)
        #         self.episode_count += 1
        
        # # Log the mean episode length if any episodes were completed
        # if len(self.episode_lengths) > 0:
        #     mean_len = sum(self.episode_lengths) / len(self.episode_lengths)
        #     # self.logger.record('train/mean_episode_length', mean_len)
        #     # Optionally reset the list after logging
        #     self.episode_lengths = []

        if self.num_timesteps % self.step_freq == 0:
            for env_idx in range(self.training_env.num_envs):
                num_wins = self.training_env.get_attr(
                    'num_wins', env_idx)
                current_sim_num = self.training_env.get_attr(
                    'current_sim_num', env_idx)
                
                #compute the average step count
                if current_sim_num[0] > 0:
                    win_percentage = num_wins[0] / current_sim_num[0]
                    #print in printy format the win percentage and the index out of the total number of simulations
                    print(f"Win Percentage: {win_percentage} at sim number {current_sim_num[0]} for env index {env_idx}") 
                    #reset the number of wins and current sim number
                    self.training_env.set_attr('num_wins', 0, env_idx)
                    self.training_env.set_attr('current_sim_num', 0, env_idx)
            
            #print new line
            print("\n")
        return True


# Function to create an instance of the environment
def create_env():
    return BattleEnv(spawn_own_space=False, spawn_own_agents=False)

if __name__ == "__main__":
    seed_num = 0
    set_global_seed(seed=seed_num)
    # Create a list of environments to run in parallel
    num_envs = 4  # Adjust this number based on your CPU cores
    LOAD_MODEL = False
    CONTINUE_TRAINING = False
    COMPARE_MODELS = False
    TOTAL_TIME_STEPS = 2000000
    model_name = "PPO_evader_2D"
    num_pursuers = str(NUM_PURSUERS) + '_pursuers'
    version_num = 0 
    save_path = './models/' + model_name + '_' \
        + num_pursuers + '_' + str(version_num)

    # Normalize the environment (observations and rewards)
    env = SubprocVecEnv([create_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env = VecMonitor(env)  # Monitor the vectorized environment
    
    # Check the environment to ensure it's correctly set up
    test_env = BattleEnv()
    check_env(test_env)
    # Add the callback
    callback = PrintNormalizedCallback(verbose=1)

    # Define the CheckpointCallback to save the model 
    # every `save_freq` steps
    # Define the custom checkpoint callback
    checkpoint_callback = SaveVecNormalizeCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path=save_path,
        name_prefix=model_name,
        vec_normalize_env=env,
        verbose=1
    )
    
    # Load or initialize the model
    if LOAD_MODEL and not CONTINUE_TRAINING:
        environment = BattleEnv()  # Create a single instance of the environment for evaluation
        model = PPO.load(model_name, env=environment)
        print("Model loaded.")
    elif LOAD_MODEL and CONTINUE_TRAINING:
        print("Loading model and continuing training:", model_name)
        model = PPO.load(model_name, env=env)
        model.learn(total_timesteps=TOTAL_TIME_STEPS, 
                    log_interval=1,
                    callback=[callback, checkpoint_callback])
        model.save(model_name)
        print("Model saved.")
    else:
        model = PPO('MlpPolicy', 
                    env, 
                    verbose=1,
                    tensorboard_log="./logs/"+model_name,
                    learning_rate=0.003,
                    ent_coef=0.01,
                    n_steps=2048,
                    batch_size=128,
                    seed=seed_num,
                    #use gpu
                    device='cuda')
        # Train the model in parallel
        model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=[callback,
                                                    checkpoint_callback])
        
        model.save(model_name)
        print("Model saved.")