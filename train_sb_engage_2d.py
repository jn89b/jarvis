from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from jarvis.envs.simple_2d_env import EngagementEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from jarvis.utils.callbacks import SaveVecNormalizeCallback
# Define a callback (optional) for printing normalized rewards

# Function to create an instance of the environment
def create_env():
    return EngagementEnv(spawn_own_space=False, 
                         spawn_own_agents=False, 
                         use_heuristic_policy=False,
                         randomize_goal=RANDOM_GOAL,
                         difficulty_level=LEVEL_DIFFICULTY,
                         randomize_start=RANDOM_START,
                         use_discrete_actions=True)  # Adjust this to match your environment creation

if __name__ == "__main__":
    # Create a list of environments to run in parallel
    num_envs = 5  # Adjust this number based on your CPU cores
    LEVEL_DIFFICULTY = 2 # 0, 1, 2, 3
    LOAD_MODEL = True
    CONTINUE_TRAINING = True
    TOTAL_TIME_STEPS = 6000000
    RANDOM_GOAL = True
    RANDOM_START = False
    
    env = SubprocVecEnv([create_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env = VecMonitor(env)  # Monitor the vectorized environment
    index_number = 0
    model_name = "PPO_engage_2D" + "_difficulty_" + str(LEVEL_DIFFICULTY) + "_" + str(index_number) 
    print("Model name:", model_name)
    # Normalize the environment (observations and rewards)

    # Check the environment to ensure it's correctly set up
    test_env = EngagementEnv(use_heuristic_policy=True)
    check_env(test_env)
    
    # Vectorized callback to save into pickle file
    callback = SaveVecNormalizeCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path='./models/'+model_name,
        name_prefix=model_name,
        vec_normalize_env=env,
        verbose=1   
    )
    
    # Define the CheckpointCallback to save the model every `save_freq` steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, 
                                             save_path='./models/'+model_name,
                                            name_prefix=model_name)

    # Load or initialize the model
    if LOAD_MODEL and not CONTINUE_TRAINING:
        environment = EngagementEnv(randomize_goal=RANDOM_GOAL,
                                    difficulty_level=LEVEL_DIFFICULTY,
                                    randomize_start=RANDOM_START,
                                    use_discrete_actions=True)  # Create a single instance of the environment for evaluation
        model = PPO.load(model_name, env=environment)
        print("Model loaded.")
    elif LOAD_MODEL and CONTINUE_TRAINING:
        print("Loading model and continuing training:", model_name)
        # model.set_env(env)

        # Load the VecNormalize statistics
        env = VecNormalize.load(model_name+'_vecnormalize.pkl', env)
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
                    learning_rate=0.0003,
                    # ent_coef=0.01,
                    n_steps=2048,
                    batch_size=64,
                    #use gpu
                    device='cuda')
        # Train the model in parallel
        model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=[callback,
                                                    checkpoint_callback])
        
        model.save(model_name)
        #save the VecNormalize statistics
        #save 
        print("Model saved.")