from jarvis.utils.Trainer2D import RLTrainer2D
from jarvis.config.env_config_2d import NUM_PURSUERS

import numpy as np

def set_global_seed(seed:int):
    np.random.seed(seed)
    

if __name__ == '__main__':
    test_num = 2
    num_pursuers = NUM_PURSUERS
    # model_name = "PPO_evader" + "_2D_" + str(num_pursuers) + "pursuers" + "_test_" + str(test_num)    
    #model_name = "PPO_evader_2D_3000000_steps"
    continue_training = True
    load_model = True
    upload_norm_obs = True
    
    model_name = "PPO_evader" + "_2D_" + str(num_pursuers) + "_pursuers" + "_test_" + str(test_num)    
        
    #set global seed for reproducibility
    seed_num = 1
    set_global_seed(seed=seed_num)
    rl_trainer = RLTrainer2D(model_name=model_name,
                             load_model=load_model,
                             continue_training=continue_training,
                             env_type=0,  
                             vec_env_path=model_name,
                             upload_norm_obs=True,
                             total_time_steps=3000000,
                             use_discrete_actions=True)
    
    rl_trainer.infer_model(num_evals=15)