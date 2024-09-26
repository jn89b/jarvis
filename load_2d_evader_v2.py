from jarvis.utils.Trainer2D import RLTrainer2D
from jarvis.config.env_config_2d import NUM_PURSUERS


if __name__ == '__main__':
    test_num = 0
    num_pursuers = NUM_PURSUERS
    model_name = "PPO_evader" + "_2D_" + str(num_pursuers) + "pursuers" + "_test_" + str(test_num)    
    continue_training = False
    load_model = True
    upload_norm_obs = True
    rl_trainer = RLTrainer2D(model_name=model_name,
                             load_model=load_model,
                             continue_training=continue_training,
                             env_type=0,  
                             vec_env_path=model_name,
                             upload_norm_obs=True,
                             total_time_steps=3000000,
                             use_discrete_actions=True)
    
    rl_trainer.infer_model()