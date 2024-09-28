from jarvis.utils.Trainer2D import RLTrainer2D
from jarvis.config.env_config_2d import NUM_PURSUERS
from jarvis.config.env_config_2d import NUM_RADARS_MAX
if __name__ == '__main__':
    
    test_num = 3

    sim_config = {
        0: "AvoidThreatEnv",
        1: "EngagementEnv",
        2: "RCSEnv"
    }
    
    num_pursuers = NUM_PURSUERS
    
    continue_training = False
    load_model = False
    use_discrete_actions = True
    total_time_steps = 12000000
    num_radars = NUM_RADARS_MAX
    if continue_training:
        upload_norm_obs = True
    else:
        upload_norm_obs = False
    
    # doing this to keep track of the training to make it easier to load the model 
    if continue_training:
        print("Loading model and increasing test number", test_num)
        test_num += 1
    print("test number", test_num)
    
    model_name = "PPO_RCS" + "_2D_" + "num_radars_" +str(num_radars)+ "_test_" + str(test_num)
    
    rl_trainer = RLTrainer2D(model_name=model_name,
                             load_model=continue_training,
                             continue_training=load_model,
                             use_discrete_actions=use_discrete_actions,
                             env_type=2,  
                             total_time_steps=total_time_steps,
                             upload_norm_obs=upload_norm_obs,
                             vec_env_path=model_name)
    rl_trainer.train()
    