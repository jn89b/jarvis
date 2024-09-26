from jarvis.utils.Trainer2D import RLTrainer2D
from jarvis.config.env_config_2d import NUM_PURSUERS
if __name__ == '__main__':
    
    test_num = 0
    num_pursuers = NUM_PURSUERS
    model_name = "PPO_evader" + "_2D_" + str(num_pursuers) + "pursuers" + "_test_" + str(test_num)
    
    continue_training = False
    load_model = False
    use_discrete_actions = True
    total_time_steps = 3000000
    rl_trainer = RLTrainer2D(model_name=model_name,
                             load_model=False,
                             continue_training=False,
                             use_discrete_actions=use_discrete_actions,
                             env_type=0,  
                             total_time_steps=total_time_steps)
    rl_trainer.train()