"""
Load a trained agent that can avoid pursuers and also allow capabilities 
to load the transformer model for deployment applications 

-This means I need to convert the inputs to feed it to my transformer model:
    - Get 20 past observations of the ego vehicle
    - Get 20 past observations of the pursuers
    - Batch the data and send into dataloader for processing


"""

import matplotlib.pyplot as plt
import gc
import ray
from jarvis.utils.sim_helper import RLSimHelper
from jarvis.utils.trainer import load_yaml_config

plt.close('all')
gc.collect()
# Used to clean up the Ray processes after training
ray.shutdown()
ray.init()

if __name__ == "__main__":
    # Load the RL simulation environment
    # checkpoint_path = "/home/justin/ray_results/PPO_2025-06-02_11-39-52/PPO_high_speed_pursuer_evader_35767_00000_0_2025-06-02_11-39-53/checkpoint_000005" 
    checkpoint_path:str = "/home/justin/ray_results/skyhunter_evader/PPO_2025-06-10_13-19-45/PPO_pursuer_evader_env_7c419_00000_0_2025-06-10_13-19-45/checkpoint_000224"
    num_episodes = 1
    use_pronav = True
    save = False
    index_save = 0
    folder_dir = 'rl_pickle'
    
    env_config = load_yaml_config(
        "config/simple_env_high_speed_config.yaml"  )['battlespace_environment']
    env_config = load_yaml_config(
        "config/simple_env_config.yaml")['battlespace_environment']
    #predictformer_config = "config/high_speed_predictformer_config.yaml"
    predictformer_config = "config/predictformer_config.yaml"
    
    rl_sim_helper = RLSimHelper(
        checkpoint_path=checkpoint_path,
        env_config=env_config,
        num_episodes=num_episodes,
        use_pronav=use_pronav,
        save=save,
        index_save=index_save,
        folder_dir=folder_dir,
        use_predictformer=True,
        predictformer_config=predictformer_config,
        env_type="pursuer_evader"
    )
    
    
    for i in range(5):
        rl_sim_helper.infer_pursuer_evader_env(head_on_placement=True,
                                            use_predictformer_for_rl=True)
        