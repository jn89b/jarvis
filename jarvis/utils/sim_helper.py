import pathlib
import time
import torch
import numpy as np
from typing import Dict, Any, List
from jarvis.envs.multi_agent_env import PursuerEvaderEnv
from jarvis.utils.mask import SimpleEnvMaskModule
from ray.rllib.core.rl_module import RLModule


class RLSimHelper():
    """
    This class is used to help load the RL simulation environment
    used to train the RL model.
    Users must specify the simulation configuration environment
    """
    def __init__(self, checkpoint_path: str, num_episodes: int = 1,
          use_pronav: bool = True, save: bool = False,
          index_save: int = 0, folder_dir: str = 'rl_pickle',
          env_type:str="pursuer_evader") -> None:
        
        self.checkpoint_path: str = checkpoint_path
        self.num_episodes: int = num_episodes
        self.use_pronav: bool = use_pronav
        self.save: bool = save
        self.index_save: int = index_save
        self.folder_dir: str = folder_dir
        self.env_type: str = env_type
        self.env = self.start_env()
        
    def start_env(self) -> None:
        if self.env_type == "pursuer_evader":
            self.env: PursuerEvaderEnv = self.create_multi_agent_env(
                config=None,
                env_config=self.checkpoint_path)
        
    def create_multi_agent_env(self,
        config: Dict[str, Any],
        env_config: Dict[str, Any]) -> PursuerEvaderEnv:

        return PursuerEvaderEnv(
            config=env_config)

    def infer_pursuer_evader_env(self, 
            use_transformer:bool=False) -> None:
        """
        This function is used to infer the pursuer evader environment
        and run the simulation for the specified number of episodes.
        """
        checkpoint_path: str = self.checkpoint_path
        env:PursuerEvaderEnv = self.env
        
        # load the model from our checkpoints
        # Create only the neural network (RLModule) from our checkpoint.
        evader_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
            pathlib.Path(checkpoint_path) /
            "learner_group" / "learner" / "rl_module"
        )["evader_policy"]

        pursuer_policy: SimpleEnvMaskModule = RLModule.from_checkpoint(
            pathlib.Path(checkpoint_path) /
            "learner_group" / "learner" / "rl_module"
        )["pursuer_policy"]

        if use_transformer:
            pass

        reward_list: List[float] = []
        
        while not terminated['__all__']:
            start_time = time.time()
            key_value = list(observation.keys())[0]
            if key_value == '1':
                obs = observation['1']
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = pursuer_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]
            elif key_value == '0':
                obs = observation['0']
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = evader_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]
                
            elif key_value == '2':
                obs = observation['2']
                torch_obs_batch = {k: torch.from_numpy(
                    np.array([v])) for k, v in obs.items()}
                action_logits = pursuer_policy.forward_inference({"obs": torch_obs_batch})[
                    "action_dist_inputs"]

            # For my action space I have a multidscrete environment
            # Since my action logits are a [1 x total_actions] tensor
            # I need to get the argmax of the tensor
            end_time = time.time()
            # print("time: ", end_time - start_time)
            action_logits = action_logits.detach().numpy().squeeze()
            unwrapped_action: Dict[str, np.array] = env.unwrap_action_mask(
                action_logits)

            discrete_actions = []
            for k, v in unwrapped_action.items():
                v = torch.from_numpy(v)
                best_action = torch.argmax(v).numpy()
                discrete_actions.append(best_action)
            
            # action = torch.argmax(action_logits).numpy()
            action_dict = {}
            action_dict[key_value] = {'action': discrete_actions}
            # print("action dict: ", action_dict)

            observation, reward, terminated, truncated, info = env.step(
                action_dict=action_dict)

            reward_list.append(reward)

            # check if done
            if (terminated['__all__'] == True):
                # print("reward: ", reward)
                break


