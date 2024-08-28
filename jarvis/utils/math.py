import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecNormalize

# def normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
#     """
#     Helper to normalize observation.
#     :param obs:
#     :param obs_rms: associated statistics
#     :return: normalized observation
#     """
    
#     return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

def normalize_obs(obs: np.ndarray, vec_env:VecNormalize) -> np.ndarray:
    """
    Helper to normalize observation.
    :param obs:
    :param obs_rms: associated statistics
    :return: normalized observation
    #TODO: This is super dumb but works for now need
    """    
    obs_rms = vec_env.obs_rms
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + vec_env.epsilon), -vec_env.clip_obs, vec_env.clip_obs)

# def unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
#     """
#     Helper to unnormalize observation.
#     :param obs:
#     :param obs_rms: associated statistics
#     :return: unnormalized observation
#     """
#     return (obs * np.sqrt(obs_rms.var + self.epsilon)) + obs_rms.mean

def unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
    """
    Helper to unnormalize observation.
    :param obs:
    :param obs_rms: associated statistics
    :return: unnormalized observation
    """
    return (obs * np.sqrt(obs_rms.var + self.epsilon)) + obs_rms.mean

