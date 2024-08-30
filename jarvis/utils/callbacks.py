from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

class SaveVecNormalizeCallback(CheckpointCallback):
    """
    This saves the VecNormalize environment statistics to a file
    Use this if you are normalizing observations and rewards to save
    it with the model as a pickle file.
    """
    def __init__(self, 
                 save_freq:int, 
                 save_path:str, 
                 name_prefix:str, 
                 vec_normalize_env:VecNormalize, 
                 verbose=0):
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

