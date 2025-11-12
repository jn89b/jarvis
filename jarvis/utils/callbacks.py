from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import datetime
import os

load_dotenv()

class SaveVecNormalizeCallback(CheckpointCallback):
    """
    This saves the VecNormalize environment statistics to a file
    Use this if you are normalizing observations and rewards to save
    it with the model as a pickle file.
    """

    def __init__(self,
                 save_freq: int,
                 save_path: str,
                 name_prefix: str,
                 vec_normalize_env: VecNormalize,
                 verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(
            save_freq, save_path, name_prefix, verbose)
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
    
class WargameCallback(DefaultCallbacks):
    """
    Minimal RLlib callback for MongoDB logging.
    - Logs metrics per iteration
    - Scans and logs checkpoints from ray_results
    - Marks experiment complete on stop
    """

    def __init__(self):
        super().__init__()
        self.enable_mongo = os.getenv("enable_mongo", "false").lower() == "true"
        self.collection = None
        self._experiment_id = None

        if self.enable_mongo:
            try:
                client = MongoClient(os.getenv("mongo_uri", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                db = client[os.getenv("db", "wargame_experiments")]
                self.collection = db[os.getenv("collection", "experiments")]
            except Exception as e:
                print(f"[WargameCallback] MongoDB connection failed: {e}")
                self.enable_mongo = False

    def _get_experiment_id(self, algorithm) -> str:
        """Get unique experiment ID from trial directory name (cached)."""
        if self._experiment_id:
            return self._experiment_id
        
        if algorithm and hasattr(algorithm, 'logdir'):
            trial_name = Path(algorithm.logdir).name
            if trial_name and trial_name != "working_dirs":
                self._experiment_id = trial_name
                return self._experiment_id
        
        self._experiment_id = f"exp_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        return self._experiment_id
    
    # TODO: This document will actually be for another collection (runs) linked to (experiments) by experiment_id
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Store minimal metrics snapshot per iteration."""
        if not self.enable_mongo or self.collection is None:
            return
        try:
            experiment_id = self._get_experiment_id(algorithm=algorithm)
            env_runners = result.get("env_runners", {})
            
            metrics_entry = {
                "iteration": result.get("training_iteration", 0),
                "reward_mean": env_runners.get("episode_return_mean"),
                "agent_rewards": env_runners.get("agent_episode_returns_mean", {}),
                "env_steps": result.get("num_env_steps_sampled_lifetime", 0),
                "wall_time_s": result.get("time_total_s", 0.0),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

            self.collection.update_one(
                {"experiment_id": experiment_id},
                {
                    "$push": {"metrics_history": metrics_entry},
                    "$set": {"last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                },
                upsert=True
            )
            
            # Scan ray_results for checkpoints
            if hasattr(algorithm, 'logdir'):
                trial_name = Path(algorithm.logdir).name
                ray_results = Path.home() / "ray_results"
                
                for exp_dir in sorted(ray_results.glob("PPO_*"), key=lambda p: p.stat().st_mtime, reverse=True): # TODO: filter by algo
                    trial_dirs = list(exp_dir.glob(trial_name))
                    if trial_dirs:
                        for ckpt_path in sorted(trial_dirs[0].glob("checkpoint_*")):
                            if not self.collection.find_one({"experiment_id": experiment_id, "checkpoints.id": ckpt_path.name}):
                                self.collection.update_one(
                                    {"experiment_id": experiment_id},
                                    {"$push": {"checkpoints": {
                                        "id": ckpt_path.name,
                                        "path": str(ckpt_path),
                                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                                    }}},
                                    upsert=True
                                )
                        break
        except Exception as e:
            print(f"[WargameCallback] Error: {e}")


    # TODO: implement on algo stop to regiester full experiment completion (this will call our parser script that we already created)
    def on_algorithm_stop(self, *, algorithm, **kwargs):
        """Mark experiment complete."""
        if not self.enable_mongo or self.collection is None:
            return
        
        try:
            experiment_id = self._get_experiment_id(algorithm=algorithm)
            
            self.collection.update_one(
                {"experiment_id": experiment_id},
                {
                    "$set": {
                        "status": "complete",
                        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }
                },
                upsert=True
            )
            
            print(f"[WargameCallback] Experiment complete: {experiment_id}")
        except Exception as e:
            print(f"[WargameCallback] Error: {e}")

