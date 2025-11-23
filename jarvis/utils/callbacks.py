from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray import tune
from ray.train import Checkpoint
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import datetime
import orjson
import os

load_dotenv()


class MongoDBMixin:
    """Shared MongoDB connection and file parsing logic for all wargame callbacks."""
    
    enable_mongo: bool
    experiments_collection: Optional[Collection]
    runs_collection: Optional[Collection]
    _mongodb_initialized: bool
    
    def _ensure_mongodb_connection(self) -> None:
        """Lazily initialize MongoDB connection when first needed (avoids pickling issues)."""
        # Check if already initialized
        if hasattr(self, '_mongodb_initialized') and self._mongodb_initialized:
            return
        
        self.enable_mongo = os.getenv("enable_mongo", "false").lower() == "true"
        self.experiments_collection = None
        self.runs_collection = None
        self._mongodb_initialized = True
        
        if self.enable_mongo:
            try:
                client = MongoClient(os.getenv("mongo_uri", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                db = client[os.getenv("db", "wargame_experiments")]
                experiments_collection = os.getenv("experiments_collection", "experiments")
                runs_collection = os.getenv("runs_collection", "runs")
                self.experiments_collection = db[experiments_collection]
                self.runs_collection = db[runs_collection]
                print(f"[{self.__class__.__name__}] Connected to MongoDB: {db.name}.{runs_collection} and {db.name}.{experiments_collection}")
                
            except Exception as e:
                print(f"[{self.__class__.__name__}] MongoDB connection failed: {e}")
                self.enable_mongo = False
        else:
            print(f"[{self.__class__.__name__}] MongoDB logging is disabled (enable_mongo={os.getenv('enable_mongo', 'not set')})")
    
    def _get_experiment_id(self, algorithm: Any = None, trial: Any = None) -> str:
        """Extract experiment ID from algorithm or trial object (experiment ID = trial directory name).
        
        Args:
            algorithm: RLlib algorithm instance (has logdir attribute)
            trial: Ray Tune trial instance (has local_path attribute)
            
        Returns:
            Experiment ID (trial directory name) or timestamped fallback if neither provided.
        """
        # Try algorithm.logdir first
        if algorithm and hasattr(algorithm, 'logdir'):
            trial_name = Path(algorithm.logdir).name
            if trial_name and trial_name != "working_dirs":
                return trial_name
        
        # Try trial.local_path second
        if trial and hasattr(trial, 'local_path'):
            trial_name = Path(trial.local_path).name
            if trial_name:
                return trial_name
        
        # Fallback: generate timestamped ID
        return f"exp_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    
    def _load_json(self, file_path: Path, keys_to_extract: Optional[list] = None) -> Dict[str, Any]:
        """Load JSON file and optionally extract specific keys."""
        if file_path.exists():
            with open(file_path, 'r') as file:
                data = orjson.loads(file.read())
                if keys_to_extract:
                    return {k: data.get(k) for k in keys_to_extract if k in data}
                return data
        return {}
    
    def _load_ndjson(self, file_path: Path) -> Dict[str, Any]:
        """Load last line of NDJSON file."""
        if file_path.exists():
            with open(file_path, 'r') as file:
                last_line = None
                for line in file:
                    last_line = line
                if last_line and last_line.strip():
                    return orjson.loads(last_line)
        return {}

    def _parse_metrics(self, file_path: Path) -> Dict[str, Any]:
        """Parse last row of progress CSV."""
        if file_path.exists():
            progress_df = pd.read_csv(file_path)
            return progress_df.iloc[-1].to_dict()
        return {}
    
class WargameCallback(DefaultCallbacks, MongoDBMixin):
    """RLlib callback for per-iteration MongoDB logging.
    
    - Logs metrics per iteration to runs collection
    - Checkpoints are captured by WargameCheckpointCallback (no directory crawling)
    """

    def __init__(self) -> None:
        super().__init__()
    
    def on_train_result(self, *, algorithm: Any, result: Dict[str, Any], **kwargs) -> None:
        """Store minimal metrics snapshot per iteration."""
        self._ensure_mongodb_connection()
        
        if not self.enable_mongo or self.runs_collection is None:
            return
        
        try:
            experiment_id = self._get_experiment_id(algorithm=algorithm)
            env_runners = result.get("env_runners", {})
            
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            metrics_entry = {
                "iteration": result.get("training_iteration", 0),
                "reward_mean": env_runners.get("episode_return_mean"),
                "agent_rewards": env_runners.get("agent_episode_returns_mean", {}),
                "env_steps": result.get("num_env_steps_sampled_lifetime", 0),
                "wall_time_s": result.get("time_total_s", 0.0),
                "timestamp": timestamp
            }

            self.runs_collection.update_one(
                {"experiment_id": experiment_id},
                {
                    "$push": {"metrics_history": metrics_entry},
                    "$set": {"last_updated": timestamp}
                },
                upsert=True
            )
        except Exception as e:
            print(f"[WargameCallback] Error during on_train_result: {e}")


class WargameCheckpointCallback(tune.Callback, MongoDBMixin):
    """Ray Tune callback for checkpoint lifecycle event logging.
    
    Hooks into Tune's on_checkpoint event to capture checkpoint metadata
    This approach:
    - Works across any Ray deployment (local, SLURM, air-gapped, cloud)
    - Uses Ray's documented lifecycle hooks
    - Captures checkpoints as they're created, not discovered after the fact
    - No filesystem layout assumptions
    """

    def __init__(self) -> None:
        super().__init__()

    def on_checkpoint(
        self,
        iteration: int,
        trials: List[Any],
        trial: Any,
        checkpoint: Checkpoint,
        **info: Dict[str, Any],
    ) -> None:
        """Called when Tune saves a checkpoint for a trial.
        
        Args:
            iteration: Current training iteration
            trials: List of all trials
            trial: Trial that saved the checkpoint
            checkpoint: Ray AIR Checkpoint object
            **info: Additional info from Tune
        """
        self._ensure_mongodb_connection()

        if not self.enable_mongo or self.runs_collection is None:
            return

        try:
            # Extract experiment ID using shared utility
            experiment_id = self._get_experiment_id(trial=trial)
            
            # Get checkpoint directory directly from Checkpoint object
            # This avoids race conditions with trial metadata and tmp directories
            ckpt_dir = Path(checkpoint.path)

            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            ckpt_metadata = {
                "checkpoint_id": ckpt_dir.name,  # e.g. "checkpoint_000120"
                "checkpoint_dir": str(ckpt_dir),
                "trial_id": trial.trial_id,
                "iteration": iteration,
                "timestamp": timestamp,
            }

            # Append checkpoint to runs collection (append-only, no deduplication needed)
            self.runs_collection.update_one(
                {"experiment_id": experiment_id},
                {
                    "$push": {"checkpoints": ckpt_metadata},
                    "$set": {"last_updated": timestamp},
                },
                upsert=True,
            )
            
            print(f"[WargameCheckpointCallback] Logged checkpoint: {ckpt_dir.name} for {experiment_id}")
        except Exception as e:
            print(f"[WargameCheckpointCallback] Error during on_checkpoint: {e}")
            import traceback
            traceback.print_exc()


class WargameTuneCallback(tune.Callback, MongoDBMixin):
    """
    Ray Tune callback for final experiment snapshot on trial completion.
    
    Triggered when a trial completes (normally or via Ctrl+C).
    Parses experiment artifacts from ray_results and stores them in experiments collection.
    """

    def __init__(self) -> None:
        super().__init__()

    def _process_trial_data(self, trial: Any, status: str = "complete") -> None:
        """Helper method to process and save trial data to MongoDB."""
        if not self.enable_mongo or self.experiments_collection is None:
            return
        
        try:
            # Extract experiment ID and get trial directory from Ray Tune API
            experiment_id = self._get_experiment_id(trial=trial)
            trial_dir = Path(trial.local_path)
            
            if not trial_dir.exists():
                print(f"[WargameTuneCallback] Trial directory not found on disk: {trial_dir}")
                return
            
            print(f"[WargameTuneCallback] Processing trial ({status}): {experiment_id}")
            print(f"[WargameTuneCallback] Trial directory: {trial_dir}")

            # Parameter keys to extract
            param_keys = ["env", "policies", "_rl_module_spec"]

            # Parse experiment files
            params = self._load_json(trial_dir / "params.json", keys_to_extract=param_keys)
            results = self._load_ndjson(trial_dir / "result.json")
            metrics = self._parse_metrics(trial_dir / "progress.csv")

            # Checkpoints are captured via AIR events and runs snapshots during training, not filesystem crawling
            runs_doc = self.runs_collection.find_one(
                {"experiment_id": experiment_id},
                {"_id": 0, "checkpoints": 1}
            )
            checkpoints = runs_doc.get("checkpoints", []) if runs_doc else []

            # Extract algorithm name from parent directory
            try:
                algorithm_name = trial_dir.parent.name.split("_")[0]
            except Exception:
                print(f"[WargameTuneCallback] Unable to extract algorithm name from {trial_dir.parent.name}")
                algorithm_name = "unknown"

            # Build experiment document
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            experiment_data = {
                "experiment_id": experiment_id,
                "env_name": params.get("env", "unknown"),
                "algorithm": algorithm_name,
                "timestamp": timestamp,
                "params": params,
                "results": results,
                "metrics": metrics,
                "checkpoints": checkpoints,
                "status": status,
                "last_updated": timestamp
            }

            # Upsert experiment document
            result = self.experiments_collection.update_one(
                {"experiment_id": experiment_id},
                {"$set": experiment_data},
                upsert=True
            )
            
            if result.upserted_id:
                print(f"[WargameTuneCallback] Experiment {experiment_id} ingested successfully.")
            else:
                print(f"[WargameTuneCallback] Experiment {experiment_id} updated in database.")
        
        except Exception as e:
            print(f"[WargameTuneCallback] Error during experiment ingestion: {e}")
            import traceback
            traceback.print_exc()

    def on_experiment_end(self, trials: List[Any], **info) -> None:
        """Called when experiment ends (completion, interruption, or error)."""
        self._ensure_mongodb_connection()
        
        if not self.enable_mongo or self.experiments_collection is None:
            return
        
        print(f"[WargameTuneCallback] Processing {len(trials)} trial(s)")
        
        for trial in trials:
            # Determine status based on trial state
            if trial.status == "TERMINATED":
                status = "complete"
            elif trial.status == "ERROR":
                status = "error"
            else:
                status = "interrupted"
            
            self._process_trial_data(trial, status=status)


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