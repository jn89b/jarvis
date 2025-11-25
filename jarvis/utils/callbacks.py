"""MongoDB logging callbacks for Ray RLlib multi-agent reinforcement learning experiments.

This module provides a three-callback architecture for capturing complete experiment telemetry:
- WargameCallback: Per-iteration metrics logging (rewards, timesteps, wall time)
- WargameCheckpointCallback: Checkpoint lifecycle event logging via Ray AIR hooks
- WargameTuneCallback: Final experiment snapshot aggregation on trial completion

All callbacks use event-driven capture (no filesystem crawling) and store data in MongoDB
with clean separation: runs collection (streaming) and experiments collection (snapshots).

Configuration:
    Environment variables required:
        - enable_mongo: "true" to enable MongoDB logging
        - mongo_uri: MongoDB connection URI (default: mongodb://localhost:27017/)
        - db: Database name (default: wargame_experiments)
        - experiments_collection: Collection name (default: experiments)
        - runs_collection: Collection name (default: runs)

Example:
    >>> from jarvis.utils.callbacks import WargameCallback, WargameCheckpointCallback, WargameTuneCallback
    >>> 
    >>> # Configure Ray Tune callbacks
    >>> run_config = tune.RunConfig(
    ...     callbacks=[
    ...         WargameTuneCallback(),
    ...         WargameCheckpointCallback(),
    ...     ]
    ... )
    >>> 
    >>> # Configure RLlib callback
    >>> algo_config = PPOConfig().callbacks(WargameCallback)
"""

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
    """Shared MongoDB connection and file parsing utilities for Ray RLlib callbacks.
    
    Provides lazy MongoDB connection initialization and common file parsing methods
    for Ray experiment artifacts. Designed as a mixin to avoid code duplication
    across callback implementations.
    
    Key features:
        - Lazy connection: MongoDB client created only when first needed (avoids Ray pickling issues)
        - Graceful degradation: Continues without MongoDB if connection fails
        - Environment-based config: All settings via environment variables
        - Shared utilities: Experiment ID extraction, JSON/NDJSON/CSV parsing
    
    Attributes:
        enable_mongo: Whether MongoDB logging is enabled (from environment)
        experiments_collection: MongoDB collection for final experiment snapshots
        runs_collection: MongoDB collection for streaming training metrics
        _mongodb_initialized: Internal flag tracking connection state
    
    Note:
        Subclasses should call `_ensure_mongodb_connection()` before any MongoDB operations.
        Connection is initialized once per callback instance, not per method call.
    """
    
    enable_mongo: bool
    experiments_collection: Optional[Collection]
    runs_collection: Optional[Collection]
    _mongodb_initialized: bool
    
    def _ensure_mongodb_connection(self) -> None:
        """Lazily initialize MongoDB connection on first use.
        
        Connects to MongoDB using environment variables and verifies connection with ping.
        If connection fails or MongoDB is disabled, sets enable_mongo=False and continues
        without logging (graceful degradation).
        
        This lazy initialization pattern avoids Ray's serialization issues when distributing
        callback objects across workers. Connection happens in the worker process, not during
        callback instantiation.
        
        Environment Variables:
            enable_mongo: "true" to enable, anything else disables
            mongo_uri: MongoDB connection string (default: mongodb://localhost:27017/)
            db: Database name (default: wargame_experiments)
            experiments_collection: Collection for final snapshots (default: experiments)
            runs_collection: Collection for streaming metrics (default: runs)
        
        Side Effects:
            Sets instance attributes: enable_mongo, experiments_collection, runs_collection,
            _mongodb_initialized. Prints connection status to stdout.
        
        Note:
            Called automatically by callback methods before MongoDB operations.
            Safe to call multiple times (returns immediately if already initialized).
        """
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
        """Extract unique experiment identifier from RLlib or Tune context.
        
        Experiment ID is derived from the trial directory name, which serves as a stable
        identifier across the training lifecycle. Supports both RLlib (algorithm.logdir)
        and Ray Tune (trial.local_path) contexts.
        
        Args:
            algorithm: RLlib Algorithm instance with logdir attribute (training context)
            trial: Ray Tune Trial instance with local_path attribute (tuning context)
            
        Returns:
            Experiment ID string. Priority order:
                1. algorithm.logdir directory name (if provided and not 'working_dirs')
                2. trial.local_path directory name (if provided)
                3. Timestamped fallback: exp_YYYYMMDD_HHMMSS_microseconds
        
        Example:
            >>> mixin._get_experiment_id(trial=trial)
            'PPO_MultiAgentEnv_2025-11-24_12-30-45_abc123'
            
            >>> mixin._get_experiment_id()  # Fallback
            'exp_20251124_123045_789012'
        
        Note:
            The 'working_dirs' filter prevents using temporary RLlib working directories
            as experiment IDs during initialization.
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
        """Load and parse JSON file with optional key filtering.
        
        Uses orjson for fast parsing of Ray Tune experiment artifacts (params.json, etc.).
        Supports extracting only specific keys to reduce memory footprint for large configs.
        
        Args:
            file_path: Absolute path to JSON file
            keys_to_extract: Optional list of top-level keys to extract. If provided,
                returns dict with only these keys. If None, returns full document.
        
        Returns:
            Parsed JSON as dict. Returns empty dict if file doesn't exist.
        
        Example:
            >>> params = mixin._load_json(
            ...     Path("/path/to/params.json"),
            ...     keys_to_extract=["env", "policies"]
            ... )
        """
        if file_path.exists():
            with open(file_path, 'r') as file:
                data = orjson.loads(file.read())
                if keys_to_extract:
                    return {k: data.get(k) for k in keys_to_extract if k in data}
                return data
        return {}
    
    def _load_ndjson(self, file_path: Path) -> Dict[str, Any]:
        """Load final entry from newline-delimited JSON file.
        
        Ray Tune writes results as NDJSON (one JSON object per line). This method
        efficiently extracts the last line (final training result) without loading
        the entire file into memory.
        
        Args:
            file_path: Absolute path to NDJSON file (typically result.json)
        
        Returns:
            Parsed last line as dict. Returns empty dict if file doesn't exist,
            is empty, or last line is blank/whitespace only.
        
        Note:
            Blank line protection: Empty or whitespace-only lines are skipped to
            prevent orjson parsing errors from trailing newlines.
        """
        if file_path.exists():
            with open(file_path, 'r') as file:
                last_line = None
                for line in file:
                    last_line = line
                if last_line and last_line.strip():
                    return orjson.loads(last_line)
        return {}

    def _parse_metrics(self, file_path: Path) -> Dict[str, Any]:
        """Extract final metrics from Ray Tune progress CSV.
        
        Ray Tune logs per-iteration metrics to progress.csv. This method reads the
        last row (final iteration) and converts it to a dictionary.
        
        Args:
            file_path: Absolute path to progress.csv file
        
        Returns:
            Dict mapping column names to values from last row. Returns empty dict
            if file doesn't exist or CSV is empty.
        
        Example:
            >>> metrics = mixin._parse_metrics(Path("/path/to/progress.csv"))
            >>> print(metrics["episode_reward_mean"])  # Final reward
        """
        if file_path.exists():
            progress_df = pd.read_csv(file_path)
            return progress_df.iloc[-1].to_dict()
        return {}
    
class WargameCallback(DefaultCallbacks, MongoDBMixin):
    """RLlib callback for streaming per-iteration training metrics to MongoDB.
    
    Captures and stores training progress metrics after each RLlib training iteration,
    including episode rewards, agent performance, environment steps, and wall time.
    Data flows to the `runs` collection for time-series analysis.
    
    MongoDB Schema:
        Updates `runs` collection with:
            - experiment_id: Trial identifier
            - metrics_history[]: Array of iteration snapshots with iteration number,
                reward_mean, agent_rewards, env_steps, wall_time_s, timestamp
            - last_updated: Timestamp of most recent update
    
    Usage:
        >>> from ray.rllib.algorithms.ppo import PPOConfig
        >>> config = PPOConfig().callbacks(WargameCallback)
        >>> algo = config.build(env="MyEnv")
        >>> algo.train()  # Metrics logged automatically
    
    Note:
        Checkpoints are managed by WargameCheckpointCallback for clean separation.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def on_train_result(self, *, algorithm: Any, result: Dict[str, Any], **kwargs) -> None:
        """Callback hook fired after each RLlib training iteration.
        
        Extracts key metrics from the training result and appends them to the MongoDB
        runs collection. Creates a new document if this is the first iteration.
        
        Args:
            algorithm: RLlib Algorithm instance (provides experiment ID via logdir)
            result: Training result dict from RLlib containing training_iteration,
                env_runners, num_env_steps_sampled_lifetime, time_total_s
            **kwargs: Additional keyword arguments (unused)
        
        Note:
            Silently returns if MongoDB is disabled. Errors are logged but don't interrupt training.
        """
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
    """Ray Tune callback for event-driven checkpoint metadata capture.
    
    Hooks into Ray Tune's on_checkpoint() lifecycle event to capture checkpoint
    metadata at creation time using the Ray AIR Checkpoint API. Eliminates filesystem
    crawling and provides authoritative checkpoint paths.
    
    Key Features:
        - Event-driven via on_checkpoint() hook
        - Uses checkpoint.path from Ray AIR (authoritative source)
        - Storage-agnostic (local, NFS, S3, GCS)
        - No race conditions or filesystem assumptions
    
    MongoDB Schema:
        Updates `runs` collection with:
            - experiment_id: Trial identifier
            - checkpoints[]: Array of checkpoint events with checkpoint_id,
                checkpoint_dir, trial_id, iteration, timestamp
            - last_updated: Timestamp of most recent update
    
    Usage:
        >>> run_config = tune.RunConfig(
        ...     callbacks=[WargameCheckpointCallback()],
        ...     checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=5)
        ... )
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
        """Ray Tune lifecycle hook fired immediately after checkpoint creation.
        
        Extracts checkpoint path from the Ray AIR Checkpoint object and stores metadata
        in the MongoDB runs collection. This is the authoritative source for checkpoint
        locations—no filesystem crawling required.
        
        Args:
            iteration: Current Ray Tune iteration
            trials: List of all Trial objects in the experiment (unused)
            trial: Trial object that created this checkpoint (provides trial_id, local_path)
            checkpoint: Ray AIR Checkpoint object (provides path - authoritative source)
            **info: Additional metadata from Ray Tune (unused)
        
        Note:
            checkpoint.path is guaranteed valid when this hook fires. Ray Tune only calls
            this after the checkpoint write completes. Silently returns if MongoDB disabled.
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
    """Ray Tune callback for final experiment snapshot aggregation.
    
    Fires when experiments complete to create immutable snapshots in the experiments
    collection. Aggregates configuration, final metrics, results, and checkpoint
    references for downstream evaluation and policy rollback.
    
    Key Features:
        - Lifecycle-aware: Fires on on_experiment_end() for all completion types
        - Status tracking: Records success, error, or interruption
        - Aggregates from runs: Pulls checkpoint list from runs collection
        - Parses Ray artifacts: Reads params.json, result.json, progress.csv
    
    MongoDB Schema:
        Creates/updates `experiments` collection with:
            - experiment_id, env_name, algorithm, timestamp
            - params (extracted keys), results (final), metrics (final row)
            - checkpoints (aggregated from runs), status, last_updated
    
    Usage:
        >>> run_config = tune.RunConfig(callbacks=[WargameTuneCallback()])
        >>> tuner = tune.Tuner(trainable, run_config=run_config)
        >>> tuner.fit()  # Snapshot created on completion
    """

    def __init__(self) -> None:
        super().__init__()

    def _process_trial_data(self, trial: Any, status: str = "complete") -> None:
        """Parse trial artifacts and create experiment snapshot in MongoDB.
        
        Reads Ray Tune experiment files, aggregates checkpoint metadata from runs
        collection, and creates a complete experiment document.
        
        Args:
            trial: Ray Tune Trial object (provides local_path, trial_id, status)
            status: Human-readable status: "complete", "error", or "interrupted"
        
        Note:
            Silently returns if MongoDB disabled or trial directory doesn't exist.
            Uses upsert, so safe to call multiple times for same experiment.
        """
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
        """Ray Tune lifecycle hook fired when experiment terminates.
        
        Processes all trials and creates snapshots in the experiments collection.
        Handles normal completion, errors, and user interruption (Ctrl+C).
        
        Args:
            trials: List of Trial objects from the completed experiment
            **info: Additional metadata from Ray Tune (unused)
        
        Note:
            Maps Ray Tune statuses: TERMINATED→"complete", ERROR→"error", other→"interrupted".
            Silently returns if MongoDB disabled.
        """
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