from jarvis.envs.battle_space import BattleSpace
from jarvis.config import env_config

def setup_battlespace() -> BattleSpace:
    #returns a battlespace object
    battlespace = BattleSpace(
        x_bounds=env_config.X_BOUNDS,
        y_bounds=env_config.Y_BOUNDS,
        z_bounds=(0, 0)  # 2D environment
    )
    
    return battlespace