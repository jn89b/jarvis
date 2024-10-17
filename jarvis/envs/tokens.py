from enum import Enum


class ControlIndex(Enum):
    """
    Control index for action space
    """
    ROLL: int = 0
    ALTITUDE: int = 1
    VELOCITY: int = 2
