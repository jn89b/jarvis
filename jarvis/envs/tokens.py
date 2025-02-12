from enum import Enum


class ControlIndex(Enum):
    """
    Control index for action space
    """
    ROLL: int = 0
    ALTITUDE: int = 1
    VELOCITY: int = 2
    HEADING: int = 3


class ObservationIndex(Enum):
    """
    Observation index for observation space
    """
    X: int = 0
    Y: int = 1
    Z: int = 2
    ROLL: int = 3
    PITCH: int = 4
    YAW: int = 5
    AIRSPEED: int = 6


class KinematicIndex(Enum):
    X: int = 0
    Y: int = 1
    Z: int = 2
    ROLL: int = 3
    PITCH: int = 4
    YAW: int = 5
    V: int = 6
    P: int = 7
    Q: int = 8
    R: int = 9
