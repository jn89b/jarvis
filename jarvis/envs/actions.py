from dataclasses import dataclass, field
import numpy as np

@dataclass
class InputControl:
    """
    Input High Level controls for the fixed wing aircraft.
    """
    def __init__(self, throttle: float, aileron: float, elevator: float, rudder: float):
        self.throttle = throttle
        self.aileron = aileron
        self.elevator = elevator
        self.rudder = rudder