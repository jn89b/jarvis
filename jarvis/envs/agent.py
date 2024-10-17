
import numpy as np
from typing import List, Tuple, Dict
from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.tokens import ControlIndex
from jarvis.utils.vector import StateVector
from jarvis.algos.pro_nav import ProNav
# import aircraftsim
from aircraftsim import (
    SimInterface,
    AircraftState
)
from aircraftsim import HighControlInputs


class Agent():
    is_pursuer: bool = None
    is_controlled: bool = None

    def __init__(self,
                 battle_space: BattleSpace,
                 state_vector: StateVector,
                 sim_interface: SimInterface,
                 id: int = None,
                 radius_bubble: float = 0.0,
                 ) -> None:
        self.battle_space: BattleSpace = battle_space
        self.state_vector: StateVector = state_vector
        self.sim_interface: SimInterface = sim_interface
        self.high_control_inputs: HighControlInputs = HighControlInputs(
            ctrl_idx=0,
            roll=0.0,
            pitch=0.0,
            alt_ref_m=0.0,
            yaw=0.0,
            vel_cmd=0.0
        )
        self.id: int = id
        self.radius_bubble: float = radius_bubble
        self.crashed: bool = False

        if self.id is None:
            raise ValueError("Agent ID must be provided.")

    def act(self, action: np.ndarray) -> None:
        """
        For the JSBIM interface for now we will
        map the action to the high control inputs
        with a vel_cmd, roll_cmd, alt_cmd
        """
        roll_idx: int = ControlIndex.ROLL.value
        alt_idx: int = ControlIndex.ALTITUDE.value
        vel_idx: int = ControlIndex.VELOCITY.value
        self.high_control_inputs = HighControlInputs(
            ctrl_idx=0,
            roll=action[roll_idx],
            pitch=0.0,
            alt_ref_m=action[alt_idx],
            yaw=0,
            vel_cmd=action[vel_idx]
        )

    def fall_down(self) -> None:
        """
        Let the agent fall down
        """
        self.high_control_inputs = HighControlInputs(
            ctrl_idx=0,
            roll=0.0,
            pitch=0.0,
            alt_ref_m=0.0,
            yaw=0.0,
            vel_cmd=0.0
        )

    def step(self) -> None:
        """
        Step the agent
        """
        if self.crashed:
            self.fall_down()

        self.sim_interface.step(self.high_control_inputs)
        self.on_state_update()

    def distance_to(self, other: "Agent",
                    use_2d: bool = False) -> float:
        if use_2d:
            return self.state_vector.distance_2D(other.state_vector)
        else:
            return self.state_vector.distance_3D(other.state_vector)

    def heading_difference(self, other: "Agent") -> float:
        return self.state_vector.heading_difference(other.state_vector)

    def is_close_to_parallel(self, other: "Agent", threshold: float = 0.7) -> bool:
        dot_product = self.state_vector.dot_product_2D(other.state_vector)
        if dot_product > threshold:
            return True

        return False

    def on_state_update(self) -> None:
        """
        Update the state of the agent
        Note since we are in 2D we only care about 
        x,y,psi,v
        """
        aircraft_state: AircraftState = self.sim_interface.get_states()
        new_vector = StateVector(
            aircraft_state.x,
            aircraft_state.y,
            aircraft_state.z,
            aircraft_state.roll,
            aircraft_state.pitch,
            aircraft_state.yaw,
            aircraft_state.airspeed
        )
        # we'll keep this information for now
        self.state_vector = new_vector

    def get_observation(self) -> np.ndarray:
        """
        Get the observation of the agent
        Which is the state vector
        """
        return self.state_vector.array


class Evader(Agent):
    is_pursuer: bool = False
    is_controlled: bool = True

    def __init__(self, battle_space: BattleSpace,
                 state_vector: StateVector,
                 sim_interface: SimInterface,
                 radius_bubble: float,
                 id: int = None) -> None:
        super().__init__(battle_space, state_vector, sim_interface, id, radius_bubble)


class Pursuer(Agent):
    is_pursuer: bool = True
    is_controlled: bool = False

    def __init__(self, battle_space: BattleSpace,
                 state_vector: StateVector,
                 sim_interface: SimInterface,
                 id: int = None,
                 radius_bubble: float = 0.0,
                 pursuer_state_limits: Dict = None,
                 pursuer_control_limits: Dict = None,
                 capture_distance: float = 5.0) -> None:
        super().__init__(battle_space, state_vector,
                         sim_interface, id, radius_bubble)
        self.pursuer_state_limits: Dict = pursuer_state_limits
        self.pursuer_control_limits: Dict = pursuer_control_limits
        self.capture_distance: float = capture_distance
        self.dt: float = self.sim_interface.dt
        self.pro_nav: ProNav = ProNav(
            dt=0.1, nav_constant=4.0, capture_distance=self.capture_distance)

    def chase(self, target: Evader) -> None:
        """
        Implement the chase algorithm and sets the action
        """
        a_cmd, latax = self.pro_nav.navigate(
            self.state_vector, target.state_vector)
        vel_cmd: float = self.state_vector.speed + a_cmd
        roll_cmd: float = np.arctan2(latax, -9.81)
        # clip the roll command
        roll_cmd = np.clip(roll_cmd, -np.pi/4, np.pi/4)
        alt_cmd: float = target.state_vector.z
        print(f"Roll: {roll_cmd}, Alt: {alt_cmd}, Vel: {vel_cmd}")
        action = np.array([roll_cmd, alt_cmd, vel_cmd])
        self.act(action)
