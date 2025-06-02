import numpy as np
from typing import List, Tuple, Dict
from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.tokens import ControlIndex
from jarvis.utils.vector import StateVector
from jarvis.algos.pro_nav import ProNav
from aircraftsim import (
    SimInterface,
    AircraftState
)
from aircraftsim import HighControlInputs

class Agent():
    """
    A generic agent class for JSBSim
    Requires Sim Interface which is the JSBSim interface and
    a state vector which is the state of the agent
    """
    is_pursuer: bool = False
    def __init__(self,
                 state_vector: StateVector,
                 sim_interface: SimInterface,
                 id: int = None,
                 radius_bubble: float = 0.0
                 ) -> None:
        
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

    def act(self, action: np.ndarray, 
            use_action:bool=True,
            heading_des_dg:float = None,
            alt_des_m:float = None,
            vel_des_m:float = None) -> None:
        """
        For the JSBIM interface for now we will
        map the action to the high control inputs
        with a vel_cmd, roll_cmd, alt_cmd
        """
        alt_idx: int = ControlIndex.ALTITUDE.value
        vel_idx: int = ControlIndex.VELOCITY.value
        heading_idx: int = ControlIndex.HEADING.value

        if use_action:
            # we will use the action
            self.high_control_inputs = HighControlInputs(
                ctrl_idx=1,
                heading_ref_deg=np.rad2deg(action[heading_idx]),
                pitch=0.0,
                alt_ref_m=action[alt_idx],
                yaw=0,
                vel_cmd=action[vel_idx]
            )
        else:            
            # we will use the action
            self.high_control_inputs = HighControlInputs(
                ctrl_idx=1,
                heading_ref_deg=heading_des_dg,
                pitch=0.0,
                alt_ref_m=alt_des_m,
                yaw=0,
                vel_cmd=vel_des_m
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
            aircraft_state.airspeed)
        
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
    def __init__(self,
                 state_vector: StateVector,
                 jsbsim_interface: SimInterface,
                 radius_bubble: float,
                 agent_id: int = None,
                 is_controlled: bool = False) -> None:
        super().__init__(
            state_vector,
            jsbsim_interface, 
            agent_id,
            radius_bubble, )
        self.old_distance_from_pursuer: float = 0.0
        self.old_line_of_sight: float = 0.0
        self.is_controlled: bool = is_controlled
        
class Pursuer(Agent):
    is_pursuer: bool = True
    def __init__(self,
                 state_vector: StateVector,
                 jsbsim_interface: SimInterface,
                 radius_bubble: float,
                 agent_id: int = None,
                 is_controlled: bool = False,
                 capture_radius: float = 10.0) -> None:
        super().__init__( 
                         state_vector,
                         jsbsim_interface, agent_id,
                         radius_bubble, )
        self.capture_radius: float = capture_radius
        self.old_distance_from_evader: float = 0.0
        self.old_line_of_sight: float = 0.0
        self.is_controlled: bool = is_controlled
        # ProNav is the Proportional Navigation controller
        # self.pro_nav: ProNav = ProNav(
        #     self.state_vector,
        #     self.sim_interface,
        #     self.capture_radius
        # )