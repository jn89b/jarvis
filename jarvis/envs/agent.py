
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

    def act(self, action: np.ndarray, ctrl_idx: int = 0) -> None:
        """
        For the JSBIM interface for now we will
        map the action to the high control inputs
        with a vel_cmd, roll_cmd, alt_cmd
        """
        roll_idx: int = ControlIndex.ROLL.value
        alt_idx: int = ControlIndex.ALTITUDE.value
        vel_idx: int = ControlIndex.VELOCITY.value
        heading_idx: int = ControlIndex.HEADING.value

        if self.is_pursuer:
            self.high_control_inputs = HighControlInputs(
                ctrl_idx=1,
                heading_ref_deg=np.rad2deg(action[heading_idx]),
                pitch=0.0,
                alt_ref_m=action[alt_idx],
                yaw=0,
                vel_cmd=action[vel_idx]
            )
        else:
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
            dt=self.dt, nav_constant=2.0, capture_distance=self.capture_distance)
        self.previous_target_state: StateVector = None
        self.previous_ego_state: StateVector = None
        self.dy_old: float = 0.0
        self.dx_old: float = 0.0

    def pursuit(self, target: Evader) -> np.ndarray:
        """
        Implements a pure pursuit algorithm to navigate the pursuer

        """
        aircraft_state: AircraftState = self.sim_interface.get_states()
        dx = target.state_vector.x - aircraft_state.x
        dy = target.state_vector.y - aircraft_state.y
        current_yaw = aircraft_state.yaw
        heading_cmd = np.arctan2(dx, dy)

        error = heading_cmd - current_yaw
        if error > np.pi:
            error = error - 2*np.pi
        elif error < -np.pi:
            error = error + 2*np.pi
        roll_cmd = np.clip(error, -np.pi/4, np.pi/4)
        vel_cmd = 25.0  # self.pursuer_control_limits["v_cmd"]['max']
        alt_cmd = target.state_vector.z
        action = np.array(
            [roll_cmd, alt_cmd, vel_cmd, heading_cmd])
        return action

    def pn(self, target: Evader) -> np.ndarray:
        """
        Use the pro nav algorithm to navigate the pursuer
        """

    def chase(self, target: Evader) -> None:
        """
        Implement the chase algorithm and sets the action
        """
        # pursuit_action: np.ndarray = self.pursuit(target)
        # # pn_action: np.ndarray = self.pn(target)

        # self.act(pursuit_action)
        aircraft_state: AircraftState = self.sim_interface.get_states()
        evader_state: AircraftState = target.sim_interface.get_states()

        if self.previous_ego_state is not None and self.previous_target_state is not None:

            dy_old = evader_state.y - self.previous_target_state.y
            dx_old = evader_state.x - self.previous_target_state.x

            dy = target.state_vector.y - aircraft_state.y
            dx = target.state_vector.x - aircraft_state.x

            # RTM_old: StateVector = self.previous_target_state - self.previous_ego_state
            # RTM_new: StateVector = target.state_vector - self.state_vector

            # theta_old: float = np.arctan2(RTM_old.array[0], RTM_old.array[1])
            # theta_new: float = np.arctan2(RTM_new.array[0], RTM_new.array[1])
            theta_old: float = np.arctan2(dx_old, dy_old)
            theta_new: float = np.arctan2(dx, dy)
            # RTM_new = RTM_new.array[:3] / np.linalg.norm(RTM_new.array[:3])
            # RTM_old = RTM_old.array[:3] / np.linalg.norm(RTM_old.array[:3])

            RTM_new = np.array([dx_old, dy_old, 0])
            RTM_old = np.array([self.previous_target_state.x - self.previous_ego_state.x,
                                self.previous_target_state.y - self.previous_ego_state.y, 0])

            if np.linalg.norm(RTM_old) == 0:
                LOS_Delta = np.array([0, 0, 0])
                LOS_Rate = 0.0
            else:
                LOS_Delta = RTM_new - RTM_old
                LOS_Rate = np.linalg.norm(LOS_Delta)

            # Calculate closing velocity
            Vc = -LOS_Rate
            # Calculate lateral acceleration (latax) using APN with At
            # check if los_rate is close to zero
            if abs(LOS_Rate) == 0:
                # use pure pursuit
                action = self.pursuit(target)
                # Vc, yaw_cmd = self.pursuit(target)
            else:
                los_dot = (theta_new - theta_old) / 0.1
                yaw_cmd = self.pro_nav.nav_constant * los_dot
                yaw_cmd = yaw_cmd - self.state_vector.yaw_rad
                # yaw_cmd = self.state_vector.yaw_rad + yaw_cmd
                airspeed_cmd = self.state_vector.speed - Vc
                airspeed_cmd = 25.0
                action = np.array(
                    [0.0, target.state_vector.z, airspeed_cmd, yaw_cmd])
        else:
            Vc = 0
            latax = 0
            yaw_desired = latax / self.state_vector.speed
            yaw_cmd = yaw_desired
            # yaw_cmd = self.state_vector.yaw_rad + yaw_cmd
            airspeed_cmd = self.state_vector.speed + Vc
            airspeed_cmd = 25.0
            action = np.array(
                [0.0, target.state_vector.z, airspeed_cmd, yaw_cmd])
            self.previous_ego_state = aircraft_state
            self.previous_target_state = evader_state
        action = self.pursuit(target)
        self.previous_ego_state = aircraft_state
        self.previous_target_state = evader_state
        self.act(action)

        # airspeed_cmd = self.state_vector.speed + Vc
        # action = np.array(
        #     [pursuit_action[0], pursuit_action[1], airspeed_cmd, yaw_cmd])
        # self.act(action)
