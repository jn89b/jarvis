import copy
import pickle as pkl
from abc import ABC
from typing import Dict, List, Optional, Text, Tuple, TypeVar

import casadi as ca
import numpy as np

from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.tokens import KinematicIndex, ControlIndex
from jarvis.utils.vector import StateVector
from jarvis.algos.pro_nav import ProNav


def wrap_to_pi(angle: ca.SX) -> ca.SX:
    """
    Wrap an angle (CasADi SX expression) to the range [-pi, pi).

    Args:
        angle (ca.SX): Input angle in radians.

    Returns:
        ca.SX: The wrapped angle.
    """
    return angle - 2 * ca.pi * ca.floor((angle + ca.pi) / (2 * ca.pi))


class DataHandler:
    """
    Handles logging of simulation data including state variables, controls, wind, time, and rewards.
    """

    def __init__(self) -> None:
        """
        Initialize empty lists for storing simulation data.
        """
        self.x: List[float] = []
        self.y: List[float] = []
        self.z: List[float] = []
        self.phi: List[float] = []
        self.theta: List[float] = []
        self.psi: List[float] = []
        self.v: List[float] = []
        self.p: List[float] = []
        self.q: List[float] = []
        self.r: List[float] = []
        self.u_phi: List[float] = []
        self.u_theta: List[float] = []
        self.u_psi: List[float] = []
        self.v_cmd: List[float] = []
        self.wind_x: List[float] = []
        self.wind_y: List[float] = []
        self.wind_z: List[float] = []
        self.time: List[float] = []
        self.rewards: List[float] = []
        self.yaw: List[float] = []

    def update_states(self, info_array: np.ndarray) -> None:
        """
        Update the state variables from a given numpy array.

        The expected order is: [x, y, z, phi, theta, psi, v, p, q, r].

        Args:
            info_array (np.ndarray): Array containing state information.
        """
        self.x.append(info_array[0])
        self.y.append(info_array[1])
        self.z.append(info_array[2])
        self.phi.append(info_array[3])
        self.theta.append(info_array[4])
        self.psi.append(info_array[5])
        self.v.append(info_array[6])

    def update_controls(self, control_array: np.ndarray) -> None:
        """
        Update the control inputs from a given numpy array.

        The expected order is: [u_phi, u_theta, u_psi, v_cmd].

        Args:
            control_array (np.ndarray): Array containing control information.
        """
        self.u_phi.append(control_array[0])
        self.u_theta.append(control_array[1])
        self.u_psi.append(control_array[2])
        self.v_cmd.append(control_array[3])

    def update_reward(self, reward: float) -> None:
        """
        Update the reward log.

        Args:
            reward (float): The reward value to log.
        """
        self.rewards.append(reward)

    def update_time(self, time: float) -> None:
        """
        Update the time log.

        Args:
            time (float): The current time stamp.
        """
        self.time.append(time)

    def update_wind(self, wind_array: np.ndarray) -> None:
        """
        Update the wind data from a given numpy array.

        The expected order is: [wind_x, wind_y, wind_z].

        Args:
            wind_array (np.ndarray): Array containing wind information.
        """
        self.wind_x.append(wind_array[0])
        self.wind_y.append(wind_array[1])
        self.wind_z.append(wind_array[2])

    def update(self, info_array: np.ndarray,
               control_array: np.ndarray,
               wind_array: np.ndarray,
               time: float,
               reward: float) -> None:
        """
        Update all logged data (states, controls, wind, time, and reward) at once.

        Args:
            info_array (np.ndarray): State information.
            control_array (np.ndarray): Control input information.
            wind_array (np.ndarray): Wind data.
            time (float): Time stamp.
            reward (float): Reward value.
        """
        self.update_states(info_array)
        self.update_controls(control_array)
        self.update_wind(wind_array)
        self.update_time(time)
        self.update_reward(reward)


class PlaneKinematicModel:
    """
    A simple kinematic model of an aircraft operating in a North-East-Down (NED) frame.

    The model includes state and control definitions, wind effects, and numerical integration using
    a 4th order Runge-Kutta (RK45) method.
    """

    def __init__(self,
                 dt_val: float = 0.05,
                 tau_v: float = 0.15,
                 tau_phi: float = 0.1,
                 tau_theta: float = 0.12,
                 tau_psi: float = 0.15,
                 tau_p: float = 0.1,
                 tau_q: float = 0.1,
                 tau_r: float = 0.1) -> None:
        """
        Initialize the plane kinematic model with the specified time step and time constants.

        Args:
            dt_val (float): Integration time step.
            tau_v (float): Time constant for airspeed dynamics.
            tau_phi (float): Time constant for roll command tracking.
            tau_theta (float): Time constant for pitch command tracking.
            tau_psi (float): Time constant for yaw command tracking.
            tau_p (float): Time constant for roll rate dynamics.
            tau_q (float): Time constant for pitch rate dynamics.
            tau_r (float): Time constant for yaw rate dynamics.
        """
        self.dt_val: float = dt_val
        self.define_states()
        self.define_controls()
        self.define_wind()
        # Time constants
        self.tau_v: float = tau_v
        self.tau_phi: float = tau_phi
        self.tau_theta: float = tau_theta
        self.tau_psi: float = tau_psi
        self.tau_p: float = tau_p
        self.tau_q: float = tau_q
        self.tau_r: float = tau_r

        self.state_info: Optional[np.ndarray] = None
        self.set_state_space()
        self.data_handler: DataHandler = DataHandler()

    def update_state_info(self, state_info: np.ndarray) -> None:
        """
        Set the current state information and update the logged state data.

        Args:
            state_info (np.ndarray): Array containing the current state.
        """
        self.state_info = state_info
        self.data_handler.update_states(state_info)

    def update_controls(self, control: np.ndarray) -> None:
        """
        Args:
            control (np.ndarray): The control input to log.
        """
        self.data_handler.update_controls(control)

    def update_time_log(self, time: float) -> None:
        """
        Log the current time.

        Args:
            time (float): The current time.
        """
        self.data_handler.update_time(time)

    def get_info(self,
                 get_as_statevector: bool = False) -> Optional[np.ndarray]:
        """
        Get the current state information.
        args:
            get_as_statevector (bool): If True, returns the state as a StateVector object.

        Returns:
            Optional[np.ndarray]: The current state as a numpy array, or None if not set.

        """
        if self.state_info is None:
            raise ValueError("State information not set.")

        if get_as_statevector:
            state_vector: StateVector = StateVector(
                self.state_info[0],
                self.state_info[1],
                self.state_info[2],
                self.state_info[3],
                self.state_info[4],
                self.state_info[5],
                self.state_info[6]
            )
            return state_vector

        return self.state_info

    def define_states(self) -> None:
        """
        Define the symbolic state variables of the system in the NED frame.

        States include:
            x_f, y_f, z_f: Positions.
            phi_f, theta_f, psi_f: Euler angles.
            v_f: Airspeed.
            p_f, q_f, r_f: Angular rates (roll, pitch, yaw).
        """
        self.x_f: ca.SX = ca.SX.sym('x_f')
        self.y_f: ca.SX = ca.SX.sym('y_f')
        self.z_f: ca.SX = ca.SX.sym('z_f')
        self.phi_f: ca.SX = ca.SX.sym('phi_f')
        self.theta_f: ca.SX = ca.SX.sym('theta_f')
        self.psi_f: ca.SX = ca.SX.sym('psi_f')
        self.v_f: ca.SX = ca.SX.sym('v_f')
        # self.p_f: ca.SX = ca.SX.sym('p')  # roll rate
        # self.q_f: ca.SX = ca.SX.sym('q')  # pitch rate
        # self.r_f: ca.SX = ca.SX.sym('r')  # yaw rate

        self.states: ca.SX = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.phi_f,
            self.theta_f,
            self.psi_f,
            self.v_f,
        )

        self.n_states: int = int(self.states.size()[0])

    def define_controls(self) -> None:
        """
        Define the symbolic control input variables for the system.

        Controls include:
            u_phi, u_theta, u_psi: Attitude commands.
            v_cmd: Airspeed command.
        """
        self.u_phi: ca.SX = ca.SX.sym('u_phi')
        self.u_theta: ca.SX = ca.SX.sym('u_theta')
        self.u_psi: ca.SX = ca.SX.sym('u_psi')
        self.v_cmd: ca.SX = ca.SX.sym('v_cmd')

        self.controls: ca.SX = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls: int = int(self.controls.size()[0])

    def define_wind(self) -> None:
        """
        Define the symbolic wind components in the inertial NED frame.
        """
        self.wind_x: ca.SX = ca.SX.sym('wind_x')
        self.wind_y: ca.SX = ca.SX.sym('wind_y')
        self.wind_z: ca.SX = ca.SX.sym('wind_z')

        self.wind: ca.SX = ca.vertcat(
            self.wind_x,
            self.wind_y,
            self.wind_z
        )

    def set_state_space(self, make_z_positive_up: bool = True) -> None:
        """
        Define the state space of the system and construct the ODE function.
        Args:
            make_z_positive (bool): If True, enforce that the z-coordinate is positive.
            This means the convention becomes NEU (North-East-Up).

        The ODE is defined using the kinematic equations for an aircraft in a NED frame,
        including wind effects and first-order lag dynamics for airspeed and angular rates.

        Frame for these equations are in NED Inertial
        Where x is North, y is East, z is Up or Down
        Positive roll is right wing down
        Positive pitch is nose up
        Positive yaw is CW
        """

        # # Airspeed dynamics (airspeed does not include wind)
        self.v_dot: ca.SX = (self.v_cmd - self.v_f) / self.tau_v
        self.g = 9.81  # m/s^2
        # #body to inertia frame

        self.x_fdot = self.v_f * \
            ca.cos(self.theta_f) * ca.cos(self.psi_f) + self.wind_x
        self.y_fdot = self.v_f * \
            ca.cos(self.theta_f) * ca.sin(self.psi_f) + self.wind_y

        if make_z_positive_up:
            self.z_fdot = self.v_f * ca.sin(self.theta_f) + self.wind_z
        else:
            self.z_fdot = -self.v_f * ca.sin(self.theta_f) + self.wind_z

        # Okay this is weird but what we are going to do is
        # relate our roll to our yaw model, we're assuming that there is some
        # proportional gain that maps the yaw desired to the roll of the aircraft
        yaw_error = wrap_to_pi(self.u_psi - self.psi_f)
        k: float = 0.2
        phi_cmd = k * yaw_error
        # self.phi_fdot: ca.SX = (self.u_phi - self.phi_f) / self.tau_phi
        self.phi_fdot: ca.SX = (phi_cmd - self.phi_f) / self.tau_phi

        # So a positive u_theta means we want the nose to be up
        self.theta_fdot: ca.SX = (
            self.u_theta - self.theta_f) / self.tau_theta
        self.psi_fdot = -self.g * (ca.tan(self.phi_f) / self.v_f)

        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.phi_fdot,
            self.theta_fdot,
            self.psi_fdot,
            self.v_dot
        )

        # Define the ODE function: f(states, controls, wind) -> state derivatives
        self.function: ca.Function = ca.Function(
            'f', [self.states, self.controls, self.wind], [self.z_dot])

    def update_reward(self, reward: float) -> None:
        """
        Update the reward data in the data handler.

        Args:
            reward (float): The reward value to x`log.
        """
        self.data_handler.update_reward(reward)

    def rk45(self, x: ca.SX, u: ca.SX, dt: float, use_numeric: bool = True,
             wind=np.array([0, 0, 0]),
             save_next_step: bool = False) -> np.ndarray:
        """
        Perform one integration step using the 4th order Runge-Kutta (RK45) method.

        Args:
            x (ca.SX): Current state.
            u (ca.SX): Current control input.
            dt (float): Integration time step.
            use_numeric (bool): If True, returns a flattened numpy array; otherwise returns a CasADi expression.
            wind (np.ndarray): Wind vector. Default is [0, 0, 0].
            save_next_step (bool): If True, save the next state in the data handler.
        Returns:
            np.ndarray: Next state as a flattened numpy array if use_numeric is True.
        """
        # check if shape is correct
        if x.shape[0] != self.n_states:
            raise ValueError("input x does not match size of states: ",
                             x.shape[0], self.n_states)

        k1: ca.SX = self.function(x, u, wind)
        k2: ca.SX = self.function(x + dt / 2 * k1, u, wind)
        k3: ca.SX = self.function(x + dt / 2 * k2, u, wind)
        k4: ca.SX = self.function(x + dt * k3, u, wind)
        next_step: ca.SX = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if use_numeric:
            next_step_np: np.ndarray = np.array(next_step).flatten()
            # Wrap the yaw angle to be within [-pi, pi]
            yaw_idx: int = KinematicIndex.YAW.value
            next_step_np[yaw_idx] = (
                next_step_np[yaw_idx] + np.pi) % (2 * np.pi) - np.pi
            if save_next_step:
                self.data_handler.update_states(next_step_np)
                self.data_handler.update_controls(u)
            return next_step_np
        else:
            if save_next_step:
                self.data_handler.update_states(next_step)
            return next_step


class SimpleAgent():
    """
    """
    is_pursuer: bool = None

    def __init__(self,
                 battle_space: BattleSpace,
                 state_vector: StateVector,
                 simple_model: PlaneKinematicModel,
                 agent_id: int = None,
                 radius_bubble: float = 0.0,
                 wind_vector: np.ndarray = np.array([0, 0, 0]),
                 start_time: float = 0.0,
                 is_controlled: bool = False
                 ) -> None:
        self.battle_space: BattleSpace = battle_space
        self.state_vector: StateVector = state_vector
        self.simple_model: PlaneKinematicModel = simple_model
        self.agent_id: int = agent_id
        self.radius_bubble: float = radius_bubble
        self.crashed: bool = False
        self.actions = None
        self.wind_vector: np.ndarray = wind_vector
        self.start_time: float = start_time
        self.is_controlled: bool = is_controlled

        if self.simple_model.state_info is None:
            self.simple_model.state_info = np.array([
                self.state_vector.x,
                self.state_vector.y,
                self.state_vector.z,
                self.state_vector.roll_rad,
                self.state_vector.pitch_rad,
                self.state_vector.yaw_rad,
                self.state_vector.speed,
            ])

        if self.agent_id is None:
            raise ValueError("Agent ID must be provided.")

    def return_data(self) -> DataHandler:
        return self.simple_model.data_handler

    def distance_to(self, other: "SimpleAgent",
                    use_2d: bool = False) -> float:
        if use_2d:
            return self.state_vector.distance_2D(other.state_vector)
        else:
            return self.state_vector.distance_3D(other.state_vector)

    def heading_difference(self, other: "SimpleAgent") -> float:
        return self.state_vector.heading_difference(other.state_vector)

    def is_close_to_parallel(self, other: "SimpleAgent", threshold: float = 0.7) -> bool:
        dot_product = self.state_vector.dot_product_2D(other.state_vector)
        if dot_product > threshold:
            return True

        return False

    def act(self, action: np.ndarray, ctrl_idx: int = 0) -> None:
        """
        Sets the action that will be taken by the agent
        Action inputs are the following:
        - roll (phi)
        - pitch (theta)
        - yaw (psi)
        - velocity (v)
        """
        # check to make sure actions match the plane kinematic model
        if len(action) != self.simple_model.n_controls:
            raise ValueError(
                "Action length does not match the number of controls in \
                    the model. Expected: {}, Received: {}".format(
                    self.simple_model.n_controls, len(action)))

        self.actions = action

    def fall_down(self) -> None:
        """
        Let the agent fall down,
        set everything to be zero
        """
        self.actions = np.zeros(self.simple_model.n_controls)

    def step(self) -> None:
        """
        Step the agent
        """
        if self.crashed:
            self.fall_down()

        # if we haven't initialized the state info
        # We need to initialize it with our state vector information
        # Since the state info has more information than the state vector
        # we will just set the attitude rates to zero
        # refer to the PlaneKinematicModel class for more information
        if self.simple_model.state_info is None:
            self.simple_model.state_info = np.array([
                self.state_vector.x,
                self.state_vector.y,
                self.state_vector.z,
                self.state_vector.phi,
                self.state_vector.theta,
                self.state_vector.psi,
                self.state_vector.v
            ])

        state = self.simple_model.get_info(get_as_statevector=False)
        controls = self.actions
        wind = self.wind_vector
        next_step: np.array = self.simple_model.rk45(x=state, u=controls,
                                                     dt=self.simple_model.dt_val,
                                                     wind=wind, use_numeric=True)
        self.update_states(next_state=next_step,
                           current_controls=controls)

    def update_states(self, next_state: np.array,
                      current_controls: np.array) -> None:
        """
        Args:
            next_state (np.array): The next state of the agent

        Updates the state of the agent as a result of the action
        """
        assert len(next_state) == self.simple_model.n_states

        self.simple_model.update_state_info(next_state)
        self.simple_model.update_controls(current_controls)
        self.state_vector = self.simple_model.get_info(
            get_as_statevector=True)

    def get_state(self) -> np.ndarray:
        """
        Get the observation of the agent
        Which is the state vector
        """
        return self.state_vector.array


class Evader(SimpleAgent):
    is_pursuer: bool = False

    def __init__(self, battle_space: BattleSpace,
                 state_vector: StateVector,
                 simple_model: SimpleAgent,
                 radius_bubble: float,
                 agent_id: int = None,
                 wind_vector: np.ndarray = np.array([0, 0, 0]),
                 start_time: float = 0.0,
                 is_controlled: bool = False) -> None:
        super().__init__(battle_space, state_vector,
                         simple_model, agent_id,
                         radius_bubble, wind_vector, start_time,
                         is_controlled)


class Pursuer(SimpleAgent):
    is_pursuer: bool = False

    def __init__(self, battle_space: BattleSpace,
                 state_vector: StateVector,
                 simple_model: SimpleAgent,
                 radius_bubble: float,
                 agent_id: int = None,
                 wind_vector: np.ndarray = np.array([0, 0, 0]),
                 start_time: float = 0.0,
                 is_controlled: bool = False,
                 capture_radius: float = 10.0) -> None:
        super().__init__(battle_space, state_vector,
                         simple_model, agent_id,
                         radius_bubble, wind_vector, start_time,
                         is_controlled)
        self.capture_radius: float = capture_radius
