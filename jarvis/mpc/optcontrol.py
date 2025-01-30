import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl

from typing import Dict, List, Tuple

from scipy.spatial import distance

import casadi as ca
from optitraj.models.plane import Plane, JSBPlane
from optitraj.close_loop import CloseLoopSim
from optitraj.mpc.PlaneOptControl import Obstacle
from optitraj.utils.data_container import MPCParams
from optitraj.utils.report import Report
from optitraj.dynamics_adapter import JSBSimAdapter
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem

# import dataclasses
from dataclasses import dataclass


@dataclass
class Effector:
    effector_range: float
    effector_angle: float


try:
    # import aircraftsim
    from aircraftsim import (
        SimInterface,
        AircraftIC
    )
    from aircraftsim import DataVisualizer

except ImportError:
    print('aircraftsim not installed')


class PlaneOmniOptControl(OptimalControlProblem):
    """
    Example of a class that inherits from OptimalControlProblem
    for the Plane model using Casadi, can be used for 
    obstacle avoidance
    """

    def __init__(self,
                 mpc_params: MPCParams,
                 casadi_model: CasadiModel,
                 use_obs_avoidance: bool = False,
                 obs_params: List[Obstacle] = None,
                 robot_radius: float = 3.0) -> None:
        super().__init__(mpc_params,
                         casadi_model)

        self.use_obs_avoidance: bool = use_obs_avoidance
        self.obs_params: List[Obstacle] = obs_params
        self.robot_radius: float = robot_radius
        self.radius_target: float = 2.0
        self.effector: Effector = Effector(effector_range=10.0,
                                           effector_angle=math.pi/4)
        if self.use_obs_avoidance:
            print("Using obstacle avoidance")
            self.is_valid_obs_params()
            self.set_obstacle_avoidance_constraints()

    def is_valid_obs_params(self) -> bool:
        """
        To use obstacle avoidance the parameters must be
        a list of Obstacle objects

        """
        if self.obs_params is None:
            raise ValueError("obs_params is None")

    def compute_omni_pew_cost(self) -> ca.SX:
        """compute the cost function for the directional effector"""

        n_states = self.casadi_model.n_states
        target_location = self.P[n_states:]

        total_effector_cost = 0
        effector_cost = 0

        v_max = self.casadi_model.control_limits['v_cmd']['max']
        v_min = self.casadi_model.control_limits['v_cmd']['min']

        k_range = 0.6
        k_elev = 5
        k_azim = 5
        x_idx: int = 0
        y_idx: int = 1
        z_idx: int = 2
        roll_idx: int = 3
        pitch_idx: int = 4
        yaw_idx: int = 5

        v_cmd_idx: int = 2

        for i in range(self.N):
            x_pos = self.X[x_idx, i]
            y_pos = self.X[y_idx, i]
            z_pos = self.X[z_idx, i]
            roll = self.X[roll_idx, i]
            pitch = self.X[pitch_idx, i]
            yaw = self.X[yaw_idx, i]
            v_cmd = self.U[v_cmd_idx, i]

            if i == self.N-1:
                v_cmd_next = self.U[v_cmd_idx, i]
            else:
                v_cmd_next = self.U[v_cmd_idx, i+1]

            ###### DIRECTIONAL EFFECTOR COST FUNCTION MAXIMIZE TIME ON TARGET BY SLOWING DOWN APPROACH######
            # right now this is set up for the directional effector
            dx = target_location[0] - x_pos
            dy = target_location[1] - y_pos
            dz = target_location[2] - z_pos

            dtarget = ca.sqrt((dx)**2 + (dy)**2 + (dz)**2)
            # los_target = ca.vertcat(dx, dy)
            los_target = ca.atan2(dy, dx)

            # slow down cost
            # normal unit vector of the target
            los_hat = ca.vertcat(dx, dy, dz) / dtarget

            # ego unit vector
            u_x = ca.cos(pitch) * ca.cos(yaw)
            u_y = ca.cos(pitch) * ca.sin(yaw)
            u_z = ca.sin(pitch)
            ego_unit_vector = ca.vertcat(u_x, u_y, u_z)

            # dot product of the unit vectors
            dot_product = ca.dot(los_hat, ego_unit_vector)
            abs_dot_product = ca.fabs(dot_product)

            # these exponential functions will be used to account for the distance and angle of the target
            # the closer we are to the target the more the distance factor will be close to 1
            error_dist_factor = ca.exp(-k_range *
                                       dtarget/self.effector.effector_range)
            # error_dist = dtarget/self.Effector.effector_range

            # the closer we are to the target the more the angle factor will be close to 1
            los_theta = ca.atan2(dz, dx)
            error_theta = ca.fabs(los_theta - pitch)
            error_phi = ca.fabs(ca.atan2(dy, dx) - roll)

            ratio_distance = (dtarget/self.effector.effector_range)**2
            ratio_theta = (error_theta/self.effector.effector_angle)**2
            ratio_phi = (error_phi/self.effector.effector_angle)**2

            effector_dmg = (2*ratio_distance + (abs_dot_product))
            # this is time on target
            # this velocity penalty will be used to slow down the vehicle as it gets closer to the target
            quad_v_max = (v_cmd - v_max)**2
            # quad_v_min = (v_cmd - v_min)**2
            quad_v_min = (v_cmd - v_min)**2
            vel_penalty = ca.if_else(error_dist_factor <= 0.005,
                                     quad_v_max, quad_v_min)

            # all other controls except for the velocity
            # controls_cost = ca.sumsqr(self.U[:3, i])

            effector_cost += effector_dmg  # + vel_penalty  # + controls_cost

            # constraint to make sure we don't get too close to the target and crash into it
            # safe_distance = 1.0
            # diff = -dtarget + \
            #     self.radius_target + safe_distance
            # # self.g = ca.vertcat(self.g, diff)

            ###### TOROID EFFECTOR TURN IT INTO A TIME CONSTRAINT FUNCTION ######

        # + time_cost
        # total_effector_cost = 1.0 * \
        #     ca.sum2(effector_cost)

        return ca.sum2(effector_cost)

    def compute_dynamics_cost(self) -> ca.MX:
        """
        Compute the dynamics cost for the optimal control problem
        """
        # initialize the cost
        cost = 0.0
        Q = self.mpc_params.Q
        R = self.mpc_params.R

        x_final = self.P[self.casadi_model.n_states:]

        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            cost += cost \
                + (states - x_final).T @ Q @ (states - x_final) \
                + controls.T @ R @ controls

        return cost

    def set_obstacle_avoidance_constraints(self) -> None:
        """
        Set the obstacle avoidance constraints for the optimal control problem
        """
        x_position = self.X[0, :]
        y_position = self.X[1, :]

        for i, obs in enumerate(self.obs_params):
            obs_center: Tuple[float] = ca.DM(obs.center)
            obs_radius: float = obs.radius
            distance = -ca.sqrt((x_position - obs_center[0])**2 +
                                (y_position - obs_center[1])**2)
            diff = distance + obs_radius + self.robot_radius
            self.g = ca.vertcat(self.g, diff[:-1].T)

    def compute_obstacle_avoidance_cost(self) -> ca.MX:
        """
        Compute the obstacle avoidance cost for the optimal control problem
        We set g to an inequality constraint that satifies the following:
        -distance + radius <= 0 for each obstacle
        """
        cost = 0.0
        x_position = self.X[0, :]
        y_position = self.X[1, :]

        for i, obs in enumerate(self.obs_params):
            obs_center: Tuple[float] = ca.DM(obs.center)
            obs_radius: float = obs.radius
            distance: float = -ca.sqrt((x_position - obs_center[0])**2 +
                                       (y_position - obs_center[1])**2)
            diff = distance + obs_radius + self.robot_radius
            cost += ca.sum1(diff[:-1].T)

        return 10*cost

    def compute_total_cost(self) -> ca.MX:
        # cost = self.compute_dynamics_cost()
        cost = self.compute_omni_pew_cost()
        # cost = cost + self.compute_obstacle_avoidance_cost()
        return cost

    def solve(self, x0: np.ndarray, xF: np.ndarray, u0: np.ndarray) -> np.ndarray:
        """
        Solve the optimal control problem for the given initial state and control

        """
        state_init = ca.DM(x0)
        state_final = ca.DM(xF)

        X0 = ca.repmat(state_init, 1, self.N+1)
        U0 = ca.repmat(u0, 1, self.N)

        n_states = self.casadi_model.n_states
        n_controls = self.casadi_model.n_controls
        # self.compute_obstacle_avoidance_cost()

        if self.use_obs_avoidance and self.obs_params is not None:
            # set the obstacle avoidance constraints
            num_obstacles = len(self.obs_params)  # + 1
            num_obstacle_constraints = num_obstacles * (self.N)
            # Constraints for lower and upp bound for state constraints
            # First handle state constraints
            lbg_states = ca.DM.zeros((n_states*(self.N+1), 1))
            ubg_states = ca.DM.zeros((n_states*(self.N+1), 1))

            # Now handle the obstacle avoidance constraints and add them at the bottom
            # Obstacles' lower bound constraints (-inf)
            # this is set up where -distance + radius <= 0
            lbg_obs = ca.DM.zeros((num_obstacle_constraints, 1))
            lbg_obs[:] = -ca.inf
            ubg_obs = ca.DM.zeros((num_obstacle_constraints, 1))
            # Concatenate state constraints and obstacle constraints (state constraints come first)
            # Concatenate state constraints and then obstacle constraints
            lbg = ca.vertcat(lbg_states, lbg_obs)
            ubg = ca.vertcat(ubg_states, ubg_obs)  # Same for the upper bounds
        else:
            num_constraints = n_states*(self.N+1)
            lbg = ca.DM.zeros((num_constraints, 1))
            ubg = ca.DM.zeros((num_constraints, 1))

        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': self.pack_variables_fn(**self.lbx)['flat'],
            'ubx': self.pack_variables_fn(**self.ubx)['flat'],
        }
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_final   # target state
        )

        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(self.N+1), 1),
            ca.reshape(U0, n_controls*self.N, 1)
        )
        # init_time = time.time()
        solution = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        return solution
