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


def plot(report: Report, cl_sim: CloseLoopSim,
         plt_jsbsim: bool = False) -> None:
    states = report.current_state
    # controls = report.current_control
    time = report.time_dict['time']
    traj = report.state_traj
    idx = 1
    next_state = {}

    for key in traj.keys():
        length = len(traj[key])
        next_state[key] = []
        for i in range(length):
            next_state[key].append(traj[key][i][idx])
        # next_state[key] = traj[key][idx]
        # print(next_state[key])

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(time, states['x'], label='x')
    ax[1].plot(time, states['y'], label='y')
    ax[2].plot(time, states['z'], label='z')

    ax[0].plot(time, next_state['x'], label='u_x', linestyle='--')
    ax[1].plot(time, next_state['y'], label='u_y', linestyle='--')
    ax[2].plot(time, next_state['z'], label='u_z', linestyle='--')

    for a in ax:
        a.legend()

    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    ax[0].plot(time, np.rad2deg(states['phi']), label='phi')
    ax[0].plot(time, np.rad2deg(next_state['phi']),
               label='u_phi', linestyle='--')

    ax[1].plot(time, np.rad2deg(states['theta']), label='theta')
    ax[1].plot(time, np.rad2deg(next_state['theta']),
               label='u_theta', linestyle='--')

    ax[2].plot(time, np.rad2deg(states['psi']), label='psi')
    ax[2].plot(time, np.rad2deg(next_state['psi']),
               label='u_psi', linestyle='--')

    ax[3].plot(time, states['v'], label='v')
    ax[3].plot(time, next_state['v'], label='v_cmd', linestyle='--')

    for a in ax:
        a.legend()

    # plot as 3d trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states['x'], states['y'], states['z'], label='actual')
    ax.plot(GOAL_X, GOAL_Y, GOAL_Z, 'ro', label='goal', color='black')

    if cl_sim.optimizer.use_obs_avoidance:
        for obs in cl_sim.optimizer.obs_params:
            ax.plot([obs.center[0]], [obs.center[1]], [
                    obs.center[2]], 'go', label='obstacle')
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obs.radius * np.outer(np.cos(u), np.sin(v)) + obs.center[0]
            y = obs.radius * np.outer(np.sin(u), np.sin(v)) + obs.center[1]
            z = obs.radius * \
                np.outer(np.ones(np.size(u)), np.cos(v)) + obs.center[2]
            ax.plot_surface(x, y, z, color='b', alpha=0.5)

    pkl.dump(states, open('states.pkl', 'wb'))
    pkl.dump(next_state, open('next_state.pkl', 'wb'))
    pkl.dump(time, open('time.pkl', 'wb'))

    # plot 3d trajectory
    if plt_jsbsim:
        jsb_sim_report = cl_sim.dynamics_adapter.simulator.report
        data_vis = DataVisualizer(jsb_sim_report)
        fig, ax = data_vis.plot_3d_trajectory()
        # plot goal location
        x_final = cl_sim.x_final
        ax.scatter(x_final[0], x_final[1], x_final[2], label='end')
        buffer = 30
        max_x = max(states['x'])
        min_x = min(states['x'])

        max_y = max(states['y'])
        min_y = min(states['y'])

        max_z = max(states['z'])
        min_z = min(states['z'])

        ax.set_xlim([min_x-buffer, max_x+buffer])
        ax.set_ylim([min_y-buffer, max_y+buffer])

        ax.set_zlim([0, 50])

        distance = np.sqrt((x_final[0] - states['x'][-1])**2 +

                           (x_final[1] - states['y'][-1])**2)

    ax.plot(states['x'], states['y'], states['z'],
            label='from mpc', color='g')
    ax.plot(GOAL_X, GOAL_Y, GOAL_Z, 'ro', label='GOAL Location')
    ax.legend()
    plt.show()


GOAL_X = 100
GOAL_Y = 0
GOAL_Z = 30

if __name__ == '__main__':
    x_init = np.array([0, 0, 15, 0, 0, np.deg2rad(0), 15])
    x_final = np.array([GOAL_X, GOAL_Y, GOAL_Z, 0, 0, 0, 30])
    u_0 = np.array([0, 0, 20])

    obstacle_list: List[Obstacle] = []
    obstacle_list.append(Obstacle(center=[GOAL_X, GOAL_Y, GOAL_Z], radius=2))
    plane = JSBPlane(dt_val=1/60)

    # let's define the limits for the states and controls
    control_limits_dict = {
        'u_phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'u_z': {'min': 0, 'max': 50},
        'v_cmd': {'min': 15.0, 'max': 30.0},
    }

    state_limits_dict = {
        'x': {'min': -np.inf, 'max': np.inf},
        'y': {'min': -np.inf, 'max': np.inf},
        'z': {'min': 0, 'max': 50},
        'phi': {'min': -np.deg2rad(55), 'max': np.deg2rad(55)},
        'theta': {'min': -np.deg2rad(25), 'max': np.deg2rad(20)},
        'psi': {'min': np.deg2rad(-360), 'max': np.deg2rad(360)},
        'v': {'min': 12, 'max': 30.0}
    }
    plane.set_control_limits(control_limits_dict)
    plane.set_state_limits(state_limits_dict)

    Q = np.array([1, 1, 1, 0, 0, 0, 0])
    Q = np.diag(Q)
    R = np.array([0, 0, 0])  # np.eye(plane.n_controls)
    R = np.diag(R)

    params = MPCParams(Q=Q, R=R, N=10, dt=1/10)
    mpc = PlaneOmniOptControl(
        mpc_params=params,
        casadi_model=plane,
        use_obs_avoidance=True,
        obs_params=obstacle_list,
        robot_radius=10.0)

    init_cond = AircraftIC(
        x=x_init[0], y=x_init[1], z=x_init[2],
        roll=np.deg2rad(0),
        pitch=np.deg2rad(0),
        yaw=x_init[5],
        airspeed_m=x_init[6])

    sim = SimInterface(
        aircraft_name='x8',
        init_cond=init_cond,
        sim_freq=60,
    )

    x_init = np.array([init_cond.x,
                       init_cond.y,
                       init_cond.z,
                       init_cond.roll,
                       init_cond.pitch,
                       init_cond.yaw,
                       init_cond.airspeed_m])

    def custom_stop_criteria(state: np.ndarray,
                             final_state: np.ndarray) -> bool:
        distance = np.linalg.norm(state[0:2] - final_state[0:2])
        if distance < 1.0:
            return True

    closed_loop_sim = CloseLoopSim(
        optimizer=mpc,
        x_init=x_init,
        x_final=x_final,
        u0=u_0,
        dynamics_adapter=JSBSimAdapter(sim),
        N=3000,
        log_data=True,
        stop_criteria=custom_stop_criteria,
        file_name='jsbsim_sim'
    )
    #
    goal_list = [
        x_final,
        np.array([GOAL_X, GOAL_X, GOAL_Z, 0, 0, 0, 30]),
        np.array([0, 0, GOAL_Z, 0, 0, 0, 30])
    ]
    for i, goal in enumerate(goal_list):
        if i == 1:
            break
        # closed_loop_sim.run_single_step()
        print(goal)
        closed_loop_sim.update_x_final(goal)
        closed_loop_sim.done = False
        closed_loop_sim.run()

report: Report = closed_loop_sim.report
# plot(report, closed_loop_sim, True)
plot(report, closed_loop_sim, True)
