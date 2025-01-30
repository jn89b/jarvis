from typing import Dict, Any, List, Tuple
import tqdm

import casadi as ca
import numpy as np
import json
import matplotlib.pyplot as plt
from jarvis.mpc.optcontrol import PlaneOmniOptControl
from optitraj.models.plane import JSBPlane
from optitraj.close_loop import CloseLoopSim
from optitraj.mpc.PlaneOptControl import Obstacle
from optitraj.utils.data_container import MPCParams
from optitraj.utils.report import Report
from optitraj.dynamics_adapter import JSBSimAdapter

# import aircraftsim
from aircraftsim import (
    SimInterface,
    AircraftIC
)
from aircraftsim import DataVisualizer


def custom_stop_criteria(state: np.ndarray,
                         final_state: np.ndarray) -> bool:
    distance = np.linalg.norm(state[0:2] - final_state[0:2])
    if distance < 1.0:
        return True


class MPCMonteCarlo():
    """
    Used to perform Monte Carlo simulations for the MPC controller for the UAV system
    """

    def __init__(self,
                 num_simulations: int,
                 folder_name: str,
                 control_limits: Dict[str, Tuple[float, float]] = None,
                 state_limits: Dict[str, Tuple[float, float]] = None,
                 randomize_goal: bool = False,
                 randomize_obstacles: bool = False,
                 randomize_start: bool = False,
                 min_distance: float = 100,
                 max_distance: float = 300,
                 jsbsim_freq: int = 60):
        self.num_simulations: int = num_simulations
        self.folder_name: str = folder_name
        self.control_limits: Dict[str, Tuple[float, float]] = control_limits
        self.state_limits: Dict[str, Tuple[float, float]] = state_limits
        self.randomize_goal: bool = randomize_goal
        self.randomize_obstacles: bool = randomize_obstacles
        self.randomize_start: bool = randomize_start
        self.min_distance: float = min_distance
        self.max_distance: float = max_distance
        self.jsbsim_freq: int = jsbsim_freq
        self.plane: JSBPlane = None
        self.mpc_params = None
        self.mpc = None
        self.closed_loop_sim: CloseLoopSim = None
        self.sim: SimInterface = None
        self.closed_loop_sim: CloseLoopSim = None

    def initialize(self):
        """
        Initialize the Monte Carlo simulation
        """
        self.plane = JSBPlane(dt_val=1.0 / self.jsbsim_freq)
        self.plane.set_control_limits(self.control_limits)
        self.plane.set_state_limits(self.state_limits)
        Q = np.array([1, 1, 1, 0, 0, 0, 0])
        Q = np.diag(Q)
        R = np.array([0, 0, 0])  # np.eye(plane.n_controls)
        R = np.diag(R)

        # TODO: This is a dummy obstacle, replace with actual obstacle

        self.mpc_params = MPCParams(Q=Q, R=R, N=10, dt=1/10)

        # # randoimzmize the distance to the goal
        # rand_x = np.random.uniform(-200, 200)
        # rand_y = np.random.uniform(-250, 250)
        # rand_z = np.random.uniform(10, 40)
        # rand_airspeed = np.random.uniform(12, 30)
        # x_init = np.array([rand_x, rand_y, rand_z, 0, 0, 0, rand_airspeed])

        if self.randomize_goal:
            rand_x = np.random.uniform(-400, 400)
            rand_y = np.random.uniform(-400, 400)
            rand_z = np.random.uniform(10, 40)
            rand_airspeed = np.random.uniform(12, 30)
            GOAL_X = rand_x
            GOAL_Y = rand_y
            GOAL_Z = rand_z
            x_final = np.array(
                [rand_x, rand_y, rand_z, 0, 0, 0, rand_airspeed])
            obstacle_list: List[Obstacle] = []
            obstacle_list.append(Obstacle(center=x_final, radius=2))
        else:
            GOAL_X = 100
            GOAL_Y = 100
            GOAL_Z = 30
            obstacle_list: List[Obstacle] = []
            obstacle_list.append(
                Obstacle(center=[GOAL_X, GOAL_Y, GOAL_Z], radius=2))
            x_final = np.array([GOAL_X, GOAL_Y, GOAL_Z, 0, 0, 0, 20])

        random_heading = np.random.uniform(-np.pi, np.pi)
        x_random = np.random.uniform(self.min_distance, self.max_distance) * np.cos(
            random_heading) + GOAL_X
        y_random = np.random.uniform(self.min_distance, self.max_distance) * np.sin(
            random_heading) + GOAL_Y

        x_init = np.array([
            x_random,
            y_random,
            np.random.uniform(10, 40),
            0,
            0,
            -random_heading,
            np.random.uniform(12, 30)])

        self.mpc = PlaneOmniOptControl(
            mpc_params=self.mpc_params,
            casadi_model=self.plane,
            use_obs_avoidance=True,
            obs_params=obstacle_list,
            robot_radius=5.0)

        init_cond = AircraftIC(
            x=x_init[0], y=x_init[1], z=x_init[2],
            roll=np.deg2rad(0),
            pitch=np.deg2rad(0),
            yaw=x_init[5],
            airspeed_m=x_init[6])

        self.sim = SimInterface(
            aircraft_name='x8',
            init_cond=init_cond,
            sim_freq=60,
        )

        # random u_0
        u_0 = np.array([0,
                        0,
                        np.random.uniform(12, 30)])

        self.closed_loop_sim = CloseLoopSim(
            optimizer=self.mpc,
            x_init=x_init,
            x_final=x_final,
            u0=u_0,
            dynamics_adapter=JSBSimAdapter(self.sim),
            N=1500,
            log_data=True,
            stop_criteria=custom_stop_criteria,
            file_name='jsbsim_sim'
        )

    def reset(self):
        """
        Reset the Monte Carlo simulation
        """
        pass

    def save_monte_carlo_results(self, sim_idx: int) -> None:
        """
        Save the results of the Monte Carlo simulation to a file
        """
        report = self.closed_loop_sim.report
        report.file_name = self.folder_name + '/simulation_' + str(sim_idx)
        # report.save_everything()
        goal = self.closed_loop_sim.x_final
        states = report.current_state
        controls = report.current_control

        simulation_data = []
        current_time = 0
        for i in range(len(states['x'])):
            current_time = current_time + 1 / self.jsbsim_freq
            sim_info = {
                "time": current_time,
                "ego": {
                    "x": float(states['x'][i]),
                    "y": float(states['y'][i]),
                    "z": float(states['z'][i]),
                    "phi": float(states['phi'][i]),
                    "theta": float(states['theta'][i]),
                    "psi": float(states['psi'][i]),
                    "v": float(states['v'][i])
                },
                "controls": {
                    "u_phi": float(controls['u_phi'][i]),
                    "u_z": float(controls['u_z'][i]),
                    "v_cmd": float(controls['v_cmd'][i])
                },
                "goal": {
                    "x": float(goal[0]),
                    "y": float(goal[1]),
                    "z": float(goal[2])
                }
            }

            simulation_data.append(sim_info)

        with open(report.file_name + '.json', 'w') as f:
            json.dump(simulation_data, f, indent=4)

        plot(report, self.closed_loop_sim, save_dir=report.file_name + '.png')

        # # save as json file
        # info = {
        #     'goal': goal,
        #     'states': states,
        #     'controls': controls
        # }

        # with open(report.file_name + '.json', 'w') as f:
        #     json.dump(info, f, indent=4)

    def run_monte_carlo_simulation(self) -> Dict[str, Any]:
        """
        Run the Monte Carlo simulation for the MPC controller
        """
        self.closed_loop_sim.run()

    def run_end_to_end_simulation(self) -> Dict[str, Any]:
        """
        Run the end-to-end simulation for the MPC controller
        """
        for i in tqdm.tqdm(range(self.num_simulations)):
            self.initialize()
            # self.run_monte_carlo_simulation()
            self.closed_loop_sim.run()
            self.save_monte_carlo_results(int(i))


def plot(report: Report, cl_sim: CloseLoopSim,
         plt_jsbsim: bool = False,
         save_dir: str = 'traj') -> None:
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

    # fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    # ax[0].plot(time, states['x'], label='x')
    # ax[1].plot(time, states['y'], label='y')
    # ax[2].plot(time, states['z'], label='z')

    # ax[0].plot(time, next_state['x'], label='u_x', linestyle='--')
    # ax[1].plot(time, next_state['y'], label='u_y', linestyle='--')
    # ax[2].plot(time, next_state['z'], label='u_z', linestyle='--')

    # for a in ax:
    #     a.legend()

    # fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    # ax[0].plot(time, np.rad2deg(states['phi']), label='phi')
    # ax[0].plot(time, np.rad2deg(next_state['phi']),
    #            label='u_phi', linestyle='--')

    # ax[1].plot(time, np.rad2deg(states['theta']), label='theta')
    # ax[1].plot(time, np.rad2deg(next_state['theta']),
    #            label='u_theta', linestyle='--')

    # ax[2].plot(time, np.rad2deg(states['psi']), label='psi')
    # ax[2].plot(time, np.rad2deg(next_state['psi']),
    #            label='u_psi', linestyle='--')

    # ax[3].plot(time, states['v'], label='v')
    # ax[3].plot(time, next_state['v'], label='v_cmd', linestyle='--')

    # for a in ax:
    #     a.legend()

    # plot as 3d trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states['x'], states['y'], states['z'], label='actual')
    GOAL_X = cl_sim.x_final[0]
    GOAL_Y = cl_sim.x_final[1]
    GOAL_Z = cl_sim.x_final[2]
    ax.plot(GOAL_X, GOAL_Y, GOAL_Z, 'ro', label='goal', color='black')
    ax.legend()
    # save figure
    plt.savefig(save_dir)

    # plot a 2d trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(states['x'], states['y'], label='actual')
    GOAL_X = cl_sim.x_final[0]
    GOAL_Y = cl_sim.x_final[1]
    ax.plot(GOAL_X, GOAL_Y, 'ro', label='goal', color='black')
    ax.legend()

    # save figure
    plt.savefig(save_dir.replace('.png', '_2d.png'))
