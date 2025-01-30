import numpy as np
from jarvis.utils.monte_carlo import MPCMonteCarlo


if __name__ == "__main__":
    num_simulations = 5
    folder_name = "mpc_data"
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
    monte_carlo = MPCMonteCarlo(
        control_limits=control_limits_dict,
        state_limits=state_limits_dict,
        num_simulations=num_simulations, folder_name=folder_name)
    monte_carlo.run_end_to_end_simulation()
