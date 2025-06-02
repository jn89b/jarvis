import unittest
import numpy as np
import matplotlib.pyplot as plt

from aircraftsim import AircraftIC, SimInterface
 
from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.jsbsim_agent import Pursuer, Evader, Agent
from jarvis.utils.trainer import load_yaml_config
from jarvis.utils.vector import StateVector
from jarvis.envs.battlespace import BattleSpace
from jarvis.envs.multi_env_jsbsim import PursuerEvaderEnv

from typing import List, Dict, Any
plt.close('all')

def compute_los(target: StateVector, current: StateVector,
                use_deg:bool=True) -> float:
    """
    Compute the line of sight (LOS) angle in degrees between the goal and current state vectors.
    The LOS is computed in the NED (North, East, Down) coordinate system.
    
    Args:
        target (StateVector): The target state vector (goal).
        current (StateVector): The current state vector of the agent.
        
    Returns:
        float: The LOS angle in degrees.
    """
    dx = target.x - current.x
    dy = target.y - current.y
    los_angle = np.arctan2(dy, dx)
    
    if use_deg:
        los_angle = np.degrees(los_angle)
    else:
        los_angle = np.rad2deg(los_angle)
        
    return los_angle


class TestJSBSim(unittest.TestCase):
    """
    Test JSBSIM environment for the following:
    - Control the agents in the environment 
    - Need to make sure the frequency of jsbsim aligns with what I want for timesteps
    for the environment 
    - Test heading control for action masking and basic pursuit is working correctly
    -  
    """    
    def setUp(self):
        # Load the configuration file
        self.config = load_yaml_config(
            "config/jsbsim_env_config.yaml")['battlespace_environment']
        self.engage_env = PursuerEvaderEnv(
            config=self.config)
        self.engage_env.reset()
        self.simulation_config = self.config['simulation']
        self.agent_config = self.config['agents']
        
        
    def test_spawn_agents(self):
        """
        Test to make sure we are spawning our agents correctly
        """
        num_evaders: int = self.config['agents']['num_evaders']
        num_pursuers: int = self.config['agents']['num_pursuers']
        print("Number of evaders: ", num_evaders)
        print("Number of pursuers: ", num_pursuers)
        all_agents: List[int] = self.engage_env.get_all_agents
        print("Number of agents: ", len(all_agents),
              "num_evaders + num_pursuers: ", num_evaders + num_pursuers)
        assert len(all_agents) == num_evaders + num_pursuers
        controlled_agents: List[int] = self.engage_env.get_controlled_agents
        print("controlled_agents: ", controlled_agents)
        
    def test_pro_nav(self) -> None:
        """
        Move the agents in the environment
        
        - Need to be able to make sure that the agents are moving correctly
        - For JSBSIM the heading command is global heading command 
        so if you send 45 degrees it will turn to 45 degrees and it 
        stays there 
        - keep in mind that frame of reference is in NED
        """
        pursuer_agent: Pursuer = self.engage_env.get_pursuer_agents()[0]
        evader_agent: Evader = self.engage_env.get_evader_agents()[0]
        
        evader_x = evader_agent.get_observation()[0]
        evader_y = evader_agent.get_observation()[1]
        evader_z = evader_agent.get_observation()[2]
        evader_state: StateVector = evader_agent.state_vector
        print(f"Evader position: ({evader_x}, {evader_y}, {evader_z})")
                
        n_steps: int = 4000
        x_history: List[float] = []
        y_history: List[float] = []
        z_history: List[float] = []
        for i in range(n_steps):
            # Get the current state of the pursuer
            pursuer_state: StateVector = pursuer_agent.state_vector
            # print(f"Step {i}: {pursuer_state}")
            los_dg:float = compute_los(evader_state, pursuer_state, use_deg=True)
            print(f"Line of Sight (LOS) angle: {los_dg} degrees")
            # set the action for the pursuer agent
            pursuer_agent.act(
                action=np.array([0.0, 0.0, 20.0]),  # roll, pitch, speed
                use_action=False,
                heading_des_dg=-los_dg,  # convert to degrees
                alt_des_m= evader_state.z,
                vel_des_m= 12,
            )
            pursuer_agent.step()
            distance = pursuer_state.distance_3D(evader_state)
            print("distance from pursuer to evader: ", distance)
                       
            x_history.append(pursuer_state.x)
            y_history.append(pursuer_state.y)
            z_history.append(pursuer_state.z)
                       
        # 3D plot
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot(x_history, y_history, z_history, label='Pursuer Path')
        ax.scatter(x_history[0], y_history[0], z_history[0], c='r', label='Start Position')
        ax.scatter(evader_x, evader_y, evader_z, color='red', label='Evader Position')
        
        plt.show()
        
if __name__ == "__main__":
    unittest.main()