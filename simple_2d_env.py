import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
import math
from typing import Any, SupportsFloat
from gymnasium import spaces # https://www.gymlibrary.dev/api/spaces/
from stable_baselines3 import PPO

# https://github.com/asack20/RL-in-Pursuit-Evasion-Game/blob/master/src/robot.py

# https://www.gymlibrary.dev/api/core/


"""
States will be x,y,theta and mapped  to a continous space

Actions will consist of linear and angular velocity:
    - [v_min , v_max] and [w_min, w_max] in discrete steps
"""

import numpy as np


class GameDrawer:
    """
    helper class to help draw the game
    """
    def __init__(self) -> None:
        pass
    
    def draw_arrow(self, position, angle_deg):
        arrow_length = 20
        head_length = 5
        head_width = 10
        end = (position[0] + arrow_length * math.cos(math.radians(angle_deg)), 
               position[1] + arrow_length * math.sin(math.radians(angle_deg)))
        right_side = (end[0] + head_length * math.sin(math.radians(angle_deg)), 
                      end[1] - head_length * math.cos(math.radians(angle_deg)))
        left_side = (end[0] - head_length * math.sin(math.radians(angle_deg)), 
                     end[1] + head_length * math.cos(math.radians(angle_deg)))


        # pygame.draw.line(self.screen, (0, 0, 255), position, end, 5)
        # pygame.draw.polygon(self.screen, (0, 0, 255), [end, right_side, left_side])


        return [end, right_side, left_side]

class Agent:
    def __init__(self, init_states:np.ndarray, agent_params:dict) -> None:
        self.x = init_states[0]
        self.y = init_states[1]
        self.psi = init_states[2]
        self.current_state = init_states
        self.start_state = init_states
        self.agent_params = agent_params
        
        self.v_min = agent_params["v_min"]
        self.v_max = agent_params["v_max"]
        
        self.w_min = agent_params["w_min"]
        self.w_max = agent_params["w_max"]
        
        self.goal_x = agent_params["goal_x"]
        self.goal_y = agent_params["goal_y"]
        
        self.min_x = agent_params["min_x"]
        self.max_x = agent_params["max_x"]
        
        self.max_y = agent_params["max_y"]
        self.min_y = agent_params["min_y"]
        
        self.min_psi = agent_params["min_psi"]
        self.max_psi = agent_params["max_psi"]
        
        
    def move(self, action: np.ndarray) -> None:
        """Move the agent according to the action"""
        new_x = self.x + (action[0]*np.cos(self.psi))
        new_y = self.y + (action[0]*np.sin(self.psi))
        new_psi = self.psi + action[1]
        #new_psi = action[1]
        
        self.x = new_x
        self.y = new_y
        self.psi = new_psi
        
        #wrap angle between -pi and pi
        if self.psi > np.pi:
            self.psi = self.psi - 2*np.pi
        elif self.psi < -np.pi:
            self.psi = self.psi + 2*np.pi
        
        self.current_state = np.array([self.x, self.y, self.psi])

        
        return self.current_state
        
    def get_state(self) -> np.ndarray:
        """returns the current state of the agent"""
        return self.current_state
    
    
    def reset(self, set_random:bool=False, new_start:np.ndarray=None) -> None:
        """reset the agent to the start state"""
        if set_random and new_start is not None:
            self.start_state = new_start
            self.x = new_start[0]
            self.y = new_start[1]
            self.psi = new_start[2]
            self.current_state = new_start
        else:
            self.x = self.start_state[0]
            self.y = self.start_state[1]
            self.psi = self.start_state[2]
            self.current_state = self.start_state
        return self.current_state



#this will be abstracted to an Agent class

class MissionGym(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 7}

    def __init__(self, evader:Agent, num_discrete_actions:int=10, 
                 render_mode:str=None, render_fps:int=7, use_random_start:bool=True):
        super(MissionGym, self).__init__()
            
        # observation space of the evader
        self.v_range = [evader.v_min, evader.v_max] #velocity range
        self.w_range = [evader.w_min, evader.w_max] #angular velocity range
        self.v_space = np.linspace(evader.v_min, evader.v_max, num_discrete_actions)
        self.w_space = np.linspace(evader.w_min, evader.w_max, num_discrete_actions)
        
        self.evader = evader

        # action space of the evader
        self.total_actions = num_discrete_actions**2
        # self.action_space = spaces.Discrete(self.total_actions)
        # self.action_space = spaces.Box(low=np.array([evader.v_min, evader.w_min]),
        #                                high=np.array([evader.v_max, evader.w_max]),
        #                                dtype=np.float32)
        self.velocity_commands = np.linspace(evader.v_min, evader.v_max, num_discrete_actions)
        self.heading_commands = np.linspace(-np.pi/4, np.pi/4, num_discrete_actions)
        self.action_space = spaces.MultiDiscrete([
            num_discrete_actions, num_discrete_actions])
        
        # self.action_space = spaces.Box(low=np.array([-1, -1]),
        #                                high=np.array([1,  1]),
        #                                dtype=np.float32)
        
        #map action space to combination of linear and angular velocity
        # self.action_map = {}
        # for i in range(num_discrete_actions):
        #     for j in range(num_discrete_actions):
        #         self.action_map[i*num_discrete_actions + j] = [self.v_space[i], self.w_space[j]]
        
    
        self.goal_location = np.array([evader.goal_x, evader.goal_y])
        distance_to_goal = np.linalg.norm(evader.get_state()[:2] - self.goal_location)
    
        # observation space
        self.evader_observation_space = spaces.Box(low=np.array([evader.min_x, evader.min_y, evader.min_psi, 0]), 
                                                   high=np.array([evader.max_x, evader.max_y, evader.max_psi, 1000.0]), 
                                                   dtype=np.float32)
        
        self.observation_space = spaces.Dict(
            {
            "evader": self.evader_observation_space,
            }
        )
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.time_limit_constant = 1000
        self.time_limit = self.time_limit_constant
        self.game_window = None 
        self.clock = None
        self.buffer = 200
        self.game_renderer = GameDrawer() 
        self.width = 1000 + self.buffer  #int(abs(evader.max_x - evader.min_x))
        self.height = 1000 + self.buffer#int(abs(evader.max_y - evader.min_y))
        self.render_fps = render_fps
        
        self.old_distance = self.compute_distance_cost(self.evader.get_state())
        self.use_random_start = use_random_start
        
    def print_info(self):
        print("observation space: ", self.observation_space)
        print("action space: ", self.action_space)
        print("evader observation space: ", self.observation_space["evader"])
        print("evader action space: ", self.action_space)
        print("goal location: ", self.goal_location)
        print("evader: ", self.evader)
        
    
    def scale_action(self, action: np.ndarray):
        """scale the action space based on the min and max values"""
        velocity_normalized = action[0]
        angular_velocity_normalized = action[1]
        velocity = self.evader.v_min + (self.evader.v_max - self.evader.v_min) * (velocity_normalized + 1)/2
        angular_velocity = self.evader.w_min + (self.evader.w_max - self.evader.w_min) * (angular_velocity_normalized + 1)/2
        
        # #wrap angular velocity between -pi and pi
        if angular_velocity > np.pi:
            angular_velocity = angular_velocity - 2*np.pi
        elif angular_velocity < -np.pi:
            angular_velocity = angular_velocity + 2*np.pi
            
        return np.array([velocity, angular_velocity])
    
    def __get_observation(self) -> dict:
        position = self.evader.get_state()
        relative_position = self.goal_location - position[:2]
        distance = np.linalg.norm(position[:2] - self.goal_location)
        return {"evader": np.array([relative_position[0], relative_position[1], position[2], distance])}
    
    def __get_info(self) -> dict[str, Any]:
        agent_location = self.evader.get_state()
        distance = np.linalg.norm(agent_location[:2] - self.goal_location)
        info_dict = {"distance": distance}
        return info_dict
    
    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        
        self.time_limit -= 1
        #map the action to the linear and angular velocity
        velocity_idx = action[0]
        velocity = self.velocity_commands[velocity_idx]
        angular_velocity_idx = action[1]
        angular_velocity = self.heading_commands[angular_velocity_idx]
        
        # velocity = action[0]
        # angular_velocity = action[1]
        
        velocity, angular_velocity = self.scale_action(action)
        self.state = self.evader.move(np.array([velocity, angular_velocity]))
        
        reward = 0
        done = False

        # #penalize if the agent goes out of bounds
        if self.state[0] < self.evader.min_x or self.state[0] > self.evader.max_x:
            reward += -10
            done = True
        if self.state[1] < self.evader.min_y or self.state[1] > self.evader.max_y:
            reward += -10
            done = True

        if self.time_limit <= 0:
            done = True
            reward += -1000
            
        self.state[0] = np.clip(self.state[0], self.evader.min_x, self.evader.max_x)
        self.state[1] = np.clip(self.state[1], self.evader.min_y, self.evader.max_y)
        self.state[2] = np.clip(self.state[2], self.evader.min_psi, self.evader.max_psi)
        
        distance = self.compute_distance_cost(self.state)
        delta_distance = distance - self.old_distance
        
        #compute the los angle between the agent and the goal
        dx = self.goal_location[0] - self.state[0]
        dy = self.goal_location[1] - self.state[1]
        #compute the angle between the agent and the goal
        los_goal = np.arctan2(dy, dx)
        
        #get current unit vector of the agent
        agent_unit_vector = np.array([np.cos(self.state[2]), np.sin(self.state[2])])
        #get the unit vector of the line of sight
        los_unit_vector = np.array([np.cos(los_goal), np.sin(los_goal)])
        dot_product = np.dot(agent_unit_vector, los_unit_vector)

        reward += dot_product
        
        #reward += -abs(los_goal - self.state[2])
        # print("reward: ", reward)
        # reward += -distance
        
        if distance < 5:
            print("goal reached", distance, self.state)
            reward += 1000
            done = True


        info = self.__get_info()
        observation =  self.__get_observation()
        
        self.old_distance = distance
        
        return observation, reward, done, False, info

    def compute_distance_cost(self, state: np.ndarray) -> float:
        dx = self.goal_location[0] - state[0]
        dy = self.goal_location[1] - state[1]
        distance = np.sqrt(dx**2 + dy**2)
        return distance

    def reset(self, seed=None, buffer:float=5) -> Any:
        #reset the evader to the start state
        #set seed to random number generator
        if seed is not None:
            #random start between 0 and 150            
            start_x = np.random.uniform(self.evader.min_x + buffer, 
                                        self.evader.max_x - buffer)
            
            start_y = np.random.uniform(self.evader.min_y + buffer, 
                                        self.evader.max_y - buffer)
            
            start_psi = np.random.uniform(self.evader.min_psi, self.evader.max_psi)
            self.evader.reset(set_random=True, new_start=np.array([start_x, start_y, start_psi]))
            self.state = self.evader.get_state()
            # self.evader = Agent(np.array([start_x, start_y, start_psi]), self.evader.agent_params)
        elif self.use_random_start and seed is None:       
            #completely random start
            start_x = np.random.uniform(self.evader.min_x + buffer, 
                                        self.evader.max_x - buffer)
            start_y = np.random.uniform(self.evader.min_y + buffer,
                                        self.evader.max_y - buffer)
            start_psi = np.random.uniform(self.evader.min_psi, self.evader.max_psi)
            self.evader.reset(set_random=True, new_start=np.array([start_x, start_y, start_psi]))
            self.state = self.evader.get_state()
            
        else:
            self.evader.reset(self.use_random_start)
            self.state = self.evader.get_state()
            
        
        observation = self.__get_observation()
        info = self.__get_info()
        
        if self.render_mode == "human":
            self.__render_frame()
            
        self.time_limit = self.time_limit_constant

        return observation, info
        
    
    def render(self, mode: str = 'pass') -> None:
        if self.render_mode == "human":
            return self.__render_frame()
        
    def __render_frame(self):
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py 
        if self.game_window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.game_window = pygame.display.set_mode((self.width, self.height))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # do your drawing here
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((255, 255, 255))
        
        [end, right_side, left_side] = self.game_renderer.draw_arrow(self.evader.current_state[:2], 
                                                                     np.rad2deg(self.evader.current_state[2]))
        
        # print("evader state: ", self.evader.current_state[:2])
        pygame.draw.line(canvas, (0, 0, 255), self.evader.current_state[:2], end, 5)
        pygame.draw.polygon(canvas, (0, 0, 255), [end, right_side, left_side])
        
        #draw the goal location
        pygame.draw.circle(canvas, (255, 0, 0), self.goal_location.astype(int), 10)
        
        if self.render_mode == "human":
            # pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            self.game_window.blit(canvas, canvas.get_rect())

            # pygame.display.flip()
            # # The following line copies our drawings from `canvas` to the visible window
            pygame.event.pump()
            pygame.display.update()

            # # We need to ensure that human-rendering occurs at the predefined framerate.
            # # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            
    
    def close(self) -> None:
        if self.game_window is not None:
            pygame.display.quit()
            pygame.quit()
            self.game_window = None
            self.clock = None
        
        
if __name__ == "__main__":
    agent = Agent(np.array([0, 0, 0]),
                    {"v_min": 3, "v_max": 10, "w_min": -np.pi/4, "w_max": np.pi/4,
                     "goal_x": 50, "goal_y": 50, "min_x": -100, "max_x": 100,
                     "min_y": -100, "max_y": 100, "min_psi": -np.pi, "max_psi": np.pi})
    
    env = MissionGym(agent, num_discrete_actions=10, render_mode="human", render_fps=7)
    # env.print_info()
    
    # obs, info = env.reset()
    
    # #train the agent with PPO
    # model = PPO('MultiInputPolicy', env, verbose=1, 
    #             tensorboard_log="./logs/mission_gym",
    #             ent_coef=0.01)
    
    # #save checkpoints
    # from stable_baselines3.common.callbacks import CheckpointCallback
    # checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/mission_gym', name_prefix="mission_gym")
    # model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    # #save the model
    # model.save("mission_gym")
    
    # load the model
    model = PPO.load("mission_gym")
    
    #evaluate the model
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        print("obs: ", obs)
        env.render()
        
        