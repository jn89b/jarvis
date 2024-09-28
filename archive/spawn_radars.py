"""
Given:
    - Ego position and heading
    - Target location
    - N radars 
    - Radar capabilities 
    - Spawn n radars
"""
import numpy as np
import matplotlib.pyplot as plt

class Radar():
    def __init__(self, 
                 id:int,
                 position:np.ndarray, 
                 fov:float, 
                 range:float):
        self.position = position
        self.fov = fov
        self.range = range
        self.lower_bound = self.lower_bound()
        self.upper_bound = self.upper_bound()
        print("radar heading: ", np.rad2deg(self.position[2]))

    def lower_bound(self) -> np.ndarray:
        x = self.position[0] + self.range*np.cos(self.position[2] - self.fov/2)
        y = self.position[1] + self.range*np.sin(self.position[2] - self.fov/2)
        return np.array([x, y])
    
    def upper_bound(self) -> np.ndarray:
        x = self.position[0] + self.range*np.cos(self.position[2] + self.fov/2)
        y = self.position[1] + self.range*np.sin(self.position[2] + self.fov/2)
        return np.array([x, y])
        

target_position = np.array([0, 500, 0])
ego_position = np.array([0, 0, 0])

n_radars = 10
radar_spawn_distance = 200
radar_fov = np.deg2rad(90)
radar_range = 300
radar_list = []
#okay let's put in the first radar
dx = ego_position[0] - target_position[0] 
dy = ego_position[1] - target_position[1]
los_angle = np.arctan2(dy, dx)

radar_x = target_position[0] + radar_spawn_distance*np.cos(los_angle)
radar_y = target_position[1] + radar_spawn_distance*np.sin(los_angle)
radar_position = np.array([radar_x, radar_y, los_angle])

radar = Radar(
        id = 0,
        position=radar_position, 
        fov=radar_fov, 
        range=radar_range)
    
radar_list.append(radar)

is_first = True
upper_radar = None
lower_radar = None

first_radar = radar_list[0]
is_first = False

# dx_lower = first_radar.lower_bound[0] - first_radar.position[0]
# dy_lower = first_radar.lower_bound[1] - first_radar.position[1]
# theta_lower = np.arctan2(dy_lower, dx_lower)
# radar_lower_x = target_position[0] + radar_spawn_distance*np.cos(theta_lower)
# radar_lower_y = target_position[1] + radar_spawn_distance*np.sin(theta_lower)
# radar_lower_position = np.array([radar_lower_x, radar_lower_y, theta_lower])
# lower_radar = Radar(
#     id = 1,
#     position=radar_lower_position, 
#     fov=radar_fov, 
#     range=radar_range)
        
# dx_lower = first_radar.upper_bound[0] - first_radar.position[0]
# dy_lower = first_radar.upper_bound[1] - first_radar.position[1]
# theta_upper = np.arctan2(dy_lower, dx_lower)
# radar_upper_x = target_position[0] + radar_spawn_distance*np.cos(theta_upper)
# radar_upper_y = target_position[1] + radar_spawn_distance*np.sin(theta_upper)

# radar_upper_position = np.array([radar_upper_x, radar_upper_y, theta_upper])
# upper_radar = Radar(
#     id = 2,
#     position=radar_upper_position,
#     fov=radar_fov,
#     range=radar_range)
# radar_list.append(lower_radar)
# radar_list.append(upper_radar)

left_over_radars = n_radars - len(radar_list)
upper_radars = left_over_radars//2
lower_radars = left_over_radars - upper_radars
print("upper radars: ", upper_radars)
print("lower radars: ", lower_radars)
next_radar = first_radar   
for i in range(upper_radars): 
    dx = next_radar.upper_bound[0] - next_radar.position[0]
    dy = next_radar.upper_bound[1] - next_radar.position[1]
    theta = np.arctan2(dy, dx)
    radar_x = target_position[0] + radar_spawn_distance*np.cos(theta)
    radar_y = target_position[1] + radar_spawn_distance*np.sin(theta)
    radar_position = np.array([radar_x, radar_y, theta])
    radar = Radar(
        id = 3 + i,
        position=radar_position, 
        fov=radar_fov, 
        range=radar_range)
    next_radar = radar
    radar_list.append(radar)
    
next_radar = first_radar
for i in range(lower_radars):
    dx = next_radar.lower_bound[0] - next_radar.position[0]
    dy = next_radar.lower_bound[1] - next_radar.position[1]
    theta = np.arctan2(dy, dx)
    radar_x = target_position[0] + radar_spawn_distance*np.cos(theta)
    radar_y = target_position[1] + radar_spawn_distance*np.sin(theta)
    radar_position = np.array([radar_x, radar_y, theta])
    radar = Radar(
        id = 3 + upper_radars + i,
        position=radar_position, 
        fov=radar_fov, 
        range=radar_range)
    next_radar = radar
    radar_list.append(radar)

#plot the radar and their fov
fig, ax = plt.subplots()
for radar in radar_list:
    #plot as orange circle
    ax.plot(radar.position[0], radar.position[1], 'ro', label='Radar')
    #draw an arrow from the radar position to the upper bound
    lower_x = radar.lower_bound[0]
    lower_y = radar.lower_bound[1]
    
    dx = lower_x - radar.position[0]
    dy = lower_y - radar.position[1]
    ax.arrow(radar.position[0], 
             radar.position[1], 
             dx, dy, 
             head_width=10, head_length=10, fc='k', ec='k')

    upper_x = radar.upper_bound[0]
    upper_y = radar.upper_bound[1]
    dx = upper_x - radar.position[0]
    dy = upper_y - radar.position[1]
    ax.arrow(radar.position[0], 
             radar.position[1], 
             dx, dy, 
             head_width=10, head_length=10, fc='k', ec='k')

    ax.scatter([lower_x, upper_x], [lower_y, upper_y], c='g')
    
ax.plot(target_position[0], target_position[1], 'orange', marker='o', label='Target')
ax.plot(ego_position[0], ego_position[1], 'bo', label='Ego')
#label the ego position
ax.legend()
plt.show()
    