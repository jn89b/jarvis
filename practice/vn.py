import numpy as np
import matplotlib.pyplot as plt

# X8 Skywalker parameters
rho = 1.225  # kg/m^3
S = 0.8      # m^2 (80 dm^2) citeturn0search0
mass = 3.0   # kg (max takeoff weight ~2.5-3.0 kg) citeturn0search0
W = mass * 9.81  # N
CL_max = 1.2  # approximate max lift coefficient

# Calculate 1g stall speed V_S1
V_S1 = np.sqrt(2 * W / (rho * S * CL_max))

# Structural load limits (assumed ±3g, ±1.5g)
n_pos = 3.0
n_neg = -1.5

# Speed range: from 0.8*V_S1 to dive speed V_D ~23.6 m/s (85 km/h) citeturn1search0
V_min = 0.8 * V_S1
V_D = 23.6
V = np.linspace(V_min, V_D, 400)

# Aerodynamic envelope (stall limits)
K = rho * S * CL_max / (2 * W)
n_stall_pos = K * V**2
n_stall_neg = -K * V**2

fig,ax = plt.subplots()
ax.plot(V, n_stall_pos, 'b--', label='Positive stall limit')
ax.plot(V, n_stall_neg, 'b--', label='Negative stall limit')
ax.axhline(n_pos,label='Positive load limit')
ax.axhline(n_neg, label='Negative load limit')
ax.set_xlabel('Airspeed V (m/s)')
ax.set_ylabel('Load factor n')
ax.set_xlim(V.min(), V.max())
ax.set_ylim(n_neg-0.5, n_pos+0.5)
ax.grid(True)
ax.legend()

def compute_lateral_acceleration(use_body:bool=False,
                                 body_accl_y:float = None,
                                 vel:float = None,
                                 psi_dot:float = None) -> float:
    """
    Args:
        use_body (bool): If True, use body frame acceleration to compute lateral acceleration.
        body_accl_y (float): Lateral acceleration in the body frame (m/s^2).
        vel (float): Velocity of the aircraft (m/s).
        psi_dot (float): Yaw rate (rad/s).
    Returns:
        float: Lateral acceleration (m/s^2).
        
    """
    if use_body:
        # Use body frame
        # Assuming body_accl_y is the lateral acceleration in the body frame
        if body_accl_y is None:
            raise ValueError("body_accl_y must be provided when use_body is True")
        return body_accl_y
    else:
        # Use inertial frame
        # Assuming vel is the velocity of the aircraft and psi_dot is the yaw rate
        if vel is None or psi_dot is None:
            raise ValueError("vel and psi_dot must be provided when use_body is False")
        return vel * psi_dot
    
def compute_bank_angle(lat_accl:float) -> float:
    """
    Args:
        lat_accl (float): Lateral acceleration (m/s^2).
        
    Returns:
        float: Bank angle in radians.
        
    """
    # Assuming g = 9.81 m/s^2
    g = 9.81
    if abs(lat_accl) < 1e-6:
        lat_accl = 1e-6
    return np.arctan(lat_accl / g)


def compute_load_factor(use_body:bool=False, 
                        body_accl_z:float = None,
                        phi_rad:float = None) -> float:
    """
    Args:
        use_body (bool): If True, use body frame acceleration to compute load factor.
        body_accl_z (float): Vertical acceleration in the body frame (m/s^2).
        phi_rad (float): Roll angle in radians.
        
    Returns:
        float: Load factor (n).
        
    https://www.aeroclass.org/load-factor-in-aviation/
    """
    if use_body:
        # Use body frame
        # Assuming body_accl_z is the vertical acceleration in the body frame
        if body_accl_z is None:
            raise ValueError("body_accl_z must be provided when use_body is True")
        load_factor:float = body_accl_z / 9.81
        return load_factor
    else:
        # assume coordinated turn so use simple trigonometry
        # make sure no vid
        if phi_rad is None:
            raise ValueError("phi_rad must be provided when use_body is False")
        cos_phi = np.cos(phi_rad)
        # make sure cnp.cos(phi_rad) is not zero
        if abs(cos_phi) < 1e-6:
            cos_phi = 1e-6
        load_factor:float = 1 / np.cos(phi_rad)
        
        return load_factor

"""
We need to map the high level action of Velocity, dz, and desired
"""
# vel_data = np.arange(10, 20, 0.2)
# #bank_angle = bank_angle_data = np.radians(np.arange(-30, 30, 1))  # bank angle in radians
# bank_angle_data = compute_bank_angle(vel_data)
# load_factor = np.array([compute_load_factor(use_body=False, phi_rad=phi) for phi in bank_angle])
# ax.scatter(vel_data, load_factor, c='r', label='Load factor from bank angle')
vel_data = np.arange(10, 20, 0.2)
psi_dot = np.radians(np.arange(-30, 30, 1))  # yaw rate in radians/s
for i in range(len(vel_data)):
    for j in range(len(psi_dot)):
        # Compute the lateral acceleration for each velocity and yaw rate
        lat_accl = compute_lateral_acceleration(use_body=False, vel=vel_data[i], psi_dot=psi_dot[j])
        # Compute the bank angle for each lateral acceleration
        bank_angle = compute_bank_angle(lat_accl)
        # Compute the load factor for each bank angle
        load_factor = compute_load_factor(use_body=False, phi_rad=bank_angle)
        # Plot the load factor against velocity
        ax.scatter(vel_data[i], load_factor, c='b', label='Load factor from lateral acceleration' if i == 0 else "")
    

plt.show()
