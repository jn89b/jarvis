import numpy as np

def normalize_vector(v):
    """Normalizes a 2D vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def calculate_angle_of_incidence(radar_pos, aircraft_pos, aircraft_heading_deg):
    """
    Calculates the angle of incidence between the radar's line of sight and the aircraft's heading,
    where East is 0 degrees and angles increase counterclockwise.
    
    Parameters:
    radar_pos (tuple): Position of the radar (x_r, y_r).
    aircraft_pos (tuple): Position of the aircraft (x_a, y_a).
    aircraft_heading_deg (float): Aircraft's heading in degrees, with 0 degrees being East.
    
    Returns:
    float: The angle of incidence in degrees.
    """
    # Transform the heading so that 0 degrees is East and angles increase counterclockwise
    transformed_heading_deg = 90 - aircraft_heading_deg
    
    # Convert the transformed heading from degrees to radians
    aircraft_heading_rad = np.radians(transformed_heading_deg)
    
    # Calculate the LOS vector from radar to aircraft
    los_vector = np.array(aircraft_pos) - np.array(radar_pos)
    
    # Normalize the LOS vector
    los_vector_normalized = normalize_vector(los_vector)
    
    # Calculate the heading vector based on the transformed aircraft's heading angle
    heading_vector = np.array([np.cos(aircraft_heading_rad), 
                               np.sin(aircraft_heading_rad)])
    
    # Calculate the dot product of the normalized LOS vector and the heading vector
    dot_product = np.dot(los_vector_normalized, heading_vector)
    
    # Ensure the dot product is within the valid range for acos (to avoid numerical errors)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate the angle of incidence
    angle_of_incidence_rad = np.arccos(dot_product)
    
    # Convert the angle from radians to degrees
    angle_of_incidence_deg = np.degrees(angle_of_incidence_rad)
    
    return angle_of_incidence_deg

# Example usage
radar_position = (0, 0)  # Radar position (x_r, y_r)
aircraft_position = (100, 100)  # Aircraft position (x_a, y_a)
aircraft_heading = 25  # Aircraft heading in degrees, where 0 degrees is East

angle_of_incidence = calculate_angle_of_incidence(radar_position, aircraft_position, aircraft_heading)
print(f"Angle of Incidence: {angle_of_incidence:.2f} degrees")
