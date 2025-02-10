from dataclasses import dataclass, field
import numpy as np

X_IDX: int = 0
Y_IDX: int = 1
Z_IDX: int = 2
ROLL_IDX: int = 3
PITCH_IDX: int = 4
YAW_IDX: int = 5
SPEED_IDX: int = 6
VX_IDX: int = 7
VY_IDX: int = 8
VZ_IDX: int = 9


class StateVector:
    """
    Represents the state of an object in 3D space, including its position, orientation, and velocity.

    The state is defined by:
      - Position (x, y, z) in meters.
      - Orientation (roll, pitch, yaw) in radians.
      - Scalar speed in meters per second, from which the velocity components are computed.

    Upon initialization, the class computes the velocity components (vx, vy, vz) based on the
    given speed, yaw, and pitch angles. The velocity components are calculated assuming a simple
    conversion from spherical to Cartesian coordinates:
        vx = speed * cos(yaw) * cos(pitch)
        vy = speed * sin(yaw) * cos(pitch)
        vz = speed * sin(pitch)

    Attributes:
        x (float): The x-coordinate in meters.
        y (float): The y-coordinate in meters.
        z (float): The z-coordinate in meters.
        roll_rad (float): The roll angle in radians.
        pitch_rad (float): The pitch angle in radians.
        yaw_rad (float): The yaw angle in radians.
        speed (float): The scalar speed in meters per second.
        vx (float): The x-component of the velocity (computed).
        vy (float): The y-component of the velocity (computed).
        vz (float): The z-component of the velocity (computed).

    Methods:
        compute_speed_components():
            Computes and updates the velocity components (vx, vy, vz) based on the current speed,
            yaw_rad, and pitch_rad.

        array (property):
            Returns a NumPy array representing the state vector. The array concatenates the position,
            orientation, speed, and velocity components.

        update(x, y, z, roll_rad, pitch_rad, yaw_rad, speed):
            Updates the state vector attributes with the provided values and recomputes the velocity
            components. Only the parameters provided (non-None) are updated.

        __add__(other):
            Defines addition of two StateVector objects. The operation is performed element-wise
            on their underlying array representations.

        __sub__(other):
            Defines subtraction between two StateVector objects. The operation is performed element-wise
            on their underlying array representations.

        __mul__(other):
            Implements scalar multiplication on the StateVector. The operation is applied element-wise
            on the underlying array representation.

        distance_2D(other):
            Computes the Euclidean distance between the 2D positions (x, y) of this state vector and
            another StateVector.

        distance_3D(other):
            Computes the Euclidean distance between the 3D positions (x, y, z) of this state vector and
            another StateVector.

        unit_vector_2D():
            Returns a new StateVector representing the unit vector in the 2D plane (x, y) in the direction
            of the yaw angle. The other attributes (z, orientation, speed, velocity components) are set to 0.

        dot_product_2D(other):
            Calculates the dot product between the 2D unit vectors (derived from the yaw angle) of this
            StateVector and another.

        dot_product_3D(other):
            Calculates the dot product between the complete state vectors of this and another StateVector.

        cross_product_3D(other):
            Computes the cross product of the 3D vectors represented by this and another StateVector.
            The resulting StateVector contains the cross product in its (x, y, z) components, while the
            orientation and speed attributes are set to 0.

        heading_difference(other):
            Calculates the difference in heading (yaw angle) between this StateVector and another,
            normalizing the result to the range [-π, π].

    Units:
        All distance measurements are in meters, and all angular measurements are in radians.
    """

    def __init__(self, x: float, y: float, z: float,
                 roll_rad: float, pitch_rad: float, yaw_rad: float,
                 speed: float):
        self.x = x
        self.y = y
        self.z = z
        self.roll_rad = roll_rad
        self.pitch_rad = pitch_rad
        self.yaw_rad = yaw_rad
        self.speed = speed
        self.compute_speed_components()

    def compute_speed_components(self):
        """Compute the velocity components (vx, vy, vz) from the current speed, yaw, and pitch angles."""
        self.vx = self.speed * np.cos(self.yaw_rad) * np.cos(self.pitch_rad)
        self.vy = self.speed * np.sin(self.yaw_rad) * np.cos(self.pitch_rad)
        self.vz = self.speed * np.sin(self.pitch_rad)

    @property
    def array(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: A NumPy array containing the state vector in the following order:
                        [x, y, z, roll_rad, pitch_rad, yaw_rad, speed, vx, vy, vz].
        """
        return np.array([self.x, self.y, self.z,
                         self.roll_rad, self.pitch_rad,
                         self.yaw_rad,
                         self.speed,
                         self.vx, self.vy, self.vz])

    def update(self, x=None, y=None, z=None,
               roll_rad=None, pitch_rad=None,
               yaw_rad=None, speed=None) -> None:
        """
        Update the state vector attributes with new values and recompute the velocity components.

        Only the parameters that are not None will be updated.

        Parameters:
            x (float, optional): New x-coordinate.
            y (float, optional): New y-coordinate.
            z (float, optional): New z-coordinate.
            roll_rad (float, optional): New roll angle in radians.
            pitch_rad (float, optional): New pitch angle in radians.
            yaw_rad (float, optional): New yaw angle in radians.
            speed (float, optional): New speed in meters per second.
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        if roll_rad is not None:
            self.roll_rad = roll_rad
        if pitch_rad is not None:
            self.pitch_rad = pitch_rad
        if yaw_rad is not None:
            self.yaw_rad = yaw_rad
        if speed is not None:
            self.speed = speed
        self.compute_speed_components()

    def __add__(self, other: "StateVector") -> "StateVector":
        """
        Add two StateVector objects element-wise.

        Parameters:
            other (StateVector): The state vector to add.

        Returns:
            StateVector: A new StateVector representing the element-wise sum.
        """
        value = self.array + other.array
        # Assuming X_IDX and SPEED_IDX are defined indices for slicing the relevant values.
        return StateVector(*value[X_IDX:SPEED_IDX+1])

    def __sub__(self, other: "StateVector") -> "StateVector":
        """
        Subtract another StateVector from this one element-wise.

        Parameters:
            other (StateVector): The state vector to subtract.

        Returns:
            StateVector: A new StateVector representing the element-wise difference.
        """
        value = self.array - other.array
        return StateVector(*value[X_IDX:SPEED_IDX+1])

    def __mul__(self, other: float) -> "StateVector":
        """
        Multiply the state vector by a scalar.

        Parameters:
            other (float): The scalar multiplier.

        Returns:
            StateVector: A new StateVector with each element scaled by the given value.
        """
        value = self.array * other
        return StateVector(*value[X_IDX:SPEED_IDX])

    def __repr__(self) -> str:
        """
        Returns:
            str: A string representation of the StateVector.
        """
        return (f"StateVector({self.x}, {self.y}, {self.z}, "
                f"{self.roll_rad}, {self.pitch_rad}, {self.yaw_rad}, {self.speed})")

    def distance_2D(self, other: "StateVector") -> float:
        """
        Compute the Euclidean distance between this state vector and another in the 2D plane (x, y).

        Parameters:
            other (StateVector): The other state vector.

        Returns:
            float: The 2D distance between the two state vectors.
        """
        position_vector = self - other
        distance = np.linalg.norm(position_vector.array[:2])
        return distance

    def distance_3D(self, other: "StateVector") -> float:
        """
        Compute the Euclidean distance between this state vector and another in 3D space.

        Parameters:
            other (StateVector): The other state vector.

        Returns:
            float: The 3D distance between the two state vectors.
        """
        position_vector = self - other
        distance = np.linalg.norm(position_vector.array[:3])
        return distance

    def unit_vector_2D(self) -> "StateVector":
        """
        Compute the unit vector in the 2D plane (x, y) based on the current yaw angle.

        Returns:
            StateVector: A new StateVector representing the 2D unit vector in the direction
                         of the yaw angle. The z-coordinate, orientation, and speed are set to 0.
        """
        yaw = self.yaw_rad
        unit_vector = np.array([np.cos(yaw), np.sin(yaw)])
        return StateVector(unit_vector[0], unit_vector[1], 0, 0, 0, 0, 0)

    def dot_product_2D(self, other: "StateVector") -> float:
        """
        Compute the dot product between the 2D unit vectors of this state vector and another.

        Parameters:
            other (StateVector): The other state vector.

        Returns:
            float: The dot product of the two 2D unit vectors.
        """
        unit_vector = self.unit_vector_2D()
        other_unit_vector = other.unit_vector_2D()
        dot_product = np.dot(
            unit_vector.array[:2], other_unit_vector.array[:2])
        return dot_product

    def dot_product_3D(self, other: "StateVector") -> float:
        """
        Compute the dot product between the full state vector arrays of this and another StateVector.

        Parameters:
            other (StateVector): The other state vector.

        Returns:
            float: The dot product of the two state vectors.
        """
        dot_product = np.dot(self.array, other.array)
        return dot_product

    def cross_product_3D(self, other: "StateVector") -> "StateVector":
        """
        Compute the cross product of the 3D vectors represented by this and another StateVector.

        Parameters:
            other (StateVector): The other state vector.

        Returns:
            StateVector: A new StateVector representing the cross product, with its position (x, y, z)
                         set to the resulting vector and other attributes (orientation, speed) set to 0.
        """
        cross_product = np.cross(self.array[:3], other.array[:3])
        return StateVector(cross_product[0], cross_product[1], cross_product[2],
                           0, 0, 0, 0)

    def heading_difference(self, other: "StateVector") -> float:
        """
        Compute the difference in heading (yaw angle) between this state vector and another,
        normalized to the range [-π, π].

        Parameters:
            other (StateVector): The other state vector.

        Returns:
            float: The normalized heading difference in radians.
        """
        heading_diff_rad = self.yaw_rad - other.yaw_rad

        # Normalize the angle to be within [-pi, pi]
        if heading_diff_rad > np.pi:
            heading_diff_rad -= 2 * np.pi
        elif heading_diff_rad < -np.pi:
            heading_diff_rad += 2 * np.pi

        return heading_diff_rad


@dataclass
class PositionVector:
    """
    Units are in meters.
    """
    x: float
    y: float
    z: float
    array: np.ndarray = field(init=False)

    def __post_init__(self):
        # Initialize the array after the instance is created
        self.array = np.array([self.x, self.y, self.z])

    def __add__(self, other: "PositionVector") -> "PositionVector":
        value = self.array + other.array
        return PositionVector(value[0], value[1], value[2])

    def __sub__(self, other: "PositionVector") -> "PositionVector":
        value = self.array - other.array
        return PositionVector(value[0], value[1], value[2])

    def __mul__(self, other: float) -> "PositionVector":
        value = self.array * other
        return PositionVector(value[0], value[1], value[2])


@dataclass
class RPYVector():
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    array: np.ndarray = field(init=False)

    def __post_init__(self):
        self.array = np.array([self.roll_rad,
                               self.pitch_rad,
                               self.yaw_rad])

    def __add__(self, other: "RPYVector") -> "RPYVector":
        value = self.array + other.array
        return RPYVector(value[0], value[1], value[2])
