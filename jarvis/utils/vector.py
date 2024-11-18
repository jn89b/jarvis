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


# @dataclass
class StateVector:
    """
    Units are in meters and radians.
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
        self.vx = self.speed * np.cos(self.yaw_rad)*np.cos(self.pitch_rad)
        self.vy = self.speed * np.sin(self.yaw_rad)*np.cos(self.pitch_rad)
        self.vz = self.speed * np.sin(self.pitch_rad)

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z,
                         self.roll_rad, self.pitch_rad,
                         self.yaw_rad,
                         self.speed,
                         self.vx, self.vy, self.vz])

    def update(self, x=None, y=None, z=None,
               roll_rad=None, pitch_rad=None,
               yaw_rad=None, speed=None) -> None:
        """
        Update the state vector values.
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

        self.vx = self.speed * np.cos(self.yaw_rad)*np.cos(self.pitch_rad)
        self.vy = self.speed * np.sin(self.yaw_rad)*np.cos(self.pitch_rad)
        self.vz = self.speed * np.sin(self.pitch_rad)

    def __add__(self, other: "StateVector") -> "StateVector":
        value = self.array + other.array
        # return StateVector(
        #     value[0], value[1], value[2], value[3], value[4], value[5],
        #     value[6])
        return StateVector(*value[X_IDX:SPEED_IDX+1])

    def __sub__(self, other: "StateVector") -> "StateVector":
        value = self.array - other.array
        # return StateVector(value[0], value[1], value[2],
        #                    value[3], value[4], value[5],
        #                    value[6])
        return StateVector(*value[X_IDX:SPEED_IDX+1])

    def __mul__(self, other: float) -> "StateVector":
        value = self.array * other
        return StateVector(*value[X_IDX:SPEED_IDX])
        # return StateVector(value[0], value[1], value[2],
        #                    value[3], value[4], value[5],
        #                    value[6])

    def __repr__(self) -> str:
        return f"StateVector({self.x}, {self.y}, {self.z}, \
            {self.roll_rad}, {self.pitch_rad}, {self.yaw_rad}, {self.speed})"

    def distance_2D(self, other: "StateVector") -> float:
        """
        Parameters
        ----------
        other : StateVector
            The other state vector.

        Returns
        -------
        float
            The distance between the two state vectors.
        """
        # get the position vector
        position_vector = self - other
        # get the distance
        distance = np.linalg.norm(position_vector.array[:2])
        return distance

    def distance_3D(self, other: "StateVector") -> float:
        """
        Parameters
        ----------
        other : StateVector
            The other state vector.

        Returns
        -------
        float
            The distance between the two state vectors.
        """
        # get the position vector
        position_vector = self - other
        # get the distance
        distance = np.linalg.norm(position_vector.array)
        return distance

    def unit_vector_2D(self) -> "StateVector":
        """
        Returns
        -------
        StateVector
            The unit vector of the state vector.
        """
        # get the direction vector
        yaw = self.array[5]
        # get the unit vector
        unit_vector = np.array([np.cos(yaw), np.sin(yaw)])
        return StateVector(unit_vector[0], unit_vector[1], 0, 0, 0, 0, 0)

    def dot_product_2D(self, other: "StateVector") -> float:
        """
        Parameters
        ----------
        other : StateVector
            The other state vector.

        Returns
        -------
        float
            The dot product of the two state vectors.
        """
        # get the unit vector
        unit_vector = self.unit_vector_2D()
        # get the other unit vector
        other_unit_vector = other.unit_vector_2D()
        # get the dot product
        dot_product = np.dot(unit_vector.array, other_unit_vector.array)
        return dot_product

    def dot_product_3D(self, other: "StateVector") -> float:
        """
        Parameters
        ----------
        other : StateVector
            The other state vector.

        Returns
        -------
        float
            The dot product of the two state vectors.
        """
        # get the unit vector
        unit_vector = self.array
        # get the other unit vector
        other_unit_vector = other.array
        # get the dot product
        dot_product = np.dot(unit_vector, other_unit_vector)
        return dot_product

    def cross_product_3D(self, other: "StateVector") -> "StateVector":
        """
        Parameters
        ----------
        other : StateVector
            The other state vector.

        Returns
        -------
        StateVector
            The cross product of the two state vectors.
        """
        # get the unit vector
        unit_vector = self.array
        # get the other unit vector
        other_unit_vector = other.array
        # get the cross product
        cross_product = np.cross(unit_vector, other_unit_vector)
        return StateVector(
            cross_product[0], cross_product[1], cross_product[2],
            0, 0, 0, 0)

    def heading_difference(self, other: "StateVector") -> float:
        """
        Parameters
        ----------
        other : StateVector
            The other state vector.

        Returns
        -------
        float
            The heading difference between the two state vectors radians.
        """
        # get the dot product
        heading_ego = self.array[5]
        heading_other = other.array[5]

        heading_diff_rad = heading_ego - heading_other

        # wrap the heading difference between 0 and 2pi
        if heading_diff_rad > np.pi:
            heading_diff_rad -= 2*np.pi
        elif heading_diff_rad < -np.pi:
            heading_diff_rad += 2*np.pi

        return heading_diff_rad


@ dataclass
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


@ dataclass
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
