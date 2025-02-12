import unittest
import numpy as np
import matplotlib.pyplot as plt
from jarvis.envs.simple_agent import SimpleAgent, PlaneKinematicModel, DataHandler
from jarvis.envs.battlespace import BattleSpace
from jarvis.utils.vector import StateVector


class TestGenerateRLData(unittest.TestCase):

    def setUp(self):
        self.battlespace = BattleSpace(x_bounds=[0, 100],
                                       y_bounds=[0, 100],
                                       z_bounds=[0, 100])
        self.init_agent()

    def init_agent(self) -> None:
        """
        Simple test to spawn an agent and see 
        that it spawns to correct location
        """
        x = 50
        y = 50
        z = 50
        roll_rad = 0.1
        pitch_rad = 0.1
        yaw_rad = 0.1
        speed = 25
        state_vector = StateVector(
            x=x,
            y=y,
            z=z,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
            speed=speed
        )
        self.plane_model = PlaneKinematicModel()
        self.agent = SimpleAgent(
            battle_space=self.battlespace,
            state_vector=state_vector,
            id=0,
            simple_model=self.plane_model)

    def test_spawn_agent(self) -> None:
        """
        Simple test to spawn an agent and see 
        that it spawns to correct location
        """
        x = 50
        y = 50
        z = 50
        roll_rad = 0.1
        pitch_rad = 0.1
        yaw_rad = 0.1
        speed = 25
        state_vector = StateVector(
            x=x,
            y=y,
            z=z,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
            speed=speed
        )
        plane_model = PlaneKinematicModel()
        agent = SimpleAgent(
            battle_space=self.battlespace,
            state_vector=state_vector,
            id=0,
            simple_model=plane_model)

        assert agent is not None
        print("agent state vector", agent.state_vector)
        assert agent.state_vector.x == x
        assert agent.state_vector.y == y
        assert agent.state_vector.z == z
        assert agent.state_vector.roll_rad == roll_rad
        assert agent.state_vector.pitch_rad == pitch_rad
        assert agent.state_vector.yaw_rad == yaw_rad

    def test_move_agent(self) -> None:
        self.init_agent()
        n_steps: int = 100

        # action commands are
        # roll, pitch, yaw, vel_cmd
        roll_cmd: float = np.deg2rad(45)
        pitch_cmd: float = np.deg2rad(0)
        yaw_cmd: float = np.deg2rad(0)
        vel_cmd: float = 15
        action = np.array([roll_cmd,
                           pitch_cmd,
                           yaw_cmd,
                           vel_cmd])
        self.agent.act(action=action)

        for i in range(n_steps):
            assert self.agent.actions.all() == action.all()
            self.agent.step()
            print("agent position",
                  self.agent.state_vector.x,
                  self.agent.state_vector.y)

        data: DataHandler = self.agent.return_data()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.plot(data.x, data.y, data.z)
        plt.show()


if __name__ == '__main__':
    unittest.main()
