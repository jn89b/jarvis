import unittest
import numpy as np
import matplotlib.pyplot as plt
from jarvis.envs.simple_agent import SimpleAgent, PlaneKinematicModel, DataHandler
from jarvis.envs.battlespace import BattleSpace
from jarvis.utils.vector import StateVector
from jarvis.algos.pro_nav import ProNavV2


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
            agent_id=0,
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
            agent_id=0,
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
        """
        Keep in mind frames are in NED
        """
        self.init_agent()
        n_steps: int = 100

        # action commands are
        # roll, pitch, yaw, vel_cmd
        roll_cmd: float = np.deg2rad(0)
        pitch_cmd: float = np.deg2rad(20)
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
                  self.agent.state_vector.y,
                  self.agent.state_vector.z)

        data: DataHandler = self.agent.return_data()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.plot(data.x, data.y, data.z)
        ax.scatter(data.x[0], data.y[0], data.z[0], c='r', label='start')
        ax.legend()

        fig, ax = plt.subplots(nrows=4, ncols=1)
        ax[0].plot(np.rad2deg(data.phi), label='phi')
        ax[0].  set_title('Phi')
        ax[1].plot(np.rad2deg(data.theta), label='theta')
        ax[1].plot(np.rad2deg(data.u_theta), c='r', label='theta_cmd')
        ax[1].set_title('Theta')
        ax[2].plot(np.rad2deg(data.psi))
        ax[2].set_title('Psi')
        ax[3].plot(data.v, label='velocity')
        ax[3].plot(data.v_cmd, c='r', label='velocity_cmd')
        ax[3].set_title('Velocity')

        for a in ax:
            a.legend()

        plt.show()

    def test_pronav(self) -> None:
        """
        A simple test to see if the pro nav policy
        is working
        """
        pro_nav = ProNavV2()

        pursuer_agent = SimpleAgent(
            battle_space=self.battlespace,
            state_vector=StateVector(
                x=150,
                y=100,
                z=50,
                roll_rad=0.1,
                pitch_rad=0.1,
                yaw_rad=0.0,
                speed=25
            ),
            agent_id=0,
            simple_model=PlaneKinematicModel()
        )

        agent_2 = SimpleAgent(
            battle_space=self.battlespace,
            state_vector=StateVector(
                x=-30,
                y=0,
                z=30,
                roll_rad=0.1,
                pitch_rad=0.1,
                yaw_rad=0,
                speed=15.0
            ),
            agent_id=1,
            simple_model=PlaneKinematicModel()
        )

        n_steps: int = 1000
        agent_2_commands = np.array([0.1, 0, 25])

        agents = [pursuer_agent, agent_2]
        distance_history: list[float] = []
        for i in range(n_steps):
            # NOTE WE need to flip this around to make it work
            relative_pos = agent_2.state_vector - pursuer_agent.state_vector
            relative_heading = pursuer_agent.state_vector.yaw_rad - \
                agent_2.state_vector.yaw_rad

            relative_pos = relative_pos.array[0:3]
            relative_vel = pursuer_agent.state_vector.speed - \
                agent_2.state_vector.speed

            # pro_nav_actions = pro_nav.compute_commands(relative_pos=relative_pos,
            #                                            current_yaw=pursuer_agent.state_vector.yaw_rad,
            #                                            current_speed=pursuer_agent.state_vector.speed)
            # print("pro nav actions", np.rad2deg(pro_nav_actions[1]))
            #
            # agent_2_commands[1] = np.deg2rad(i)
            agent_2.act(action=agent_2_commands)
            # augmented_pro_nav_actions = pro_nav.augmented_pro_nav(
            #     relative_pos=relative_pos,
            #     current_yaw=pursuer_agent.state_vector.yaw_rad,
            #     current_speed=pursuer_agent.state_vector.speed,
            # )
            # augmented_pro_nav_actions = pro_nav.calculate(
            #     relative_pos=relative_pos,
            #     current_yaw=pursuer_agent.state_vector.yaw_rad,
            #     current_speed=pursuer_agent.state_vector.speed,
            #     evader_yaw=agent_2.state_vector.yaw_rad,
            #     evader_speed=agent_2.state_vector.speed
            # )
            augmented_pro_nav_actions = pro_nav.predict(
                current_pos=pursuer_agent.state_vector.array[0:3],
                relative_pos=relative_pos,
                current_heading=pursuer_agent.state_vector.yaw_rad,
                current_speed=pursuer_agent.state_vector.speed,
                relative_vel=relative_vel)

            pursuer_agent.act(augmented_pro_nav_actions)
            print("augmented pro nav actions", augmented_pro_nav_actions)
            print("augmented pro nav actions", np.rad2deg(
                augmented_pro_nav_actions[1]))

            for agent in agents:
                agent.step()

            # compute the distance between the two agents
            distance: float = pursuer_agent.state_vector.distance_3D(
                agent_2.state_vector)
            distance_history.append(distance)
            if distance < 30.0:
                print("caught")
                break

        datas: list[DataHandler] = [agent.return_data() for agent in agents]

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        for data in datas:
            ax.plot(data.x, data.y, data.z)
            ax.scatter(data.x[0], data.y[0], data.z[0], c='r', label='start')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        ax.legend()
        fig, ax = plt.subplots()
        ax.plot(distance_history)

        for agent in agents:
            print("agent position",
                  agent.state_vector.x,
                  agent.state_vector.y,
                  agent.state_vector.z)

        fig, ax = plt.subplots(nrows=4, ncols=1)
        ax[0].plot(np.rad2deg(datas[0].phi), label='phi')
        ax[0].set_title('Phi')

        plt.show()


if __name__ == '__main__':
    unittest.main()
