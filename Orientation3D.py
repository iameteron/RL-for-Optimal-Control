import gym
import random
import numpy as np

import matplotlib.pyplot as plt

class Orientation3D(gym.Env):
    def __init__(
            self, 
            t_start: int = 0, 
            t_end: int = 1,
            dt: float = 0.1, 
            R1: int = 10,
            R2: int = 10,
            R3: int = 10,
            l: float = 5 / 6,
            mu: float = 5 / 6,
            a1: float = 5 / 6,
            a2: float = 1 / 6,
            a3: float = 1 / 6,
            max_action: int = 200,
            observable_states: list = [0, 1, 2, 3],
            ic_boundaries: list = [[23.5, 24.5], [15.5, 16.5], [15.5, 16.5]]
    ):

        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt

        self.R1 = R1
        self.R2 = R2
        self.R3 = R3

        self.l = l
        self.mu = mu
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

        self.max_action = max_action

        self.ic_boundaries = np.array(ic_boundaries)

        self.observable_states = observable_states
        low = np.array([0, -50, -50, -50])
        high = np.array([1, 50, 50, 50])

        self.observation_space = gym.spaces.Box(
            low=low[observable_states], 
            high=high[observable_states], 
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None, random_time=False):
        if random_time:
            self.state = np.array(
                [random.randint(0, int(1 / self.dt) - 1) * self.dt] + [np.random.uniform(x[0], x[1]) for x in self.ic_boundaries]
            )
        else:
            self.state = np.array(
                [0] + [np.random.uniform(x[0], x[1]) for x in self.ic_boundaries]
            )

        return self.state, {}
    
    def step(self, u):
        t_next = self.state[0] + self.dt
        x_curr = self.state[1:]
        df1_dt = self.a1 * u[0] - (self.l - self.mu) * x_curr[1] * x_curr[2]
        df2_dt = (self.a2 * u[1] - (1 - self.l) * x_curr[2] * x_curr[0]) / self.mu
        df3_dt = (self.a3 * u[2] - (self.mu - 1) * x_curr[0] * x_curr[1]) / self.l
        df_dt = np.array([df1_dt, df2_dt, df3_dt])
        x_next = self.integration(x_curr, df_dt)
        self.state = np.hstack((t_next, x_next)) 

        if np.isclose(t_next, self.t_end):
            reward = - self.R1 * self.state[1]**2 - self.R2 * self.state[2]**2 - self.R3 * self.state[3]**2
            reward -= self.dt * (np.sum(np.abs(u)))
            done = True

        else:
            reward = - self.dt * (np.sum(np.abs(u)))
            done = False

        return self.state, reward, False, done, {}

    def integration(self, x_curr, df_dt, method='Euler'):
        if method == 'Euler':
            x_next = df_dt * self.dt + x_curr
            return x_next


def plot_u(env, agent, n=1, initial_state=None):
    for _ in range(n):
        traj = agent.get_trajectory(env, prediction=True, initial_state=initial_state)
        actions = np.array(traj['actions'])
        u1 = agent.max_action * actions[:, 0]
        u2 = agent.max_action * actions[:, 1]
        u3 = agent.max_action * actions[:, 2]
        t = np.arange(env.t_start, env.t_end + env.dt, env.dt)

        kwargs_x1 = {'color': 'blue'}
        kwargs_x2 = {'color': 'orange'}
        kwargs_x3 = {'color': 'green'}
        plt.plot(t[:-1], u1, **kwargs_x1)
        plt.plot(t[:-1], u2, **kwargs_x2)
        plt.plot(t[:-1], u3, **kwargs_x3)

    plt.plot([], [], label=r'$ p $', **kwargs_x1)
    plt.plot([], [], label=r'$ q $', **kwargs_x2)
    plt.plot([], [], label=r'$ r $', **kwargs_x2)
    plt.title('Bundle of Control Signals')
    plt.xlabel('t')
    plt.legend()


def plot_x(env, agent, n=1, initial_state=None):
    for _ in range(n):
        traj = agent.get_trajectory(env, prediction=True, initial_state=initial_state)
        t = np.arange(env.t_start, env.t_end + env.dt, env.dt)
        states = np.array(traj['states'])
        p = states[:, 1]
        q = states[:, 2]
        r = states[:, 3]

        print(p[-1], q[-1], r[-1])
        
        kwargs_x1 = {'color': 'blue'}
        kwargs_x2 = {'color': 'orange'}
        kwargs_x3 = {'color': 'green'}
        plt.plot(t, p, **kwargs_x1)
        plt.plot(t, q, **kwargs_x2)
        plt.plot(t, r, **kwargs_x3)

    plt.plot([], [], label=r'$ p $', **kwargs_x1)
    plt.plot([], [], label=r'$ q $', **kwargs_x2)
    plt.plot([], [], label=r'$ r $', **kwargs_x2)
    plt.title('Bundle of Trajectories')
    plt.xlabel('t')
    plt.legend()