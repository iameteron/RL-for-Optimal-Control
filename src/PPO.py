import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

import matplotlib.pyplot as plt
from matplotlib import animation


class PPO(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            max_action: float, 
            observable_states: list,
            gamma: float = 0.999, 
            batch_size: int = 128, 
            epsilon: float = 0.2, 
            epoch_n: int = 30,
            pi_lr: float = 3e-4, 
            v_lr: float = 3e-4,
            hjb_lambda: float = 0,
            v_lambda: float = 1,
            dt: float = 0.1,
        ):

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.observable_states = observable_states

        self.n_neurons = 128

        self.pi_model = nn.Sequential(
            nn.Linear(len(self.observable_states), self.n_neurons), 
            nn.ReLU(),
            nn.Linear(self.n_neurons, self.n_neurons), 
            nn.ReLU(),
            nn.Linear(self.n_neurons, 2 * self.action_dim), 
            nn.Tanh()
        )

        self.v_model = nn.Sequential(
            nn.Linear(len(self.observable_states), self.n_neurons), 
            nn.ReLU(),
            nn.Linear(self.n_neurons, self.n_neurons), 
            nn.ReLU(),
            nn.Linear(self.n_neurons, 1)
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

        self.max_action = max_action

        self.dt = dt
        self.hjb_lambda = hjb_lambda
        self.v_lambda = v_lambda

        self.history = []
        self.hjb_history = []

    def get_action(
            self, 
            state, 
            prediction=False
        ):

        logits = self.pi_model(torch.FloatTensor(state)[self.observable_states])
        mean, log_std = logits[:self.action_dim], logits[self.action_dim:]
        if prediction:
            action = mean.detach()
        else:
            dist = Normal(mean, torch.exp(log_std))
            action = dist.sample()
        return action.numpy().reshape(self.action_dim)

    def fit(
            self, 
            states, 
            actions, 
            rewards, 
            dones, 
            advantage='default'
        ):

        states, actions, rewards, dones = map(np.array, [states, actions, rewards, dones])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, next_states, actions, rewards, returns, dones = map(torch.FloatTensor, [states, next_states, actions, rewards, returns, dones])
        states = states[:, self.observable_states]

        logits = self.pi_model(states)
        mean, log_std = logits[:, :self.action_dim], logits[:, self.action_dim:]

        dist = Normal(mean, torch.exp(log_std))
        old_log_probs = dist.log_prob(actions).detach()

        for _ in range(self.epoch_n):

            idxs = np.random.permutation(returns.shape[0])
            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i: i + self.batch_size]
                b_states = states[b_idxs]
                b_next_states = next_states[b_idxs]
                b_dones = dones[b_idxs]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_returns = returns[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                if advantage == 'default':
                    b_advantage = b_returns.detach() - self.v_model(b_states)

                if advantage == 'bellman':
                    b_advantage = b_rewards.detach() + (1 - b_dones.detach()) * self.gamma * self.v_model(b_next_states.detach()) - self.v_model(b_states) 

                b_logits = self.pi_model(b_states)
                b_mean, b_log_std = b_logits[:, :self.action_dim], b_logits[:, self.action_dim:]

                b_dist = Normal(b_mean, torch.exp(b_log_std))
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1. - self.epsilon,  1. + self.epsilon) * b_advantage.detach()
                pi_loss = - torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = self.v_lambda * torch.mean(b_advantage ** 2)

                # HJB loss
                if self.hjb_lambda != 0:
                    b_hjb_loss = hjb_loss(self, b_states, b_next_states, b_rewards)
                    v_loss += self.hjb_lambda * b_hjb_loss

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()

    def get_trajectory(
        self, 
        env, 
        initial_state=None, 
        prediction=False, 
        visualize=False, 
        filename='gym_animation.gif', 
        sb3=False
    ):

        trajectory = {'states':[], 'actions': [], 'rewards': [], 'dones': []}

        if initial_state is None:
            state = env.reset()[0]
        else:
            env.reset()
            env.observation = initial_state
            env.state = [observation / max_value for observation, max_value in zip(env.observation, env.high)]
            state = env.state

        frames = []
        while True:
            trajectory['states'].append(state)

            if sb3 == True:
                action, _ = self.predict(state, deterministic=True)
            else:
                action = self.get_action(state, prediction=prediction)

            trajectory['actions'].append(action)

            if sb3 == True:
                next_state, reward, _, done, _ = env.step(action)
            else:
                next_state, reward, _, done, _ = env.step(self.max_action * action)

            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)

            state = next_state

            if done:
                break

            if visualize:
                frames.append(env.render(mode="rgb_array"))

        if visualize:
            save_frames_as_gif(frames, filename=filename)

        trajectory['states'].append(state)

        return trajectory


def train_ppo(env, agent, episode_n=50, trajectory_n=20, advantage='default', hjb=False):

    for episode in range(episode_n):

        states, actions, rewards, dones = [], [], [], []

        for i in range(trajectory_n):

            trajectory = agent.get_trajectory(env)
            if hjb == True:
                agent.hjb_history.append(get_hjb_loss(agent, trajectory))

            states.extend(trajectory['states'][:-1])
            actions.extend(trajectory['actions'])
            rewards.extend(trajectory['rewards'])
            dones.extend(trajectory['dones'])

            agent.history.append(np.sum(trajectory['rewards']))

        print(f"{episode}: mean reward = {np.mean(agent.history[-trajectory_n:])}")

        agent.fit(states, actions, rewards, dones, advantage)


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', fps=60):

    plt.figure(figsize=(frames[0].shape[1] / 50.0, frames[0].shape[0] / 50.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=fps)


def validation(env, agent, validation_n, prediction=False, sb3=False):
    total_rewards = []
    for _ in range(validation_n):
        trajectory = agent.get_trajectory(env, prediction=prediction, sb3=sb3)
        total_rewards.append(np.sum(trajectory['rewards']))

    return np.mean(total_rewards)


def plot_history(history, alpha=0.1, ylim=[-10**3, 0]):
    smoothed_history = np.zeros_like(history)
    smoothed_history[0] = history[0]

    for i in range(1, smoothed_history.size):
        smoothed_history[i] = alpha * history[i] + (1 - alpha) * smoothed_history[i - 1]

    plt.ylim(ylim)
    plt.title('PPO')
    plt.xlabel('Trajectory number')
    plt.ylabel('Smoothed trajectory reward')
    plt.legend()
    plt.plot(smoothed_history)


def hjb_loss(agent, states, next_states, rewards):

    states.requires_grad = True
    next_states.requires_grad = True

    values_grad = agent.v_model(states)
    value_derivative = torch.autograd.grad(
        values_grad,
        states,
        grad_outputs=torch.ones_like(values_grad),
        create_graph=True,
        retain_graph=True,
    )[0]

    states.requires_grad = False

    dynamics = (next_states - states) / agent.dt

    value_derivative_dot_f = torch.bmm(
        value_derivative.view(value_derivative.shape[0], 1, value_derivative.shape[1]), 
        dynamics.view(value_derivative.shape[0], value_derivative.shape[1], 1)
        ).flatten()

    hjb_loss = torch.nn.functional.mse_loss(values_grad * np.log(agent.gamma), value_derivative_dot_f[:, None] + rewards)
    return hjb_loss

def get_hjb_loss(agent, trajectory):
    states, actions, rewards = map(np.array, [trajectory['states'][:-1], trajectory['actions'], trajectory['rewards']])
    rewards = rewards.reshape(-1, 1)

    next_states = np.zeros_like(states)
    next_states[:-1] = states[1:]

    states, next_states, actions, rewards = map(torch.FloatTensor, [states, next_states, actions, rewards])
    states = states[:, agent.observable_states]

    return hjb_loss(agent, states, next_states, rewards).detach().numpy()

