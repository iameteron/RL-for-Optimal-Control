from stable_baselines3 import PPO
from Orientation3D import *

if __name__ == '__main__':

    env = Orientation3D(dt=1e-2)

    tensorboard_log = './tensorboard_logs/'

    agent = PPO(
        policy='MlpPolicy',
        env=env,
        gamma=1,
        tensorboard_log=tensorboard_log,
    )

    agent.learn(
        total_timesteps=int(10e4),
    )

    model_path = './models/orientation3D_sb3_ppo'
    agent.save(model_path)

