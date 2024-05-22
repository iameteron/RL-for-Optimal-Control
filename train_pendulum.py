import os

from sgs_environment import *
from multienv import *
from validation_functions import *

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

import torch

if __name__ == '__main__':

    # device для обучения

    device = 'cuda:7'
    torch.cuda.set_device(device)

    # пути для логов

    new_folder = 'pendulum_ppo_test'
    exist_ok = True # if new_folder == 'ssj_ppo_test' else False

    train_log_dir = "./gym_logs/train_logs/"
    os.makedirs(train_log_dir, exist_ok=exist_ok)

    eval_log_dir = "./gym_logs/eval_logs/" + new_folder
    os.makedirs(eval_log_dir, exist_ok=exist_ok)
    
    # multienv для обучения

    def make_env(env_id: str, rank: int, seed: int = 0):
        def _init():
            env = gym.make(env_id, render_mode="human")
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)        
        return _init

    env_id = 'Pendulum-v1'
    num_train_envs = 8
    train_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_train_envs)], start_method='fork')
    # train_env = make_vec_env(env_id, num_train_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10)

    # env для валидации

    num_eval_envs = 1
    eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_eval_envs)], start_method='fork')
    # eval_env = make_vec_env(env_id, num_eval_envs)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10)

    # callback для периодического сохранения модели

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000 // num_train_envs,
        save_path=eval_log_dir,
        name_prefix=new_folder,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # callback для сохранения лучшей модели

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir, 
        eval_freq=20_000 // num_train_envs,
        n_eval_episodes=5,
        deterministic=True, 
        render=False,
    )
    
    # создание агента

    agent = PPO(
        policy='MlpPolicy',
        env=train_env,
        n_steps=2048 // num_train_envs,
        device=device,
        tensorboard_log=train_log_dir,
        verbose=1,
    )

    # обучение

    tb_log_name = new_folder + '_logs'

    agent.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=tb_log_name,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # сохранение модели после обучения

    model_path = './gym_logs/eval_logs/' + new_folder + '/final_model'
    agent.save(model_path)