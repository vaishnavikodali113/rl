import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from env_setup import make_env

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

env = make_env("walker", "walk")
eval_env = make_env("walker", "walk")

log_dir = "./logs/ppo_walker_mac"
os.makedirs(log_dir, exist_ok=True)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    device=device,
    tensorboard_log=log_dir,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir + "/best/",
    log_path=log_dir + "/eval/",
    eval_freq=10000,
)

model.learn(total_timesteps=300_000, callback=eval_callback)
model.save(log_dir + "/final_model")