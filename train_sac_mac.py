import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from env_setup import make_env

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

env = make_env("cheetah", "run")
eval_env = make_env("cheetah", "run")

log_dir = "./logs/sac_cheetah_mac"
os.makedirs(log_dir, exist_ok=True)

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=500_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    device=device,
    tensorboard_log=log_dir,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir + "/best/",
    log_path=log_dir + "/eval/",
    eval_freq=20000,
)

model.learn(total_timesteps=500_000, callback=eval_callback)
model.save(log_dir + "/final_model")