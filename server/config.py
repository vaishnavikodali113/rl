import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Registry of all trained algorithms.
# Each entry: display label -> (artifact subfolder, algo type, env_name, task)
ALGORITHM_REGISTRY = {
    "PPO (Walker)": {
        "artifact_dir": "ppo_walker",
        "algo_type": "ppo",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.zip",      # SB3 saves as .zip
    },
    "SAC (Cheetah)": {
        "artifact_dir": "sac_cheetah",
        "algo_type": "sac",
        "env_name": "cheetah",
        "task": "run",
        "checkpoint": "model.zip",
    },
    "TD-MPC2 MLP": {
        "artifact_dir": "tdmpc2_walker_mlp",
        "algo_type": "tdmpc",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.pt",
    },
    "TD-MPC2 S4": {
        "artifact_dir": "tdmpc2_walker_s4",
        "algo_type": "tdmpc",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.pt",
    },
    "TD-MPC2 S5": {
        "artifact_dir": "tdmpc2_walker_s5",
        "algo_type": "tdmpc",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.pt",
    },
    "TD-MPC2 Mamba": {
        "artifact_dir": "tdmpc2_walker_mamba",
        "algo_type": "tdmpc",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.pt",
    },
}

STREAM_FPS = 20          # WebSocket target frame rate
MPPI_HORIZON = 5         # Planning horizon used during live rollout
MPPI_SAMPLES = 256       # Reduced vs. training for real-time speed
RENDER_WIDTH = 320
RENDER_HEIGHT = 240
