from env_setup import make_env
env = make_env("hopper", "hop")

obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
env.close()