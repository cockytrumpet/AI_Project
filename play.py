import os
from stable_baselines3 import A2C  # DQN, A2C, PPO, ...
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


if os.path.exists("model.zip"):
    print("loading model...")
    model = A2C.load("model", verbose=1, device="mps")
else:
    print("model not found")
    exit()

env = make_atari_env(
    "ALE/Frogger-v5", n_envs=4, env_kwargs=dict(render_mode="rgb_array")
)

env.metadata["render_fps"] = 10
env.metadata["render.modes"] = ["human", "rgb_array"]

env = VecFrameStack(env, n_stack=4)

model.set_env(env)

obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    env.render("human")
