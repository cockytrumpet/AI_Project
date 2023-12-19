import os
from stable_baselines3 import A2C, DQN  # DQN, A2C, PPO, ...
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


if os.path.exists("model.zip"):
    print("loading model...")
    model = A2C.load("model", verbose=1, device="mps")
else:
    print("model not found")
    exit()

# env = make_atari_env("ALE/Berzerk-v5", n_envs=3, env_kwargs=dict(mode=1), seed=0)
env = make_atari_env(
    "ALE/Frogger-v5", n_envs=4, env_kwargs=dict(render_mode="rgb_array")
)

env = VecFrameStack(env, n_stack=4)

model.set_env(env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
print(f"mean_reward: {mean_reward:.0f} +/- {std_reward:.2f}")
