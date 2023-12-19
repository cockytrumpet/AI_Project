import os
import sys
from shutil import copyfile
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "ALE/Frogger-v5"
algo = "A2C"
policy = "CnnPolicy"
time_steps = 10_000

if len(sys.argv) > 3:
    algo = sys.argv[1]
    policy = sys.argv[2]
    time_steps = int(sys.argv[3])

if time_steps < 1_000_000:
    directory = str(time_steps)[:-3] + "k"
else:
    directory = str(time_steps)[:-6] + "m"

if not os.path.exists(directory):
    os.makedirs(directory)

output_results = f"{directory}/{algo}-{policy}.txt"
output_model = f"{directory}/{algo}-{policy}.zip"

env = make_atari_env(
    env_name,
    n_envs=4,
    env_kwargs=dict(render_mode="rgb_array"),
)
env = VecFrameStack(env, n_stack=4)

if os.path.exists("model.zip"):
    print("loading model...")
    exec_str = algo + '.load("model", verbose=1, device="mps")'
    model = eval(exec_str)
else:
    print("creating model...")
    exec_str = algo + '("' + policy + '", env, verbose=1, device="mps")'
    model = eval(exec_str)

old_mean_reward, old_std_reward = evaluate_policy(
    model, env, n_eval_episodes=100, warn=False
)

model.set_env(env)
model.learn(total_timesteps=time_steps, progress_bar=True, log_interval=1000)
model.save("model")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)

old_data = f"old_mean_reward: {old_mean_reward:.2f} +/- {old_std_reward:.2f}\n"
new_data = f"new_mean_reward: {mean_reward:.1f} +/- {std_reward:.2f}\n"
with open(output_results, "w") as f:
    f.write(old_data)
    f.write(new_data)
copyfile("model.zip", output_model)

print(old_data)
print(new_data)
