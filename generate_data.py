#!python
# Generate models and results
import os
import sys

algos = ["A2C", "DQN", "PPO"]  # algos that support discreet action spaces
policies = ["CnnPolicy", "MlpPolicy"]
time_steps = [10_000, 90_000, 100_000, 300_000, 500_000]


for algo in algos:
    for policy in policies:
        for time_step in time_steps:
            if time_step < 1_000_000:
                directory = str(time_step)[:-3] + "k"
            else:
                directory = str(time_step)[:-6] + "m"

            if os.path.exists(directory + "/" + algo + "-" + policy + ".zip"):
                print(f"skipping {directory}/{algo}-{policy}...")
                continue

            print(f"algo: {algo}, policy: {policy}, time_steps: {time_step}")
            exec_str = "python train.py " + algo + " " + policy + " " + str(time_step)
            os.system(exec_str)

        if os.path.exists("model.zip"):
            os.remove("model.zip")

print("done!\a")
