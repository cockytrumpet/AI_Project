import matplotlib.pyplot as plt

# x-axis: observation number
#         { 0: '0', 1: '10k', 2: '100k', 3: '200k', 4: '500k', 5: '1m' }
# y-axis: mean reward


# it was faster to just type these out than to write a script to generate them
a2c_cnn = [9.6, 1.1, 25.8, 30.6, 46.0, 65.1]
a2c_mlp = [4.7, 1.1, 1.2, 1.1, 1.2, 1.2]
dqn_cnn = [0.0, 0.1, 6.0, 8.8, 15.8, 35.0]
dqn_mlp = [0.0, 0.1, 5.0, 13.8, 12.2, 16.9]
ppo_cnn = [0.0, 10.2, 39.3, 41.6, 110.1, 122.0]
ppo_mlp = [0.0, 1.2, 1.2, 1.2, 1.3, 1.2]

x_labels = ["0", "10k", "100k", "200k", "500k", "1m"]

x_axis = range(len(x_labels))

plt.xticks(x_axis, x_labels)

plt.plot(x_axis, a2c_cnn, label="A2C-CNN")
plt.plot(x_axis, a2c_mlp, label="A2C-MLP")
plt.plot(x_axis, dqn_cnn, label="DQN-CNN")
plt.plot(x_axis, dqn_mlp, label="DQN-MLP")
plt.plot(x_axis, ppo_cnn, label="PPO-CNN")
plt.plot(x_axis, ppo_mlp, label="PPO-MLP")

plt.xlabel("Time Steps")
plt.ylabel("Mean Reward")
plt.title("Model-Policy Performance at Various Time Steps")
plt.legend()
plt.show()
