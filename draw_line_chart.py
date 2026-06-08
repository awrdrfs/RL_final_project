import matplotlib.pyplot as plt
import csv
import numpy as np

dqn_list = []
dqn_woReplayBuffer_list = []
Qtransformer_list = []

# 以下是繪製驗證的圖表
''''''
with open("train_result/log/DQN_valid.csv",newline='') as dqnfile:
    rows = csv.reader(dqnfile)

    # 跳過header
    next(rows)

    for row in rows:
        dqn_list.append(float(row[2])) #抓return_pct 的資料(第三個數值)

with open("train_result/log/DQN_valid_woReplayBuffer.csv",newline='') as Qtransfile:
    rows = csv.reader(Qtransfile)

    # 跳過header
    next(rows)

    for row in rows:
        dqn_woReplayBuffer_list.append(float(row[2])) #抓return_pct 的資料(第三個數值)

# x 軸長度自動配合資料
count = np.arange(len(dqn_list))

plt.plot(count, dqn_list, label='with replay buffer')
plt.plot(count, dqn_woReplayBuffer_list, label='without replay buffer')

plt.legend()
plt.xlabel("Episode")
plt.ylabel("Profit&Loss Ratio")
plt.title("DQN with and without replay buffer Valid P&L Ratio")

plt.savefig("train_result/image/valid_P&L_Ratio_compare_w&woReplayBuffer.png")
plt.show()

# 以下是繪製訓練的圖表
'''
with open("train_result/log/DQN_train.csv",newline='') as dqnfile:
    rows = csv.reader(dqnfile)

    # 跳過header
    next(rows)

    for row in rows:
        dqn_list.append(float(row[1])) #抓total reward 的資料(第二個數值)

with open("train_result/log/DQN_train_woReplayBuffer.csv",newline='') as Qtransfile:
    rows = csv.reader(Qtransfile)

    # 跳過header
    next(rows)

    for row in rows:
        dqn_woReplayBuffer_list.append(float(row[1])) #抓total reward 的資料(第二個數值)

# x 軸長度自動配合資料
count = np.arange(len(dqn_list))

plt.plot(count, dqn_list, label='with replay buffer')
plt.plot(count, dqn_woReplayBuffer_list, label='without replay buffer')

plt.legend()
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("DQN with and without replay buffer Training Reward")

plt.savefig("train_result/image/train_total_reward_compare_w&woReplayBuffer.png")
plt.show()
'''