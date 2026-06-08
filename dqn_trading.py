import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 超參數
episodes = 100
share_num = 1
gamma = 0.99    # 折扣因子
learning_rate = 0.001   # 學習率
batch_size = 64        # Replay Buffer 每次取樣數
buffer_capacity  = 10_000    # Replay Buffer 容量，註：底線不影響數值、旨在加強可讀性
target_update_freq = 100     # 每幾步同步 Target Network
epsilon_start = 1.0     # 初始探索率
epsilon_end = 0.05    # 最低探索率
epsilon_decay = 0.95   # 每 episode 後的衰減倍率

# 維度計算
state_dim = 5 * share_num + (share_num + 1)  # OHLCV + 持股數 + 現金
action_dim = share_num + 1                    # 股票權重 + 現金權重

# 載入資料集
# 加入盤前、盤後資訊會導致step 更新困難。目前暫改只有營業資料
# 註：test 資料一開始all in hold 到最後的損益比是27.07%
train_df = pd.read_csv("dataset/train/XOM_wo_prepost.csv")
valid_df = pd.read_csv("dataset/valid/2m/XOM_wo_prepost.csv")
test_df = pd.read_csv("dataset/test/XOM_wo_prepost.csv")

# 儲存訓練資料
train_log = []
valid_log = []
test_log = []

# 儲存模型路徑
FILE_best = 'train_result/model/DQN_best_model_state_dict.pt'
FILE_end = 'train_result/model/DQN_100episode_model_state_dict.pt'

#  交易環境
class TradingEnv:
    def __init__(self, df, initial_cash=1000, transaction_rate=0.003):
        self.df = df
        self.initial_cash = initial_cash
        self.transaction_rate = transaction_rate
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = np.array([0.0]) # 假設只處理一支股票: XOM
        self.portfolio_value = self.initial_cash
        self.old_value = self.initial_cash

        return self._get_observation()

    def _get_observation(self):
        # 狀態包含: OHLCV (當前列) + 目前持有股數 + 目前現金
        obs = self.df.iloc[self.current_step, 1:].values.tolist()
        obs.extend(self.shares.tolist())
        obs.append(self.cash)

        return np.array(obs, dtype=np.float32)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def step(self, action):
        # 1. Softmax 轉換動作為比例 [share_rate, cash_rate]
        new_action  = self._softmax(action)
        share_rates = new_action[:1]

        # 2. 取得當前價格 (Close) 與 下一期開盤價 (用於交易)
        # 這裡假設 current_price 用於計算當前價值，next_open 用於執行交易
        current_prices = np.array([self.df.iloc[self.current_step]['Close']])

        # 檢查是否還有下一行資料
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True

        next_open_prices = np.array([self.df.iloc[self.current_step + 1]['Open']])

        # 3. 計算總資產 (Portfolio Value)
        self.portfolio_value = np.sum(self.shares * current_prices) + self.cash
        
        # 4. 計算舊權重與差值
        old_share_rates = (self.shares * current_prices) / self.portfolio_value
        delta_shares = share_rates - old_share_rates 

        # 5. 執行交易 (依照你提供的邏輯)
        for i in range(len(self.shares)):
            trade_value = delta_shares[i] * self.portfolio_value
            price = next_open_prices[i]

            # 破產懲罰預設 = 0
            # 避免RL打破規則(買超過現金的股票、賣超過持有量的股票)
            break_rule_punish = 0.0

            if trade_value > 0: # 買入
                fee = trade_value * self.transaction_rate
                # 判斷現金是否足夠
                if self.cash >= trade_value + fee: 
                    self.cash -= (trade_value + fee)
                    self.shares[i] += trade_value / price   
                else:
                    # 改成用現有現金能買多少就買多少
                    affordable = self.cash / (1 + self.transaction_rate)
                    fee = affordable * self.transaction_rate
                    self.cash -= (affordable + fee)
                    self.shares[i] += affordable / price
                    # 破產了，懲罰 -1
                    break_rule_punish = -1.0    
            else:  # 賣出
                pos = abs(trade_value)
                max_sellable = self.shares[i] * price
                if pos > max_sellable: # 若賣出數量超過手上的持有量則修正為手上的持有量
                    pos = max_sellable
                    # 超賣了，懲罰 -1
                    break_rule_punish = -1.0

                fee = pos * self.transaction_rate
                self.cash += (pos - fee)
                self.shares[i] -= pos / price

        # 6. 計算新價值與 Reward
        # 交易後的價值更新（以交易當下的價格估算）
        new_value = np.sum(self.shares * next_open_prices) + self.cash
        reward = (new_value - self.old_value) / self.old_value + break_rule_punish

        self.old_value = new_value
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done

#  Replay Buffer
class ReplayBuffer:
    '''儲存 (state, action_idx, reward, next_state, done) 經驗'''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)

# Q Network
class QNetwork(nn.Module):
    def __init__(self, input_dim=state_dim, output_dim=action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 完整 DQN 訓練
def train_dqn(env=None):
    if env is None:
        env = TradingEnv(train_df)

    # model、Target Network、優化器
    model = QNetwork(state_dim, action_dim)
    target_model = QNetwork(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())  # 初始權重同步
    target_model.eval()                               # Target Network 只做推論

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    step_count = 0  # 全域步數，用於 Target Network 同步
    best_ret = float('-inf')

    for ep in range(episodes):
        model.train()
        state = env.reset()
        done = False
        total_reward = 0.0
        step_in_ep = 0

        while not done:
            # Epsilon-Greedy 選擇 Action
            if random.random() < epsilon:
                # 探索：隨機產生
                action = np.random.randn(action_dim).astype(np.float32)
                action_idx = int(np.argmax(action))          # 記錄 argmax 作為代表索引
            else:
                # 利用：取 Q 值最大的動作
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_tensor)
                action = q_values.numpy()[0]
                action_idx = int(torch.argmax(q_values).item())

            next_state, reward, done = env.step(action)

            # 存入 Replay Buffer
            buffer.push(state, action_idx, reward, next_state, float(done))

            state = next_state
            total_reward += reward
            step_count += 1
            step_in_ep += 1

            # 從 Buffer 取樣並更新
            if len(buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)

                # 當前網路對所有 action 的 Q 值: shape [batch, action_dim]
                q_all = model(states_b)

                # 正確 DQN：只更新被執行的那個 action
                # 先複製一份 Q 值作為 target（未被執行的 action 不產生梯度差）
                with torch.no_grad():
                    # Target Network 預測下個狀態的最大 Q 值
                    q_next = target_model(next_states_b)          # [batch, action_dim]
                    max_q_next = q_next.max(dim=1).values             # [batch]
                    td_target  = rewards_b + gamma * max_q_next * (1.0 - dones_b)

                # 建立 target tensor：複製當前預測，只改被選中的 action
                q_target = q_all.detach().clone()                     # [batch, action_dim]
                q_target[torch.arange(batch_size), actions_b] = td_target

                loss = criterion(q_all, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ── 定期同步 Target Network ───────────────────
            if step_count % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

            # 每100步印出一次資料
            step_in_ep += 1
            if step_in_ep % 100 == 0:
                print(f"Step:{step_in_ep:4d} | Reward:{reward:+.4f} | value:{env.portfolio_value:.3f}")

        # Epsilon 衰減(每 episode 結束後)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f"[Episode{ep+1:3d}/{episodes}] Total Reward:{total_reward:+.4f} | value:{env.portfolio_value:.3f} | ε:{epsilon:.4f}\n")

        # 每一個Episode 跑一次驗證
        ret = valid_dqn(model=model, env=TradingEnv(valid_df))
        if ret > best_ret:
            best_ret = ret
            # 儲存valid 成績最好的model
            torch.save(model.state_dict(), FILE_best)
            print("Best model Episode:", ep+1)

        if ep + 1 == episodes:
            # 儲存最後一個model
            torch.save(model.state_dict(), FILE_end)
            print("End model saved\n")

        train_log.append({
            "Episode": ep + 1,
            "Total_Reward": round(total_reward, 4),
            "Portfolio_Value": round(env.portfolio_value, 3),
            "Epsilon": round(epsilon, 4)
        })

    # 儲存訓練結果成csv
    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv('train_result/log/DQN_train.csv', index=False)
    # 儲存驗證結果
    test_log_df = pd.DataFrame(valid_log)
    test_log_df.to_csv("train_result/log/DQN_valid.csv", index=False)

    return model

# DQN 驗證
def valid_dqn(model, env=None):
    if env is None:
        env = TradingEnv(valid_df)

    model = model.eval()

    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.numpy()[0]

        next_state, reward, done = env.step(action)

        total_reward += reward
        step_count += 1

         # 每 100 步印一次
        if step_count % 100 == 0:
            print(f"[Valid] Step:{step_count:4d} | Reward:{reward:+.4f} | Value:{env.portfolio_value:.3f}")

        state = next_state

    final_value = env.portfolio_value
    ret = (final_value - env.initial_cash) / env.initial_cash * 100  # 報酬率 % (淨損益/投資成本 * 100)
    
    print(f"Total Reward:{total_reward:+.4f} | Final Value:{final_value:.3f} | Return:{ret:+.2f}%\n")

    valid_log.append({
        "Total_Reward": round(total_reward, 4),
        "Final_value": round(env.portfolio_value, 3),
        "Return_pct(%)": round(ret, 2)
    })

    return ret

# DQN 測試
def test_dqn(model, name, env=None):
    if env is None:
        env = TradingEnv(test_df)

    model = model.eval()

    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.numpy()[0]

        next_state, reward, done = env.step(action)

        total_reward += reward
        step_count += 1

         # 每 100 步印一次
        if step_count % 100 == 0:
            print(f"[Test] Step:{step_count:4d} | Reward:{reward:+.4f} | Value:{env.portfolio_value:.3f}")

        state = next_state

    final_value = env.portfolio_value
    ret = (final_value - env.initial_cash) / env.initial_cash * 100  # 報酬率 % (淨損益/投資成本 * 100)
    
    print(f"Total Reward:{total_reward:+.4f} | Final Value:{final_value:.3f} | Return:{ret:+.2f}%\n")

    test_log.append({
        "Total_Reward": round(total_reward, 4),
        "Final_value": round(env.portfolio_value, 3),
        "Return_pct(%)": round(ret, 2)
    })

    # 儲存驗證結果
    test_log_df = pd.DataFrame(test_log)
    test_log_df.to_csv(f"train_result/log/{name}.csv", index=False)

    return 0

# 開始訓練
environment = TradingEnv(train_df)
trained_model = train_dqn(env=environment)
# best model 測試一次
load_best_model = QNetwork(state_dim, action_dim)
load_best_model.load_state_dict(torch.load(FILE_best))
test_dqn(model=load_best_model,name="DQN_best_test", env=environment)

# 重置test_log
if test_log:
    test_log = []
# 最後一個Episode model 測試一次
test_dqn(model=trained_model,name="DQN_test", env=environment)