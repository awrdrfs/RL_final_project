import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# 超參數
episodes = 100
share_num = 1
gamma = 0.99
learning_rate = 0.001
target_update_freq = 100
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.95

# Q-Transformer 專用超參數
ACTION_BINS = 21    # 每個 action 維度離散化成幾個 bins（連續值 → 類別）
TRANSFORMER_DIM = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_DEPTH = 2
DROPOUT = 0.1

# 維度計算
state_dim = 5 * share_num + (share_num + 1)  # OHLCV + 持股數 + 現金
action_dim = share_num + 1                     # 股票權重 + 現金權重（autoregressive 解碼維度數）

# 載入資料集
train_df = pd.read_csv("dataset/train/DIS_wo_prepost.csv")
valid_df = pd.read_csv("dataset/validTest/DIS_wo_prepost.csv")
test_df  = pd.read_csv("dataset/validTest/DIS_wo_prepost.csv")

# 儲存訓練資料
train_log = []
valid_log = []
test_log  = []

# 儲存模型路徑
model_folder = 'train_result/model/Qtrans_DIS_woReplayBuffer_vt'
FILE_best = f'{model_folder}/best_model_state_dict.pt'
FILE_end = f'{model_folder}/100episode_model_state_dict.pt'

# 儲存log 路徑
log_folder = 'train_result/log/Qtransformer_DIS_woReplayBuffer_vt'
log_train = f'{log_folder}/train.csv'
log_valid = f'{log_folder}/valid.csv'
log_best_test = f'{log_folder}/best_test.csv'
log_test = f'{log_folder}/test.csv'

# 交易環境（與原版相同，不動）
class TradingEnv:
    def __init__(self, df, initial_cash=1000, transaction_rate=0.003):
        self.df = df
        self.initial_cash = initial_cash
        self.transaction_rate = transaction_rate
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = np.array([0.0])
        self.portfolio_value = self.initial_cash
        self.old_value = self.initial_cash
        return self._get_observation()

    def _get_observation(self):
        obs = self.df.iloc[self.current_step, 1:].values.tolist()
        obs.extend(self.shares.tolist())
        obs.append(self.cash)
        return np.array(obs, dtype=np.float32)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def step(self, action):
        new_action = self._softmax(action)
        share_rates = new_action[:1]
        current_prices = np.array([self.df.iloc[self.current_step]['Close']])

        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True

        next_open_prices = np.array([self.df.iloc[self.current_step + 1]['Open']])
        self.portfolio_value = np.sum(self.shares * current_prices) + self.cash
        old_share_rates = (self.shares * current_prices) / self.portfolio_value
        delta_shares = share_rates - old_share_rates

        for i in range(len(self.shares)):
            trade_value = delta_shares[i] * self.portfolio_value
            price = next_open_prices[i]
            break_rule_punish = 0.0

            if trade_value > 0:
                fee = trade_value * self.transaction_rate
                if self.cash >= trade_value + fee:
                    self.cash -= (trade_value + fee)
                    self.shares[i] += trade_value / price
                else:
                    affordable = self.cash / (1 + self.transaction_rate)
                    fee = affordable * self.transaction_rate
                    self.cash -= (affordable + fee)
                    self.shares[i] += affordable / price
                    break_rule_punish = -1.0
            else:
                pos = abs(trade_value)
                max_sellable = self.shares[i] * price
                if pos > max_sellable:
                    pos = max_sellable
                    break_rule_punish = -1.0
                fee = pos * self.transaction_rate
                self.cash += (pos - fee)
                self.shares[i] -= pos / price

        new_value = np.sum(self.shares * next_open_prices) + self.cash
        reward = (new_value - self.old_value) / self.old_value + break_rule_punish
        self.old_value = new_value
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done

# Action Tokenizer：連續 action 轉離散 bin 值
class ActionTokenizer:
    """把 softmax 後的 action 值(0~1)離散化成 [0, ACTION_BINS-1] 的整數索引，並能反向解碼回連續值供環境使用。"""
    def __init__(self, num_bins: int = ACTION_BINS, low: float = 0.0, high: float = 1.0):
        self.num_bins = num_bins
        self.low  = low
        self.high = high
        # 每個 bin 的中心值（解碼用）
        self.bin_centers = np.linspace(low, high, num_bins)

    def encode(self, action: np.ndarray) -> np.ndarray:
        # 連續值轉成 bin 值(int)
        clipped = np.clip(action, self.low, self.high)
        indices = ((clipped - self.low) / (self.high - self.low) * (self.num_bins - 1))
        return np.round(indices).astype(np.int64)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        # bin 值轉回連續值
        return self.bin_centers[indices].astype(np.float32)

# Q-Transformer 模型
# 設計參考：
#   "Q-Transformer: Scalable Offline RL via Autoregressive Q-Functions"
#   (Chebotar et al., CoRL 2023)
# 架構：
#   1. 狀態編碼器（Linear → Transformer Encoder）
#   2. 自迴歸解碼器（一次預測一個 action 維度的 Q 值分布）
#   3. Dueling Head（Value + Advantage 分離）
class QTransformerModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bins: int = ACTION_BINS,
        dim: int = TRANSFORMER_DIM,
        depth: int = TRANSFORMER_DEPTH,
        heads: int = TRANSFORMER_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_bins = action_bins
        self.dim = dim

        # 1. 狀態編碼器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

        # 2. Action Token Embedding（自迴歸輸入)
        # 每個 action dimension 都有獨立的 bin embedding
        # 解碼第 t 個 action 時，把前 t-1 個已選 bin 的 embedding concat 進去
        self.action_token_emb = nn.Embedding(action_bins + 1, dim)  # +1 for BOS token
        self.BOS_TOKEN = action_bins  # 用 action_bins 當作開始符號

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # Pre-LN（更穩定）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        #4. Dueling Head（每個 action 維度獨立）
        # Value head：scalar
        self.value_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1))
            for X in range(action_dim) #建立action_dim 個value_head
        ])
        # Advantage head：action_bins 個值
        self.advantage_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, action_bins))
            for X in range(action_dim) #建立action_dim 個advantage_head
        ])

    def _dueling_q(self, features: torch.Tensor, action_idx: int) -> torch.Tensor:
        """用 Dueling 架構計算第 action_idx 個 action 維度的 Q 值分布。
        Args:
            features: [B, dim]
        Returns:
            q_values: [B, action_bins]
        """
        v = self.value_heads[action_idx](features)          # [B, 1]
        adv = self.advantage_heads[action_idx](features)       # [B, action_bins]
        q = v + adv - adv.mean(dim=-1, keepdim=True)        # Dueling formula
        return q

    def forward(
        self,
        state: torch.Tensor,              # [B, state_dim]
        prev_action_tokens: torch.Tensor  # [B, t]已選的 bin indices（含 BOS）
    ) -> torch.Tensor:
        """
        自迴歸推理：給定狀態和前 t 個 action tokens，輸出第 t+1 個 action 的 Q 值分布。
        Returns:
            q_values: [B, action_bins]
        """
        B = state.shape[0]

        # 狀態轉成 token
        state_token = self.state_encoder(state).unsqueeze(1)   # [B, 1, dim]

        # 歷史動作轉成 tokens
        act_tokens = self.action_token_emb(prev_action_tokens)  # [B, t, dim]

        # 拼接：[state | a0 | a1 | ... | a_{t-1}]
        seq = torch.cat([state_token, act_tokens], dim=1)      # [B, 1+t, dim]

        # Transformer encoding（含 causal mask 讓 action 只看左邊）
        T = seq.shape[1]
        mask = torch.triu(torch.ones(T, T, device=state.device), diagonal=1).bool() # triu:保留上三角的數值
        out = self.transformer(seq, mask=mask)                # [B, 1+t, dim]

        # 取最後一個 token 的特徵來預測下一個 action 的 Q 值
        last = out[:, -1, :]                                   # [B, dim]

        # 目前解碼到第幾個 action（BOS 不算，所以 -1）
        action_idx = prev_action_tokens.shape[1] - 1
        action_idx = max(0, min(action_idx, self.action_dim - 1))

        return self._dueling_q(last, action_idx)               # [B, action_bins]

    @torch.no_grad()
    def get_all_action_q(self, state: torch.Tensor) -> list[torch.Tensor]:
        """自迴歸地對所有 action 維度計算 greedy 最大 Q 值的 bin index。
        Returns:
            q_list: list of [B, action_bins], 長度 = action_dim
        """
        B = state.shape[0]
        # BOS token
        tokens = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=state.device)
        q_list = []

        for X in range(self.action_dim):
            q = self.forward(state, tokens)    # [B, action_bins]
            q_list.append(q)
            best_bin = q.argmax(dim=-1, keepdim=True)  # [B, 1]
            tokens = torch.cat([tokens, best_bin], dim=1)

        return q_list  # length == action_dim

    @torch.no_grad()
    def get_action_bins(self, state: torch.Tensor, epsilon: float = 0.0) -> np.ndarray:
        """回傳每個 action 維度的 bin index(numpy)。"""
        B = state.shape[0]
        tokens = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=state.device)
        bins = []

        for X in range(self.action_dim):
            q = self.forward(state, tokens)
            if random.random() < epsilon:
                best = torch.randint(0, self.action_bins, (B, 1), device=state.device)
            else:
                best = q.argmax(dim=-1, keepdim=True)
            bins.append(best.squeeze(-1).cpu().numpy())
            tokens = torch.cat([tokens, best], dim=1)

        return np.stack(bins, axis=-1)  # [B, action_dim]

# Q-Transformer 訓練
def train_qtransformer(env=None):
    if env is None:
        env = TradingEnv(train_df)

    tokenizer = ActionTokenizer(num_bins=ACTION_BINS)
    model = QTransformerModel(state_dim, action_dim)
    target_model = QTransformerModel(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    criterion = nn.SmoothL1Loss()   # Huber loss 比 MSE 更穩定
    epsilon = epsilon_start
    step_count = 0
    best_ret = float('-inf')

    for ep in range(episodes):
        model.train()
        state = env.reset()
        done = False
        total_reward = 0.0
        step_in_ep = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]

            # 自迴歸選 action（epsilon-greedy）
            action_bin_indices = model.get_action_bins(state_tensor, epsilon=epsilon)[0]
            # bin index 轉連續值、轉環境使用
            action_continuous = tokenizer.decode(action_bin_indices)

            next_state, reward, done = env.step(action_continuous)

            # Q-Transformer 論文的 Autoregressive TD Target
            # 前 n-1 個 action token：reward = 0（只是自迴歸結構的一部份）
            # 最後一個 action token：reward = 實際 reward
            # 在此 action_dim=2，所以第 0 個 token 給 0，第 1 個給 reward
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # 對每個 action 維度個別計算 loss
            total_loss = torch.tensor(0.0)

            # 建立 BOS + 已選 tokens 的前綴
            bos = torch.full((1, 1), model.BOS_TOKEN, dtype=torch.long)
            chosen_bins = torch.LongTensor(action_bin_indices).unsqueeze(0)  # [1, action_dim]

            for t in range(action_dim):
                # 前綴：BOS + a_0 ... a_{t-1}
                prefix = torch.cat([bos, chosen_bins[:, :t]], dim=1)

                # 當前 Q 值
                q_all = model(state_tensor, prefix)      # [1, ACTION_BINS]
                q_chosen = q_all[0, action_bin_indices[t]]  # scalar

                # TD Target（用 target network）
                if t == (action_dim - 1):
                    is_last_action = True
                else:
                    is_last_action = False

                if is_last_action: # 只有最後一個 action 給真實 reward
                    r_t = reward
                else:
                    r_t = 0.0

                with torch.no_grad():
                    next_prefix = torch.cat([bos, chosen_bins[:, :t]], dim=1)
                    # target network 自迴歸選最優 next bin
                    next_q_all = target_model(next_state_tensor, next_prefix)
                    max_next_q = next_q_all.max(dim=-1).values.item()
                    td_target = r_t + gamma * max_next_q * (1.0 - float(done))

                loss_t = criterion(q_chosen.unsqueeze(0), torch.tensor([td_target]))
                total_loss += loss_t

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪。若norm>1，則縮小到1
            optimizer.step()

            # 定期同步 Target Network
            step_count += 1
            if step_count % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

            state = next_state
            total_reward += reward
            step_in_ep += 1

            if step_in_ep % 100 == 0:
                print(f"Step:{step_in_ep:4d} | Reward:{reward:+.4f} | value:{env.portfolio_value:.3f} | loss:{total_loss.item():.4f}")

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"[Episode{ep+1:3d}/{episodes}] Total Reward:{total_reward:+.4f} | value:{env.portfolio_value:.3f} | ε:{epsilon:.4f}\n")

        # 每 episode 驗證一次
        ret = valid_qtransformer(model=model, env=TradingEnv(valid_df))
        if ret > best_ret:
            best_ret = ret
            torch.save(model.state_dict(), FILE_best)
            print("Best model Episode:", ep + 1)

        if ep + 1 == episodes:
            torch.save(model.state_dict(), FILE_end)
            print("End model saved\n")

        train_log.append({
            "Episode": ep + 1,
            "Total_Reward": round(total_reward, 4),
            "Portfolio_Value": round(env.portfolio_value, 3),
            "Epsilon": round(epsilon, 4),
        })

    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(log_train, index=False)
    valid_log_df = pd.DataFrame(valid_log)
    valid_log_df.to_csv(log_valid, index=False)

    return model

# 驗證
def valid_qtransformer(model, env=None):
    if env is None:
        env = TradingEnv(valid_df)

    tokenizer = ActionTokenizer(num_bins=ACTION_BINS)
    model.eval()

    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_bins = model.get_action_bins(state_tensor, epsilon=0.0)[0]
        action = tokenizer.decode(action_bins)

        next_state, reward, done = env.step(action)
        total_reward += reward
        step_count += 1

        if step_count % 100 == 0:
            print(f"[Valid] Step:{step_count:4d} | Reward:{reward:+.4f} | Value:{env.portfolio_value:.3f}")

        state = next_state

    final_value = env.portfolio_value
    ret = (final_value - env.initial_cash) / env.initial_cash * 100

    print(f"Total Reward:{total_reward:+.4f} | Final Value:{final_value:.3f} | Return:{ret:+.2f}%\n")

    valid_log.append({
        "Total_Reward": round(total_reward, 4),
        "Final_value": round(env.portfolio_value, 3),
        "Return_pct(%)": round(ret, 2),
    })

    return ret

# 測試
def test_qtransformer(model, name, env=None):
    if env is None:
        env = TradingEnv(test_df)

    tokenizer = ActionTokenizer(num_bins=ACTION_BINS)
    model.eval()

    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_bins = model.get_action_bins(state_tensor, epsilon=0.0)[0]
        action = tokenizer.decode(action_bins)

        next_state, reward, done = env.step(action)
        total_reward += reward
        step_count += 1

        if step_count % 100 == 0:
            print(f"[Test] Step:{step_count:4d} | Reward:{reward:+.4f} | Value:{env.portfolio_value:.3f}")

        state = next_state

    final_value = env.portfolio_value
    ret = (final_value - env.initial_cash) / env.initial_cash * 100

    print(f"Total Reward:{total_reward:+.4f} | Final Value:{final_value:.3f} | Return:{ret:+.2f}%\n")

    test_log.append({
        "Total_Reward":  round(total_reward, 4),
        "Final_value":   round(env.portfolio_value, 3),
        "Return_pct(%)": round(ret, 2),
    })

    test_log_df = pd.DataFrame(test_log)
    test_log_df.to_csv(name, index=False)

    return 0

# 執行
environment = TradingEnv(train_df)
trained_model = train_qtransformer(env=environment)

environment = TradingEnv(test_df)

# best model 測試
load_best_model = QTransformerModel(state_dim, action_dim)
load_best_model.load_state_dict(torch.load(FILE_best))
test_qtransformer(model=load_best_model, name=log_best_test, env=environment)

# 重置 test_log，跑最後一個 episode 的 model
if test_log:
    test_log = []
test_qtransformer(model=trained_model, name=log_test, env=environment)
