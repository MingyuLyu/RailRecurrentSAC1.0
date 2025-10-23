import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box

class BehaviorCloning(Env):
    """
    离线 (s, a*) 数据的“滚动”环境（用于 Actor 的 BC 预训练 / warm-start）：
    - 观测按与真实环境一致的 Min-Max 方式归一化到 [-1, 1]
    - 每个 episode 从随机起点抽一段连续序列（长度 episode_len）
    - 奖励 = -MSE(action, expert_action)
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 episode_len: int = 128,
                 use_terminal_mask: bool = False,
                 pre_normalize: bool = True,
                 eps: float = 1e-8):
        super().__init__()

        # ---------------- 读取专家数据 ----------------
        csv_path = r"C:\Users\root\Documents\GitHub\RailRecurrentSAC1.0\mass_360.csv"
        data = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        assert data.shape[1] >= 7, "CSV 至少需要 7 列（前 6 列 obs，最后 1 列 expert action）"
        # 观测顺序必须与真实环境一致：
        # [distance_left, time_left, velocity, speed_limit, distance_to_next_change, next_speed_limit]
        self.obs_raw = data[:, :6]               # [N, 6]
        self.act_raw = data[:, 6].reshape(-1)    # [N,]
        self.N, self.obs_dim = self.obs_raw.shape
        self.act_dim = 1

        # ---------------- 和真实环境一致的 min/max ----------------
        # 这些值要和你的 TrainSpeedControl2 保持一致！
        track_length   = 2500.0
        station        = 2000.0
        max_speed      = 30.0     # ≈ 108 km/h（你代码里就是 30）
        episode_time   = 200.0
        sensor_range   = 500.0

        self.state_min = np.array([0.0, 0.0,       0.0,      0.0,       0.0,        0.0], dtype=np.float32)
        self.state_max = np.array([track_length,
                                   episode_time,
                                   max_speed,
                                   max_speed,   # speed_limit 的上界按 max_speed 设就行
                                   sensor_range,
                                   max_speed],  dtype=np.float32)

        # 预计算分母，防除零
        self._denom = np.maximum(self.state_max - self.state_min, eps).astype(np.float32)

        # ---------------- 动作范围（如专家不在 [-1,1]，可线性缩放到 [-1,1]） ----------------
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._act_min_assumed = -1.0
        self._act_max_assumed =  1.0
        # 如需把专家动作线性缩放到 [-1,1]，解除注释：
        # a_min, a_max = self.act_raw.min(), self.act_raw.max()
        # self.act_raw = 2.0 * (self.act_raw - a_min) / max(a_max - a_min, eps) - 1.0

        # ---------------- 观测空间（已归一化到 [-1,1]） ----------------
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # ---------------- episode 抽样策略 ----------------
        self.episode_len = int(episode_len)
        self.ptr = 0
        self.use_terminal_mask = bool(use_terminal_mask)

        if self.use_terminal_mask:
            # 假设第 0 维是 distance_left；只在末端（<300 m）抽样
            term_mask = (self.obs_raw[:, 0] < 300.0)
            self.valid_idx = np.nonzero(term_mask)[0]
            if len(self.valid_idx) < self.episode_len + 1:
                # 兜底：末段样本太少则改用全量
                self.valid_idx = np.arange(self.N, dtype=np.int32)
        else:
            self.valid_idx = np.arange(self.N, dtype=np.int32)

        # ---------------- 预归一化（提速可选） ----------------
        self._pre_normalize = bool(pre_normalize)
        if self._pre_normalize:
            self.obs_norm = self._normalize_obs(self.obs_raw)  # [N, 6]
        else:
            self.obs_norm = None

    # ---------- 归一化/反归一化 ----------
    def _normalize_obs(self, s: np.ndarray) -> np.ndarray:
        """按 env 的 min-max 规范化到 [-1, 1]；支持 [6] 或 [N,6]。"""
        return (2.0 * (s - self.state_min) / self._denom - 1.0).astype(np.float32)

    def _denormalize_obs(self, s_norm: np.ndarray) -> np.ndarray:
        """从 [-1,1] 反归一化回原尺度（一般用不到，这里留个工具）。"""
        return ((s_norm + 1.0) * 0.5 * self._denom + self.state_min).astype(np.float32)

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 从 valid 区随机挑可以容纳 episode_len 的起点（保证还有 episode_len+1 个索引）
        hi = len(self.valid_idx) - self.episode_len - 1
        if hi <= 0:
            start = np.random.randint(0, max(self.N - self.episode_len - 1, 1))
        else:
            k = np.random.randint(0, hi + 1)
            start = int(self.valid_idx[k])

        # 序列索引：长度 L+1，便于 step 后取 next_obs
        end = min(start + self.episode_len + 1, self.N)
        self.indices = np.arange(start, end, dtype=np.int32)
        # 如尾部不够长，回绕补足（很少发生，做个兜底）
        if len(self.indices) < self.episode_len + 1:
            extra = (self.episode_len + 1) - len(self.indices)
            self.indices = np.concatenate([self.indices, np.arange(extra, dtype=np.int32)])

        self.ptr = 0

        s0 = self.obs_norm[self.indices[self.ptr]] if self._pre_normalize \
             else self._normalize_obs(self.obs_raw[self.indices[self.ptr]])

        info = {}
        return s0.astype(np.float32), info

    def step(self, action):
        idx = int(self.indices[self.ptr])

        # 专家动作（假定已在 [-1,1]）
        a_star = float(np.clip(self.act_raw[idx], -1.0, 1.0))
        # agent 动作
        a = float(np.clip(action, -1.0, 1.0)[0])

        # 奖励：-MSE
        mse = (a - a_star) ** 2
        reward = -float(mse)

        # 前进一步
        self.ptr += 1
        terminated = False
        truncated = (self.ptr >= self.episode_len)

        next_idx = int(self.indices[min(self.ptr, self.episode_len)])
        next_obs = self.obs_norm[next_idx] if self._pre_normalize \
                   else self._normalize_obs(self.obs_raw[next_idx])

        # print(next_obs)

        info = {"idx": idx, "expert_action": a_star, "pred_action": a, "mse": float(mse)}
        return next_obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        pass
