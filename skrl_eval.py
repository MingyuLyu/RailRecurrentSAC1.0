# evaluate_skrl_agent.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # quick fix for the OMP crash
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # optional: quiet/simplify TF kernels

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from rail_env import TrainSpeedControl2
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC_RNN as SAC
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from rail_env import TrainSpeedControl2

# ===== 重要：用你训练时的 Actor / Critic 实现 =====
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=400, sequence_length=128):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.linear_layer_1 = nn.Linear(self.hidden_size, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self._dbg_times = 0

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        # states = inputs["states"]
        # terminated = inputs.get("terminated", None)
        # hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        states = inputs["states"]
        terminated = inputs.get("terminated", None)

        # ---- RNN 状态健壮获取 + 调试打印（只打前 5 次，避免刷屏）----
        rnn_in = inputs.get("rnn", None)

        if not hasattr(self, "_dbg_times"):
            self._dbg_times = 0
        dbg = self._dbg_times < 5

        if dbg:
            print(f"[DBG][{self.__class__.__name__}] training={self.training} role={role} "
                  f"states.shape={tuple(states.shape)} "
                  f"has_rnn={(isinstance(rnn_in, (list, tuple)) and len(rnn_in) == 2)} "
                  f"keys={list(inputs.keys())}")

        if isinstance(rnn_in, (list, tuple)) and len(rnn_in) == 2:
            hidden_states, cell_states = rnn_in[0], rnn_in[1]
        else:
            # 没传 rnn：按当前批次维度零初始化（不会崩，并能看到 has_rnn=False）
            if self.training:
                assert states.dim() == 2, "expect (N*L, obs_dim) during training"
                N = states.shape[0] // self.sequence_length
                size = (self.num_layers, N * self.sequence_length, self.hidden_size)
            else:
                N = states.shape[0]  # rollout: (num_envs, obs_dim)
                size = (self.num_layers, N, self.hidden_size)

            hidden_states = torch.zeros(size, device=states.device, dtype=states.dtype)
            cell_states = torch.zeros(size, device=states.device, dtype=states.dtype)

        if dbg:
            print(f"[DBG][{self.__class__.__name__}] h.shape={tuple(hidden_states.shape)} "
                  f"c.shape={tuple(cell_states.shape)}")
            self._dbg_times += 1
        # ---------------------------------------------------------

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = F.relu(self.linear_layer_1(rnn_output))
        x = F.relu(self.linear_layer_2(x))

        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}
        # 通用动作缩放：tanh 输出 ∈ [-1, 1] → 线性映射到 [low, high]
        # raw = torch.tanh(self.action_layer(x))  # (N*L, num_actions)
        #
        # act_low = torch.as_tensor(self.action_space.low, device=raw.device, dtype=raw.dtype)
        # act_high = torch.as_tensor(self.action_space.high, device=raw.device, dtype=raw.dtype)
        #
        # mean = (act_high + act_low) / 2 + raw * (act_high - act_low) / 2
        # return mean, self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=400, sequence_length=128):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.linear_layer_1 = nn.Linear(self.hidden_size + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

        self._dbg_times = 0

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        # states = inputs["states"]
        # terminated = inputs.get("terminated", None)
        # hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        states = inputs["states"]
        terminated = inputs.get("terminated", None)

        # ---- RNN 状态健壮获取 + 调试打印（只打前 5 次，避免刷屏）----
        rnn_in = inputs.get("rnn", None)

        if not hasattr(self, "_dbg_times"):
            self._dbg_times = 0
        dbg = self._dbg_times < 5

        if dbg:
            print(f"[DBG][{self.__class__.__name__}] training={self.training} role={role} "
                  f"states.shape={tuple(states.shape)} "
                  f"has_rnn={(isinstance(rnn_in, (list, tuple)) and len(rnn_in) == 2)} "
                  f"keys={list(inputs.keys())}")

        if isinstance(rnn_in, (list, tuple)) and len(rnn_in) == 2:
            hidden_states, cell_states = rnn_in[0], rnn_in[1]
        else:
            # 没传 rnn：按当前批次维度零初始化（不会崩，并能看到 has_rnn=False）
            if self.training:
                assert states.dim() == 2, "expect (N*L, obs_dim) during training"
                N = states.shape[0] // self.sequence_length
                size = (self.num_layers, N * self.sequence_length, self.hidden_size)
            else:
                N = states.shape[0]  # rollout: (num_envs, obs_dim)
                size = (self.num_layers, N, self.hidden_size)

            hidden_states = torch.zeros(size, device=states.device, dtype=states.dtype)
            cell_states = torch.zeros(size, device=states.device, dtype=states.dtype)

        if dbg:
            print(f"[DBG][{self.__class__.__name__}] h.shape={tuple(hidden_states.shape)} "
                  f"c.shape={tuple(cell_states.shape)}")
            self._dbg_times += 1
        # ---------------------------------------------------------

        # critic is only used during training
        rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

        hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
        cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
        # get the hidden/cell states corresponding to the initial sequence
        sequence_index = 1 if role in ["target_critic_1", "target_critic_2"] else 0  # target networks act on the next state of the environment
        hidden_states = hidden_states[:,:,sequence_index,:].contiguous()  # (D * num_layers, N, Hout)
        cell_states = cell_states[:,:,sequence_index,:].contiguous()  # (D * num_layers, N, Hcell)

        # reset the RNN state in the middle of a sequence
        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = terminated.view(-1, self.sequence_length)
            indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

            for i in range(len(indexes) - 1):
                i0, i1 = indexes[i], indexes[i + 1]
                rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                hidden_states[:, (terminated[:,i1-1]), :] = 0
                cell_states[:, (terminated[:,i1-1]), :] = 0
                rnn_outputs.append(rnn_output)

            rnn_states = (hidden_states, cell_states)
            rnn_output = torch.cat(rnn_outputs, dim=1)
        # no need to reset the RNN state in the sequence
        else:
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = F.relu(self.linear_layer_1(torch.cat([rnn_output, inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))

        return self.linear_layer_3(x), {"rnn": [rnn_states[0], rnn_states[1]]}

def build_agent(env, device, checkpoint_path):
    """构建 SAC_RNN agent 并加载 checkpoint"""
    # 和训练时完全一致的 5 个模型
    models = {
        "policy":          Actor(env.observation_space, env.action_space, device, clip_actions=True, num_envs=env.num_envs),
        "critic_1":        Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs),
        "critic_2":        Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs),
        "target_critic_1": Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs),
        "target_critic_2": Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs),
    }

    cfg = SAC_DEFAULT_CONFIG.copy()   # 评估不依赖这些训练超参，但需要一个 cfg
    agent = SAC(models=models,
                memory=None,  # 评估不需要 replay
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    # 加载训练得到的权重（路径替换成你的）
    agent.load(checkpoint_path)
    agent.init()
    # 评估模式：确定性动作（高斯取均值），且不更新参数
    agent.set_running_mode("eval")
    # 同时把底层 torch 模型也设成 eval（BN/Dropout 等关闭；你这套一般没用到）
    for m in models.values():
        m.eval()
    return agent

def evaluate_once(env, agent, seed=None, render=False):
    """评估单个 episode；返回日志 dict（和你的原始代码字段对齐）"""
    if render:
        try:
            env.unwrapped.render_mode = "human"
        except Exception:
            pass

    # Gymnasium reset: (obs, info)
    state, _ = env.reset()
    done = False
    t = 0

    positions, velocities, accelerations, jerks = [], [], [], []
    times, powers, rewards, actions_safe, actions_raw, energy = [], [], [], [], [], []
    total_reward = 0.0
    start = datetime.now()

    with torch.no_grad():
        while not done:
            # agent.act 期望 torch.Tensor（形状 [num_envs, obs_dim]）
            state_t = torch.as_tensor(state, device=env.device).unsqueeze(0)
            # 产生动作（确定性；如需随机评估可改 agent.set_running_mode("explore")）
            action_t, _, _ = agent.act(state_t, timestep=t, timesteps=0)
            # action = action_t.cpu().numpy()           # -> (num_envs, act_dim)
            # action = action_t
            # print(action)
            # 环境 step：skrl 的 wrap_env 统一成批维，返回 shape 也带批维
            next_state, reward, terminated, truncated, info = env.step(action_t)

            done = bool(terminated[0] or truncated[0])
            total_reward += float(reward[0])

            # 你环境在 info 里塞了各类物理量，这里按你的键取
            positions.append(info['position'])
            velocities.append(info['velocity'])
            accelerations.append(info['acceleration'])
            jerks.append(info['jerk'])
            times.append(info['time'])
            powers.append(info['power'])
            rewards.append(info['reward'])
            actions_safe.append(info['action_safe'])
            actions_raw.append(info['action_raw'])
            energy.append(info['energy'])

            state = next_state
            t += 1

    elapsed = (datetime.now() - start).total_seconds()
    print(f"Episode return: {total_reward:.3f} | time: {elapsed:.3f}s | final energy: {energy[-1]}")

    log = {
        "time": times, "position": positions, "velocity": velocities,
        "acceleration": accelerations, "jerk": jerks, "power": powers,
        "energy": energy, "reward": rewards, "action_safe": actions_safe, "action_raw": actions_raw,
        "episode_return": total_reward, "elapsed_sec": elapsed
    }
    return log

def plot_curves(log):
    # 你的原图保持
    plt.figure(0, figsize=(8,6))
    plt.plot(log["position"], log["velocity"], label='Velocity-Position plot')
    plt.xlabel('Position [m]'); plt.ylabel('Velocity [m/s]'); plt.title('Velocity vs Position'); plt.legend(); plt.grid(True)

    plt.figure(1)
    plt.plot(log["time"], log["velocity"], label='Velocity-Time plot')
    plt.xlabel('Time'); plt.ylabel('Velocity'); plt.title('Velocity-Time plot'); plt.legend(); plt.grid(True)

    plt.figure(2)
    plt.plot(log["time"], log["action_safe"], label='Action_safe-Time plot')
    plt.xlabel('Time'); plt.ylabel('Action'); plt.title('Action-Time plot'); plt.legend(); plt.grid(True)

    plt.figure(3)
    plt.plot(log["time"][:min(138, len(log["time"]))], log["reward"][:min(138, len(log["reward"]))], label='Rewards-Time plot')
    plt.xlabel('Time'); plt.ylabel('Rewards'); plt.title('Rewards-Time plot'); plt.legend(); plt.grid(True)

    plt.figure(4)
    plt.plot(log["time"], log["action_raw"], label='Action_raw-Time plot')
    plt.xlabel('Time'); plt.ylabel('Action'); plt.title('Action-Time plot'); plt.legend(); plt.grid(True)

    plt.show()

def save_csv(log, out_name="skrl_eval_log.csv"):
    df = pd.DataFrame({
        "time": log["time"],
        "position": log["position"],
        "velocity": log["velocity"],
        "acceleration": log["acceleration"],
        "jerk": log["jerk"],
        "power": log["power"],
        "energy": log["energy"],
        "reward": log["reward"],
        "action_safe": log["action_safe"],
        "action_raw": log["action_raw"],
    })
    df.to_csv(out_name, index=False)
    print(f"Saved ➜ {out_name}  ({len(df)} rows)")

if __name__ == "__main__":
    # 1) 实例化你的环境（保持与训练一致）
    env = wrap_env(TrainSpeedControl2())
    device = env.device
    print("Device:", device)
    print("Obs space:", env.observation_space, "Act space:", env.action_space)

    # 2) 构建 agent 并加载 checkpoint（把路径换成你的 skrl 训练保存目录）
    checkpoint = r"C:\Users\root\Documents\GitHub\RailRecurrentSAC1.0\runs\torch\LSTMSAC_TrainSpeedControl2\25-10-10_21-23-05-356690_SAC_RNN\checkpoints\agent_200000.pt"
    agent = build_agent(env, device, checkpoint)

    # 3) 评估一回合（也可以写个循环评多个）
    log = evaluate_once(env, agent, seed=0, render=False)
    # print(log)
    # 4) 可视化 & 保存
    plot_curves(log)
    # print("plot over")
    save_csv(log, out_name="skrl_eval_log.csv")
