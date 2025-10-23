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
from bc_env import BehaviorCloning


# ======= patch: robust load + eval (trainer-style) =======

class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=128, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.linear_layer_1 = nn.Linear(self.hidden_size, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer   = nn.Linear(300, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])[:,:,0,:].contiguous()
            cell_states   = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])[:,:,0,:].contiguous()

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                idx = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]
                for i in range(len(idx) - 1):
                    i0, i1 = idx[i], idx[i + 1]
                    rnn_out, (hidden_states, cell_states) = self.lstm(rnn_input[:, i0:i1, :], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:, i1-1]), :] = 0
                    cell_states[:, (terminated[:, i1-1]), :] = 0
                    rnn_outputs.append(rnn_out)
                rnn_output = torch.cat(rnn_outputs, dim=1)
                rnn_states = (hidden_states, cell_states)
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        rnn_output = torch.flatten(rnn_output, 0, 1)
        x = F.relu(self.linear_layer_1(rnn_output))
        x = F.relu(self.linear_layer_2(x))
        mu = torch.tanh(self.action_layer(x))
        return mu, self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=128, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.linear_layer_1 = nn.Linear(self.hidden_size + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        rnn_input = states.view(-1, self.sequence_length, states.shape[-1])

        # target critics用 next-state，其他用 current-state
        seq_idx = 1 if role in ["target_critic_1", "target_critic_2"] else 0
        hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])[:,:,seq_idx,:].contiguous()
        cell_states   = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])[:,:,seq_idx,:].contiguous()

        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = terminated.view(-1, self.sequence_length)
            idx = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]
            for i in range(len(idx) - 1):
                i0, i1 = idx[i], idx[i + 1]
                rnn_out, (hidden_states, cell_states) = self.lstm(rnn_input[:, i0:i1, :], (hidden_states, cell_states))
                hidden_states[:, (terminated[:, i1-1]), :] = 0
                cell_states[:, (terminated[:, i1-1]), :] = 0
                rnn_outputs.append(rnn_out)
            rnn_output = torch.cat(rnn_outputs, dim=1)
            rnn_states = (hidden_states, cell_states)
        else:
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        rnn_output = torch.flatten(rnn_output, 0, 1)
        x = F.relu(self.linear_layer_1(torch.cat([rnn_output, inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        q = self.linear_layer_3(x)
        return q, {"rnn": [rnn_states[0], rnn_states[1]]}


def build_agent(env, device, checkpoint_path):
    """构建 SAC_RNN agent 并加载 checkpoint（兼容只保存 policy 的 .pt）"""
    models = {
        "policy":          Actor(env.observation_space, env.action_space, device, clip_actions=True, num_envs=env.num_envs),
        "critic_1":        Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs),
        "critic_2":        Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs),
        "target_critic_1": Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs),
        "target_critic_2": Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs),
    }

    cfg = SAC_DEFAULT_CONFIG.copy()
    # 评估口径：默认使用“确定性”（关闭随机探索）
    cfg["stochastic_evaluation"] = False


    agent = SAC(models=models,
                memory=None,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    # 根据文件类型选择加载方式
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pt"):
        # 仅 policy 权重
        agent.models["policy"].load(checkpoint_path)
        # 同步目标 critic（尽管评估不用，但保持干净）
        agent.models["target_critic_1"].load_state_dict(agent.models["critic_1"].state_dict())
        agent.models["target_critic_2"].load_state_dict(agent.models["critic_2"].state_dict())
    else:
        # 完整 checkpoint 目录
        agent.load(checkpoint_path)

    agent.init()
    agent.set_running_mode("eval")  # 评估模式
    for m in models.values():
        m.eval()
    torch.set_grad_enabled(False)
    return agent


def evaluate_trainer_style(env, agent, total_timesteps=10_000,
                           headless=True,
                           environment_info_key=None,   # 比如 "skrl_env_info"；不需要就传 None
                           seed=None):
    """
    完全仿照 trainer.eval() 的评估循环（单/多环境均可）。
    - 动作：按 cfg["stochastic_evaluation"] 选择随机 or 均值（确定性）
    - TB 写入：通过 pre_interaction / record_transition / post_interaction
    - reset：单环境用 (terminated|truncated)；多环境直接接 next_obs
    返回：与你原先 log 字段兼容的 dict（单环境时）。
    """
    if seed is not None and hasattr(env, "seed"):
        try:
            env.reset(seed=seed)
        except TypeError:
            pass

    # reset
    observations, infos = env.reset()
    try:
        states = env.state()  # 有些 env/wrapper 不支持
    except Exception:
        states = observations  # 回退：用 obs 作为 states

    # 你原先汇总用的字段（仅单环境时填充；多环境就只返回回合总回报）
    positions, velocities, accelerations, jerks = [], [], [], []
    times, powers, rewards, actions_safe, actions_raw, energy = [], [], [], [], [], []
    episode_return = 0.0
    start = datetime.now()

    # timesteps 主循环
    for t in range(total_timesteps):
        # --- pre-interaction（让 agent 维护计步与写入节奏） ---
        agent.pre_interaction(timestep=t, timesteps=total_timesteps)

        with torch.no_grad():
            # --- act ---
            # print("time step:", t)
            actions, _, outputs = agent.act(observations, t, total_timesteps)

            # 按 trainer.eval 口径：确定性评估时取均值动作
            stochastic_eval = agent.cfg.get("stochastic_evaluation", False)
            if not stochastic_eval:
                actions = outputs.get("mean_actions", actions)

            # --- env.step ---
            next_observations, rs, terminated, truncated, infos = env.step(actions)
            try:
                next_states = env.state()
            except Exception:
                next_states = next_observations

            # --- 可选渲染 ---
            if not headless and hasattr(env, "render"):
                try:
                    env.render()
                except Exception:
                    pass

            # --- 记录 transition（这一步会让 TB 获得 Reward / Total reward 等曲线） ---
            agent.record_transition(
                states=states,
                actions=actions,
                rewards=rs,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=t,
                timesteps=total_timesteps
            )

            # --- 可选：从 infos 里收集环境指标（与 trainer.eval 一致的口径） ---
            if environment_info_key is not None and environment_info_key in infos:
                for k, v in infos[environment_info_key].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        agent.track_data(k if "/" in k else f"Info / {k}", v.item())

        # # --- post-interaction（触发写入 TB 的节奏） ---
        # super(agent.__class__, agent).post_interaction(timestep=t, timesteps=total_timesteps)

        # ===== 单环境：同步你原先的日志收集 =====
        if env.num_envs == 1:
            # 注意：TrainSpeedControl2 在 infos 里直接放了标量（非batched张量），这里按你原代码取
            if isinstance(infos, dict):
                # 允许 keys 缺失：用 get 安全访问
                positions.append(infos.get('position'))
                velocities.append(infos.get('velocity'))
                accelerations.append(infos.get('acceleration'))
                jerks.append(infos.get('jerk'))
                times.append(infos.get('time'))
                powers.append(infos.get('power'))
                rewards.append(infos.get('reward' , float(rs[0]) if torch.is_tensor(rs) else float(rs)))
                actions_safe.append(infos.get('action_safe'))
                actions_raw.append(infos.get('action_raw'))
                energy.append(infos.get('energy'))

            # episode return
            if torch.is_tensor(rs):
                episode_return += float(rs[0])
            else:
                episode_return += float(rs)

        should_reset = False
        # --- reset 逻辑（严格对齐 trainer.eval） ---
        if env.num_envs > 1:
            observations = next_observations
        else:
            should_reset = bool(terminated.any() or truncated.any())
            if should_reset:
                observations, infos = env.reset()
            else:
                observations = next_observations

        # 单环境：到达一次自然/截断结束后可提前退出（可选）
        if env.num_envs == 1 and should_reset:
            break

    elapsed = (datetime.now() - start).total_seconds()

    # 输出与返回
    if env.num_envs == 1:
        print(f"[Eval] return={episode_return:.3f} | steps={t+1} | time={elapsed:.2f}s | final_energy={energy[-1] if energy else None}")
        log = {
            "time": times, "position": positions, "velocity": velocities,
            "acceleration": accelerations, "jerk": jerks, "power": powers,
            "energy": energy, "reward": rewards,
            "action_safe": actions_safe, "action_raw": actions_raw,
            "episode_return": episode_return, "elapsed_sec": elapsed
        }
    else:
        print(f"[Eval-vec] steps={total_timesteps} | time={elapsed:.2f}s")
        log = {"episode_return": None, "elapsed_sec": elapsed}
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
    env = wrap_env(TrainSpeedControl2())
    device = env.device
    print("Device:", device)
    print("Obs space:", env.observation_space, "Act space:", env.action_space)

    checkpoint = r"C:\Users\root\Documents\GitHub\RailRecurrentSAC1.0\runs\torch\LSTMSAC_TrainSpeedControl2_parallel\25-10-21_14-57-23-047771_SAC_RNN\checkpoints\policy_580000.pt"   # 也可给到完整 checkpoint 目录
    agent = build_agent(env, device, checkpoint)

    # 如果你想暂时用随机采样评估（不推荐），可改成：
    # agent.cfg["stochastic_evaluation"] = True

    log = evaluate_trainer_style(env, agent,
                                 total_timesteps=10_000,
                                 headless=True,
                                 environment_info_key=None,  # 若你的 env 在 infos 里放了汇总 dict，这里填 key
                                 seed=0)

    plot_curves(log)
    save_csv(log, out_name="skrl_eval_log.csv")

