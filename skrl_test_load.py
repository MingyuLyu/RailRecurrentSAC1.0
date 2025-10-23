import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv  # 两种都导入，二选一
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC_RNN as SAC
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from rail_env import TrainSpeedControl2
from bc_env import BehaviorCloning

# ---------------------------
# Seed
# ---------------------------
set_seed(42)

# ---------------------------
# RNN Actor / Critic (沿用你的实现，略)
# ---------------------------
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


# ---------------------------
# 并行环境（两选一）
# ---------------------------
NUM_ENVS = 16

def make_env():
    def _init():
        return TrainSpeedControl2()
        # return BehaviorCloning()
    return _init

# 选项 A：更稳（建议先用）
env = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

# 选项 B：更快（确认环境可多进程后再用）
# env = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

# skrl 包装到 torch
env = wrap_env(env)
device = env.device

# ---------------------------
# Replay Memory
# ---------------------------
memory = RandomMemory(
    memory_size=1_000_000,
    num_envs=env.num_envs,
    device=device,
    replacement=False
)

# ---------------------------
# 模型
# ---------------------------
models = {
    "policy":          Actor(env.observation_space, env.action_space, device, clip_actions=True,  num_envs=env.num_envs),
    "critic_1":        Critic(env.observation_space, env.action_space, device,                  num_envs=env.num_envs),
    "critic_2":        Critic(env.observation_space, env.action_space, device,                  num_envs=env.num_envs),
    "target_critic_1": Critic(env.observation_space, env.action_space, device,                  num_envs=env.num_envs),
    "target_critic_2": Critic(env.observation_space, env.action_space, device,                  num_envs=env.num_envs),
}

for m in models.values():
    m.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# ---------------------------
# SAC 配置（RSAC for 200-step）
# ---------------------------
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["discount_factor"]   = 0.99
cfg["tau"]               = 0.01
cfg["batch_size"]        = 256
cfg["random_timesteps"]  = 0   # ← 必须：探索热身
cfg["learning_starts"]   = 5000
cfg["grad_norm_clip"]    = 1.0

# 熵温度：**推荐自动**
# cfg["learn_entropy"]     = True
cfg["target_entropy"]    = -1.0   # 1D action

# 如果你更想用固定 alpha（手动二选一）:
cfg["learn_entropy"]   = False
cfg["entropy_coef"]    = 0.03

# log & checkpoint
cfg["experiment"]["write_interval"]      = 1000
cfg["experiment"]["checkpoint_interval"] = 10000
cfg["experiment"]["directory"]           = "runs/torch/LSTMSAC_TrainSpeedControl2_parallel/"
cfg["experiment"]["store_separately"] = True

agent = SAC(models=models, memory=memory, cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space, device=device)

# ---------------------------
# 训练
# ---------------------------
checkpoint = r"C:\Users\root\Documents\GitHub\RailRecurrentSAC1.0\runs\torch\LSTMSAC_TrainSpeedControl2_parallel\25-10-21_12-25-41-169023_SAC_RNN\checkpoints\policy_10000.pt"
agent.load(checkpoint)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
