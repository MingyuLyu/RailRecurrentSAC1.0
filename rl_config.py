from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Env / rollout
    max_ep_len: int = 2000
    total_steps: int = 1_000_000
    start_steps: int = 80_000
    update_after: int = 80_000
    update_every: int = 50
    gradient_steps: int = 50


    # Replay / sequences
    replay_size: int = 1_000_000
    seq_len: int = 96
    burn_in: int = 16
    n_step: int = 3
    gamma: float = 0.997


    # Network sizes
    obs_dim: int = 4 # set from env.metadata at runtime
    act_dim: int = 1
    hidden: int = 256
    lstm_layers: int = 1


    # SAC
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    tau: float = 0.005
    target_entropy_scale: float = 0.8 # target_entropy ≈ scale * (−|A|)
    auto_alpha: bool = True


    # Training details
    batch_size: int = 64
    grad_clip: float = 1.0


    # IO
    log_dir: str = "runs/rsac"
    ckpt_every_steps: int = 100_000


    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"