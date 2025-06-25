# === flappy_risk_attention.py ===
# PPO Agent with Risk-Guided Attention for flappy-bird-gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env

# -----------------------------
# Zone Aggregator
# -----------------------------
def aggregate_lidar_to_zones(lidar, num_zones=12):
    lidar = np.array(lidar)
    zone_size = len(lidar) // num_zones
    zones = [np.mean(lidar[i*zone_size:(i+1)*zone_size]) for i in range(num_zones)]
    return np.array(zones, dtype=np.float32)

# -----------------------------
# Risk Estimator
# -----------------------------
class RiskEstimator(nn.Module):
    def __init__(self, zone_dim):
        super().__init__()
        self.risk_fc = nn.Sequential(
            nn.Linear(zone_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, zones):
        # zones shape: [batch_size, num_zones]
        return self.risk_fc(zones)  # Remove unnecessary unsqueeze/squeeze operations

# -----------------------------
# Guided Attention Module
# -----------------------------
class GuidedAttention(nn.Module):
    def __init__(self, num_zones, embed_dim):
        super().__init__()
        self.zone_embed = nn.Linear(1, embed_dim)
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.attn_fc = nn.Linear(embed_dim, 1)

    def forward(self, zones, risks):
        B, Z = zones.shape
        zone_feats = self.zone_embed(zones.unsqueeze(-1))
        attn_logits = self.attn_fc(zone_feats).squeeze(-1) + risks * 5.0
        attn_weights = F.softmax(attn_logits, dim=-1)
        context = torch.sum(attn_weights.unsqueeze(-1) * zone_feats, dim=1)
        return context, attn_weights

# -----------------------------
# Custom Feature Extractor for PPO
# -----------------------------
class RiskAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.num_zones = 12  # Fixed aggregation to 12 zones
        self.risk_estimator = RiskEstimator(self.num_zones)
        self.attention = GuidedAttention(self.num_zones, features_dim)

    def forward(self, observations):
        # Convert observations to zones
        if isinstance(observations, torch.Tensor):
            obs_np = observations.cpu().numpy()
        else:
            obs_np = observations

        batch_zones = np.array([aggregate_lidar_to_zones(obs, self.num_zones) for obs in obs_np])
        zones = torch.tensor(batch_zones, dtype=torch.float32, device=observations.device)

        risks = self.risk_estimator(zones)
        context, _ = self.attention(zones, risks)
        return context

# -----------------------------
# Setup Environment and PPO Training
# -----------------------------
if __name__ == "__main__":
    env = make_vec_env("FlappyBird-v0", n_envs=8)

    policy_kwargs = dict(
        features_extractor_class=RiskAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=64)
    )

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_flappy_risk_attention")
