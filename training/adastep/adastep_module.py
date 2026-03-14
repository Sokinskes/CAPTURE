"""
AdaStep HorizonPredictor (migrated from adastep_extension)
- Copied/adapted for act-plus-plus integration (paper-core implementation).
- Keep implementation close to the paper: 3-layer MLP predictor + StateClusterAnalyzer + AdaptiveHorizonLoss.
"""

# --- Implementation copied/adapted from adastep_extension/core/adastep_module.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional
import pickle
import os


class HorizonPredictor(nn.Module):
    """Lightweight 3-layer MLP horizon predictor (paper implementation).

    API:
      - forward(latent) -> normalized value in [0,1]
      - predict_horizon(latent, k_min, k_max) -> integer horizon
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(latent))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def predict_horizon(self, latent: torch.Tensor,
                        k_min: int = 5, k_max: int = 50) -> torch.Tensor:
        normalized = self.forward(latent)
        k = normalized * (k_max - k_min) + k_min
        return k.squeeze(-1).round().long()


class StateClusterAnalyzer:
    """State-level clustering and Pareto label generator (paper).

    - fit_clusters(states)
    - pareto_analysis(states, action_seqs, ...) -> cluster_horizons
    - get_labels(states, k_min, k_max) -> normalized labels
    """

    def __init__(self, num_clusters: int = 10, error_threshold: float = 0.5):
        self.num_clusters = num_clusters
        self.error_threshold = error_threshold
        self.kmeans = None
        self.cluster_horizons = None

    def fit_clusters(self, states: np.ndarray):
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        self.kmeans.fit(states)

    def calculate_linearity_deviation(self, actions: np.ndarray, k: int) -> float:
        if k >= len(actions) or k < 2:
            return 0.0
        action_chunk = actions[:k]
        start = action_chunk[0]
        end = action_chunk[-1]
        linear_traj = np.linspace(start, end, k)
        deviations = np.linalg.norm(action_chunk - linear_traj, axis=1)
        avg_deviation = np.mean(deviations)
        action_magnitude = np.linalg.norm(action_chunk, axis=1).mean() + 1e-6
        return avg_deviation / action_magnitude

    def pareto_analysis(self, states, action_sequences,
                        k_min: int = 5, k_max: int = 50, sample_size: int = 200,
                        lambda_param: float = 1.0) -> Dict[int, int]:
        if self.kmeans is None:
            raise ValueError("please run fit_clusters() first")
        if isinstance(states, list):
            # states is list of episode states
            all_states = np.concatenate(states, axis=0)
            labels = self.kmeans.predict(all_states)
            cluster_complexities = {}
            timestep = 0
            for cid in range(self.num_clusters):
                idx = np.where(labels == cid)[0]
                if len(idx) == 0:
                    cluster_complexities[cid] = 0.0
                    continue
                if len(idx) > sample_size:
                    idx = np.random.choice(idx, sample_size, replace=False)
                complexities = []
                for i in idx:
                    # find which episode and timestep
                    ep = 0
                    while i >= len(states[ep]):
                        i -= len(states[ep])
                        ep += 1
                    action_seq = action_sequences[ep]
                    k_probe = min(k_min + (k_max - k_min) // 2, len(action_seq))
                    if k_probe >= 2 and i + k_probe <= len(action_seq):
                        complexities.append(self.calculate_linearity_deviation(action_seq[i:i+k_probe], k_probe))
                cluster_complexities[cid] = np.mean(complexities) if complexities else 0.0
        else:
            # old way, assume states is concatenated, action_sequences is list
            labels = self.kmeans.predict(states)
            cluster_complexities = {}
            for cid in range(self.num_clusters):
                idx = np.where(labels == cid)[0]
                if len(idx) == 0:
                    cluster_complexities[cid] = 0.0
                    continue
                if len(idx) > sample_size:
                    idx = np.random.choice(idx, sample_size, replace=False)
                complexities = []
                for i in idx:
                    action_seq = action_sequences[i] if isinstance(action_sequences, list) else action_sequences[i]
                    k_probe = min(k_min + (k_max - k_min) // 2, len(action_seq))
                    if k_probe >= 2:
                        complexities.append(self.calculate_linearity_deviation(action_seq, k_probe))
                cluster_complexities[cid] = np.mean(complexities) if complexities else 0.0
        all_c = [c for c in cluster_complexities.values() if c > 0]
        if len(all_c) == 0:
            dynamic_threshold = 0.1
        else:
            percentile = int(self.error_threshold * 100)
            base = np.percentile(all_c, percentile)
            dynamic_threshold = base * lambda_param
        cluster_horizons = {}
        k_candidates = np.arange(k_min, k_max + 1, 5)
        for cid in range(self.num_clusters):
            comp = cluster_complexities[cid]
            if comp == 0:
                cluster_horizons[cid] = k_min
                continue
            if comp > dynamic_threshold:
                complexity_ratio = min(comp / dynamic_threshold - 1, 1.0)
                assigned_k = k_min + int((1 - complexity_ratio) * (k_max - k_min) / 2)
            else:
                complexity_ratio = comp / dynamic_threshold
                assigned_k = k_min + int((1 + complexity_ratio) * (k_max - k_min) / 2)
            assigned_k = min(k_candidates, key=lambda x: abs(x - assigned_k))
            cluster_horizons[cid] = int(assigned_k)
        self.cluster_horizons = cluster_horizons
        return cluster_horizons

    def get_labels(self, states: np.ndarray, k_min: int = 5, k_max: int = 50) -> np.ndarray:
        if self.kmeans is None or self.cluster_horizons is None:
            raise ValueError("fit_clusters + pareto_analysis required")
        labels = self.kmeans.predict(states)
        horizons = np.array([self.cluster_horizons[l] for l in labels])
        normalized = (horizons - k_min) / (k_max - k_min)
        return normalized.reshape(-1, 1).astype(np.float32)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'kmeans': self.kmeans, 'cluster_horizons': self.cluster_horizons,
                         'num_clusters': self.num_clusters, 'error_threshold': self.error_threshold}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.cluster_horizons = data['cluster_horizons']
            self.num_clusters = data['num_clusters']
            self.error_threshold = data['error_threshold']


class AdaptiveHorizonLoss(nn.Module):
    def __init__(self, kl_weight: float = 10.0, horizon_weight: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.horizon_weight = horizon_weight

    def forward(self, action_pred: torch.Tensor, action_gt: torch.Tensor, is_pad: torch.Tensor,
                kl_loss: torch.Tensor, horizon_pred: torch.Tensor, horizon_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_l1 = F.l1_loss(action_pred, action_gt, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        horizon_loss = F.mse_loss(horizon_pred, horizon_gt)
        total_loss = l1 + self.kl_weight * kl_loss + self.horizon_weight * horizon_loss
        return {'l1': l1, 'kl': kl_loss, 'horizon': horizon_loss, 'loss': total_loss}

