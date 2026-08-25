from __future__ import annotations

import torch
from pathlib import Path
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch.nn.functional as F


class ConstantSchedule:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, _: float) -> float:
        return self.value


class DAGMessagePassingExtractor(BaseFeaturesExtractor):
    """Encode bidirectional messages on a padded workflow DAG."""

    def __init__(
        self,
        observation_space: spaces.Box,
        max_nodes: int = 12,
        node_feature_dim: int = 12,
        action_count: int = 5,
        global_feature_dim: int = 6,
        hidden_dim: int = 64,
        message_steps: int = 2,
        features_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.node_flat_dim = max_nodes * node_feature_dim
        self.adj_flat_dim = max_nodes * max_nodes
        self.tail_dim = action_count + global_feature_dim

        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(message_steps))
        self.pred_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(message_steps))
        self.succ_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(message_steps))
        self.norm_layers = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(message_steps))

        pooled_dim = hidden_dim * 3 + self.tail_dim
        self.readout = nn.Sequential(
            nn.Linear(pooled_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        expected_dim = self.node_flat_dim + self.adj_flat_dim + self.tail_dim
        actual_dim = int(observation_space.shape[0])
        if actual_dim != expected_dim:
            raise ValueError(
                f"DAGMessagePassingExtractor observation dim mismatch: expected {expected_dim}, got {actual_dim}"
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        nodes, adjacency, tail = self._split_observation(observations)
        node_mask = (nodes.abs().sum(dim=-1, keepdim=True) > 0).float()
        adjacency = adjacency * node_mask * node_mask.transpose(1, 2)

        h = F.relu(self.node_encoder(nodes)) * node_mask
        for step in range(len(self.self_layers)):
            pred_msg = torch.bmm(adjacency.transpose(1, 2), h)
            succ_msg = torch.bmm(adjacency, h)

            pred_deg = adjacency.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            succ_deg = adjacency.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
            pred_msg = pred_msg / pred_deg
            succ_msg = succ_msg / succ_deg

            h_new = (
                self.self_layers[step](h)
                + self.pred_layers[step](pred_msg)
                + self.succ_layers[step](succ_msg)
            )
            h = F.relu(self.norm_layers[step](h_new)) * node_mask

        current_weight = nodes[:, :, 0:1] * node_mask
        current_emb = (h * current_weight).sum(dim=1) / current_weight.sum(dim=1).clamp_min(1.0)
        mean_emb = (h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp_min(1.0)
        masked_h = h.masked_fill(node_mask == 0, -1e9)
        max_emb = masked_h.max(dim=1).values
        max_emb = torch.where(torch.isfinite(max_emb), max_emb, torch.zeros_like(max_emb))

        return self.readout(torch.cat([current_emb, mean_emb, max_emb, tail], dim=1))

    def _split_observation(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        node_end = self.node_flat_dim
        adj_end = node_end + self.adj_flat_dim
        nodes = observations[:, :node_end].reshape(batch, self.max_nodes, self.node_feature_dim)
        adjacency = observations[:, node_end:adj_end].reshape(batch, self.max_nodes, self.max_nodes)
        tail = observations[:, adj_end:]
        return nodes, adjacency, tail


def dtodrl_node_features(nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Map the shared observation to the six task features used by DTODRL.

    The original method uses computation, data size, in/out degree, assigned
    location, and availability. In this sequential environment, completed/not
    completed is the observable location-assignment proxy and ``is_current`` is
    the availability flag.
    """
    indegree = adjacency.sum(dim=1).clamp(max=4.0) / 4.0
    outdegree = adjacency.sum(dim=2).clamp(max=4.0) / 4.0
    compute_proxy = 0.35 * nodes[:, :, 3] + 0.65 * nodes[:, :, 4]
    data_proxy = nodes[:, :, 3]
    assigned_proxy = nodes[:, :, 1]
    availability = nodes[:, :, 0]
    return torch.stack(
        [
            compute_proxy,
            data_proxy,
            indegree,
            outdegree,
            assigned_proxy,
            availability,
        ],
        dim=-1,
    )


class DenseGATLayer(nn.Module):
    """Small dense graph-attention layer for the fixed 12-node DAG budget."""

    def __init__(self, input_dim: int, output_dim: int, heads: int = 4) -> None:
        super().__init__()
        if output_dim % heads != 0:
            raise ValueError("output_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_target = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, node_count, _ = features.shape
        projected = self.projection(features).reshape(
            batch,
            node_count,
            self.heads,
            self.head_dim,
        )
        source_score = (projected * self.attn_source).sum(dim=-1).unsqueeze(2)
        target_score = (projected * self.attn_target).sum(dim=-1).unsqueeze(1)
        scores = F.leaky_relu(source_score + target_score, negative_slope=0.2)

        valid_nodes = node_mask.squeeze(-1).bool()
        undirected = (adjacency + adjacency.transpose(1, 2)) > 0
        identity = torch.eye(node_count, device=features.device, dtype=torch.bool).unsqueeze(0)
        graph_mask = (undirected | identity) & valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        scores = scores.masked_fill(~graph_mask.unsqueeze(-1), -1e9)
        attention = torch.softmax(scores, dim=2)
        attention = attention * graph_mask.unsqueeze(-1)
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1e-9)
        aggregated = torch.einsum("bijh,bjhd->bihd", attention, projected)
        output = aggregated.reshape(batch, node_count, -1) + self.bias
        return output * node_mask


class DTODRLGATEncoder(nn.Module):
    """Two-layer GAT task encoder following the core DTODRL representation."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.gat1 = DenseGATLayer(input_dim, hidden_dim, heads=heads)
        self.gat2 = DenseGATLayer(hidden_dim, hidden_dim, heads=heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.elu(self.norm1(self.gat1(features, adjacency, node_mask))) * node_mask
        hidden = F.elu(self.norm2(self.gat2(hidden, adjacency, node_mask))) * node_mask
        return hidden


class DTODRLGATExtractor(BaseFeaturesExtractor):
    """DTODRL-adapted GAT task encoder plus location-profile MLP for PPO."""

    def __init__(
        self,
        observation_space: spaces.Box,
        max_nodes: int = 12,
        node_feature_dim: int = 12,
        action_count: int = 5,
        global_feature_dim: int = 6,
        hidden_dim: int = 32,
        features_dim: int = 128,
        checkpoint_path: str | None = None,
        freeze_gat: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.node_flat_dim = max_nodes * node_feature_dim
        self.adj_flat_dim = max_nodes * max_nodes
        self.tail_dim = action_count + global_feature_dim

        expected_dim = self.node_flat_dim + self.adj_flat_dim + self.tail_dim
        actual_dim = int(observation_space.shape[0])
        if actual_dim != expected_dim:
            raise ValueError(
                f"DTODRLGATExtractor observation dim mismatch: expected {expected_dim}, got {actual_dim}"
            )

        self.gat_encoder = DTODRLGATEncoder(input_dim=6, hidden_dim=hidden_dim, heads=4)
        if checkpoint_path:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.is_absolute():
                checkpoint = Path(__file__).resolve().parents[2] / checkpoint
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("encoder_state_dict", checkpoint)
            self.gat_encoder.load_state_dict(state_dict)
        if freeze_gat:
            for parameter in self.gat_encoder.parameters():
                parameter.requires_grad = False

        self.register_buffer(
            "location_capability",
            torch.tensor([0.46, 0.70, 0.84, 0.94, 1.00], dtype=torch.float32),
        )
        self.location_encoder = nn.Sequential(
            nn.Linear(action_count * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(max_nodes * hidden_dim + 32, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        node_end = self.node_flat_dim
        adj_end = node_end + self.adj_flat_dim
        nodes = observations[:, :node_end].reshape(batch, self.max_nodes, self.node_feature_dim)
        adjacency = observations[:, node_end:adj_end].reshape(
            batch,
            self.max_nodes,
            self.max_nodes,
        )
        tail = observations[:, adj_end:]
        node_mask = (nodes.abs().sum(dim=-1, keepdim=True) > 0).float()
        adjacency = adjacency * node_mask * node_mask.transpose(1, 2)
        task_features = dtodrl_node_features(nodes, adjacency) * node_mask
        task_embedding = self.gat_encoder(task_features, adjacency, node_mask)
        task_embedding = task_embedding.reshape(batch, -1)

        queues = tail[:, : self.location_capability.numel()]
        capabilities = self.location_capability.unsqueeze(0).expand(batch, -1)
        location_embedding = self.location_encoder(torch.cat([queues, capabilities], dim=1))
        return self.readout(torch.cat([task_embedding, location_embedding], dim=1))
