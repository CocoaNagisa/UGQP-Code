from __future__ import annotations

import math

from gymnasium import spaces
import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DirectionalResidualBlock(nn.Module):
    """One directed, degree-aware residual message-passing block.

    Both scaled sums and means are retained.  The sums preserve branch/merge
    multiplicity, while the means keep feature magnitude stable when degrees
    differ.  Separate predecessor and successor channels preserve direction.
    """

    def __init__(self, hidden_dim: int, max_nodes: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sum_scale = math.sqrt(max(max_nodes, 1))
        aggregate_dim = hidden_dim * 5 + 4
        self.update = nn.Sequential(
            nn.Linear(aggregate_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        predecessor_sum = torch.bmm(adjacency.transpose(1, 2), hidden)
        successor_sum = torch.bmm(adjacency, hidden)
        indegree = adjacency.sum(dim=1).unsqueeze(-1)
        outdegree = adjacency.sum(dim=2).unsqueeze(-1)
        predecessor_mean = predecessor_sum / indegree.clamp_min(1.0)
        successor_mean = successor_sum / outdegree.clamp_min(1.0)

        degree_denominator = float(max(adjacency.shape[1] - 1, 1))
        log_denominator = math.log1p(degree_denominator)
        degree_features = torch.cat(
            (
                indegree / degree_denominator,
                outdegree / degree_denominator,
                torch.log1p(indegree) / log_denominator,
                torch.log1p(outdegree) / log_denominator,
            ),
            dim=-1,
        )
        aggregate = torch.cat(
            (
                hidden,
                predecessor_sum / self.sum_scale,
                successor_sum / self.sum_scale,
                predecessor_mean,
                successor_mean,
                degree_features,
            ),
            dim=-1,
        )
        delta = self.update(aggregate)
        gate = torch.sigmoid(self.gate(torch.cat((hidden, delta), dim=-1)))
        updated = F.silu(self.norm(hidden + gate * delta))
        return updated * node_mask


class DirectionalResidualJKExtractor(BaseFeaturesExtractor):
    """SB3-compatible V12H DAG extractor with directional residual JK layers.

    The flattened observation layout is identical to the existing extractor::

        [max_nodes * node_feature_dim, max_nodes**2 adjacency, tail]

    where ``tail`` has ``action_count + global_feature_dim`` values.  Node row
    0 is not special: feature column 0 remains the ``is_current`` flag, so a
    synchronized permutation of node rows and both adjacency axes leaves the
    output invariant.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        max_nodes: int = 12,
        node_feature_dim: int = 12,
        action_count: int = 5,
        global_feature_dim: int = 6,
        hidden_dim: int = 96,
        message_steps: int = 4,
        features_dim: int = 160,
    ) -> None:
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if node_feature_dim <= 0:
            raise ValueError("node_feature_dim must be positive")
        if action_count < 0 or global_feature_dim < 0:
            raise ValueError("tail dimensions must be non-negative")
        if hidden_dim <= 0 or message_steps < 0 or features_dim <= 0:
            raise ValueError("hidden, message-step and output dimensions are invalid")
        super().__init__(observation_space, features_dim)
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.node_flat_dim = max_nodes * node_feature_dim
        self.adj_flat_dim = max_nodes * max_nodes
        self.tail_dim = action_count + global_feature_dim
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps

        expected_dim = self.node_flat_dim + self.adj_flat_dim + self.tail_dim
        actual_dim = int(observation_space.shape[0])
        if actual_dim != expected_dim:
            raise ValueError(
                "DirectionalResidualJKExtractor observation dim mismatch: "
                f"expected {expected_dim}, got {actual_dim}"
            )

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            DirectionalResidualBlock(hidden_dim, max_nodes)
            for _ in range(message_steps)
        )

                                                                               
                                                                          
                                
        graph_jk_dim = 3 * hidden_dim * (message_steps + 1)
        self.graph_readout = nn.Sequential(
            nn.Linear(graph_jk_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, features_dim),
            nn.SiLU(),
        )
        self.tail_encoder = nn.Sequential(
            nn.Linear(self.tail_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
                                                                               
                                                                    
        self.fusion = nn.Sequential(
            nn.Linear(features_dim + hidden_dim, features_dim),
            nn.SiLU(),
            nn.Linear(features_dim, features_dim),
            nn.SiLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        nodes, adjacency, tail = self._split_observation(observations)
        node_mask = (nodes.abs().sum(dim=-1, keepdim=True) > 0).to(nodes.dtype)
        adjacency = adjacency * node_mask * node_mask.transpose(1, 2)

        hidden = self.node_encoder(nodes) * node_mask
        layers = [hidden]
        for block in self.blocks:
            hidden = block(hidden, adjacency, node_mask)
            layers.append(hidden)

        current_weight = nodes[:, :, 0:1] * node_mask
        current_denominator = current_weight.sum(dim=1).clamp_min(1.0)
        node_denominator = node_mask.sum(dim=1).clamp_min(1.0)
        has_nodes = node_mask.sum(dim=1) > 0
        summaries: list[torch.Tensor] = []
        for layer in layers:
            current = (layer * current_weight).sum(dim=1) / current_denominator
            mean = (layer * node_mask).sum(dim=1) / node_denominator
            masked = layer.masked_fill(node_mask == 0, -torch.inf)
            maximum = masked.max(dim=1).values
            maximum = torch.where(has_nodes, maximum, torch.zeros_like(maximum))
            summaries.extend((current, mean, maximum))

        graph_embedding = self.graph_readout(torch.cat(summaries, dim=1))
        tail_embedding = self.tail_encoder(tail)
        return self.fusion(torch.cat((graph_embedding, tail_embedding), dim=1))

    def _split_observation(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        node_end = self.node_flat_dim
        adjacency_end = node_end + self.adj_flat_dim
        nodes = observations[:, :node_end].reshape(
            batch,
            self.max_nodes,
            self.node_feature_dim,
        )
        adjacency = observations[:, node_end:adjacency_end].reshape(
            batch,
            self.max_nodes,
            self.max_nodes,
        )
        tail = observations[:, adjacency_end:]
        return nodes, adjacency, tail


                                                                             
                                                                           
                                                        
V12HDirectionalResidualJKExtractor = DirectionalResidualJKExtractor


def self_test() -> None:
    torch.manual_seed(20260716)
    max_nodes = 12
    node_feature_dim = 12
    action_count = 8
    global_feature_dim = 22
    total_dim = (
        max_nodes * node_feature_dim
        + max_nodes * max_nodes
        + action_count
        + global_feature_dim
    )
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(total_dim,),
        dtype=float,
    )
    extractor = DirectionalResidualJKExtractor(
        observation_space,
        max_nodes=max_nodes,
        node_feature_dim=node_feature_dim,
        action_count=action_count,
        global_feature_dim=global_feature_dim,
        hidden_dim=32,
        message_steps=3,
        features_dim=48,
    )

    nodes = torch.rand(max_nodes, node_feature_dim)
    nodes[:, 0] = 0.0
    nodes[5, 0] = 1.0
    adjacency = torch.zeros(max_nodes, max_nodes)
    for src, dst in (
        (0, 4),
        (1, 4),
        (2, 5),
        (3, 5),
        (4, 8),
        (5, 8),
        (5, 9),
        (6, 9),
        (7, 10),
        (8, 11),
        (9, 11),
        (10, 11),
    ):
        adjacency[src, dst] = 1.0
    tail = torch.rand(action_count + global_feature_dim)
    observation = torch.cat((nodes.flatten(), adjacency.flatten(), tail)).unsqueeze(0)
    output = extractor(observation)
    assert output.shape == (1, 48)
    assert torch.isfinite(output).all()

    permutation = torch.tensor([6, 2, 9, 0, 11, 3, 5, 8, 1, 10, 4, 7])
    permuted = torch.cat(
        (
            nodes[permutation].flatten(),
            adjacency[permutation][:, permutation].flatten(),
            tail,
        )
    ).unsqueeze(0)
    permuted_output = extractor(permuted)
    assert torch.allclose(output, permuted_output, atol=2e-6, rtol=1e-5)

    reversed_observation = torch.cat(
        (nodes.flatten(), adjacency.transpose(0, 1).flatten(), tail)
    ).unsqueeze(0)
    reversed_output = extractor(reversed_observation)
    assert not torch.allclose(output, reversed_output, atol=1e-7, rtol=1e-7)

    output.square().sum().backward()
    trainable = [parameter.grad for parameter in extractor.parameters() if parameter.requires_grad]
    assert trainable and all(gradient is not None for gradient in trainable)
    assert all(torch.isfinite(gradient).all() for gradient in trainable)


__all__ = [
    "DirectionalResidualBlock",
    "DirectionalResidualJKExtractor",
    "V12HDirectionalResidualJKExtractor",
    "self_test",
]
