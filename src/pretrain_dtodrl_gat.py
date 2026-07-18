from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from llm_offload_env.gnn_policy import DTODRLGATEncoder
from llm_offload_env.taskgen import generate_task


class GATAutoencoder(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = DTODRLGATEncoder(input_dim=6, hidden_dim=hidden_dim, heads=4)
        self.feature_decoder = nn.Linear(hidden_dim, 6)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features, adjacency, node_mask)
        reconstructed_features = self.feature_decoder(hidden)
        structure_logits = torch.bmm(hidden, hidden.transpose(1, 2)) / math.sqrt(hidden.shape[-1])
        return reconstructed_features, structure_logits


def _sample_graphs(
    count: int,
    seed: int,
    max_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    feature_rows = []
    adjacency_rows = []
    mask_rows = []
    for task_id in range(count):
        task = generate_task(
            rng,
            task_id=task_id,
            min_nodes=5,
            max_nodes=max_nodes,
            deadline_scale=1.25,
            graph_mode="standard",
        )
        progress_index = rng.randrange(len(task.nodes))
        features = torch.zeros(max_nodes, 6, dtype=torch.float32)
        adjacency = torch.zeros(max_nodes, max_nodes, dtype=torch.float32)
        node_mask = torch.zeros(max_nodes, 1, dtype=torch.float32)
        for node in task.nodes:
            features[node.idx] = torch.tensor(
                [
                    0.35 * min(node.input_tokens / 1200.0, 1.0)
                    + 0.65 * min(node.output_tokens / 1300.0, 1.0),
                    min(node.input_tokens / 1200.0, 1.0),
                    min(len(node.predecessors) / 4.0, 1.0),
                    min(len(node.successors) / 4.0, 1.0),
                    1.0 if node.idx < progress_index else 0.0,
                    1.0 if node.idx == progress_index else 0.0,
                ],
                dtype=torch.float32,
            )
            node_mask[node.idx] = 1.0
            for successor in node.successors:
                adjacency[node.idx, successor] = 1.0
        feature_rows.append(features)
        adjacency_rows.append(adjacency)
        mask_rows.append(node_mask)
    return (
        torch.stack(feature_rows),
        torch.stack(adjacency_rows),
        torch.stack(mask_rows),
    )


def _loss(
    model: GATAutoencoder,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    node_mask: torch.Tensor,
    corrupt: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_input = features
    if corrupt:
        corruption = (torch.rand_like(features) < 0.15).float() * node_mask
        model_input = features * (1.0 - corruption)
    reconstructed, structure_logits = model(model_input, adjacency, node_mask)
    feature_loss = (((reconstructed - features) ** 2) * node_mask).sum() / (
        node_mask.sum().clamp_min(1.0) * features.shape[-1]
    )

    undirected_target = ((adjacency + adjacency.transpose(1, 2)) > 0).float()
    pair_mask = node_mask * node_mask.transpose(1, 2)
    diagonal = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0)
    pair_mask = pair_mask * (1.0 - diagonal)
    positive = (undirected_target * pair_mask).sum().clamp_min(1.0)
    negative = ((1.0 - undirected_target) * pair_mask).sum().clamp_min(1.0)
    positive_weight = (negative / positive).clamp(max=20.0)
    structure_element = F.binary_cross_entropy_with_logits(
        structure_logits,
        undirected_target,
        reduction="none",
        pos_weight=positive_weight,
    )
    structure_loss = (structure_element * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    total = feature_loss + 0.15 * structure_loss
    return total, feature_loss, structure_loss


def pretrain(
    output: Path,
    seed: int = 1701,
    graphs: int = 1600,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_nodes: int = 12,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    features, adjacency, node_mask = _sample_graphs(graphs, seed, max_nodes)
    split = int(0.85 * graphs)
    model = GATAutoencoder(hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)
    history = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(split, generator=generator)
        train_total = 0.0
        train_batches = 0
        for start in range(0, split, batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            total, _, _ = _loss(
                model,
                features[indices],
                adjacency[indices],
                node_mask[indices],
                corrupt=True,
            )
            total.backward()
            optimizer.step()
            train_total += float(total.detach())
            train_batches += 1

        model.eval()
        with torch.no_grad():
            val_total, val_features, val_structure = _loss(
                model,
                features[split:],
                adjacency[split:],
                node_mask[split:],
                corrupt=False,
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_total / max(train_batches, 1),
                "validation_loss": float(val_total),
                "validation_feature_loss": float(val_features),
                "validation_structure_loss": float(val_structure),
            }
        )

    checkpoint = output / "dtodrl_gat_pretrained.pt"
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "seed": seed,
            "graphs": graphs,
            "epochs": epochs,
            "feature_definition": [
                "compute_proxy",
                "input_data_proxy",
                "normalized_indegree",
                "normalized_outdegree",
                "assigned_proxy",
                "availability",
            ],
        },
        checkpoint,
    )
    (output / "pretraining_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (output / "pretraining_protocol.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "graphs": graphs,
                "train_graphs": split,
                "validation_graphs": graphs - split,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "feature_loss_weight": 1.0,
                "structure_loss_weight": 0.15,
                "feature_corruption_probability": 0.15,
                "checkpoint": str(checkpoint),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--graphs", type=int, default=1600)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    checkpoint = pretrain(
        args.output,
        seed=args.seed,
        graphs=args.graphs,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"saved DTODRL GAT checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
