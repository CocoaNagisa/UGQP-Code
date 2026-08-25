from __future__ import annotations

from dataclasses import dataclass, field
import random

from .config import NODE_TYPES, NODE_TYPE_TO_ID


@dataclass
class Node:
    idx: int
    kind: str
    input_tokens: int
    output_tokens: int
    uncertainty: float
    privacy: int
    risk: float
    deadline_weight: float
    predecessors: list[int] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)
    criticality: float = 0.0
    downstream_risk: float = 0.0
    first_token_p1: float = 0.0
    first_token_p2: float = 0.0
    token_uncertainty: float = 0.0

    def kind_id(self) -> int:
        return NODE_TYPE_TO_ID[self.kind]


@dataclass
class Task:
    task_id: int
    scenario: str
    nodes: list[Node]
    deadline: float


def generate_task(
    rng: random.Random,
    task_id: int,
    min_nodes: int,
    max_nodes: int,
    deadline_scale: float = 1.0,
    graph_mode: str = "standard",
) -> Task:
    scenario = rng.choice(["emergency", "traffic", "municipal_service", "facility_inspection"])
    n = rng.randint(min_nodes, max_nodes)
    if graph_mode == "dependency_stress" and n != 12:
        raise ValueError("dependency_stress requires min_nodes=max_nodes=12")
    if graph_mode not in {"standard", "dependency_stress"}:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    if graph_mode == "dependency_stress":
        middle = [
            "retrieve", "tool_call", "plan", "generate", "verify",
            "retrieve", "tool_call", "plan", "generate",
        ]
        rng.shuffle(middle)
        kinds = ["intent", "extract", *middle, "summarize"]
    else:
        skeleton = ["intent", "extract", "retrieve", "plan", "verify", "summarize"]
        kinds = []
        for i in range(n):
            if i < len(skeleton):
                kinds.append(skeleton[i])
            else:
                kinds.append(rng.choice(["retrieve", "tool_call", "plan", "generate", "verify"]))
        middle = kinds[2:-1]
        rng.shuffle(middle)
        kinds = kinds[:2] + middle + kinds[-1:]

    nodes: list[Node] = []
    for idx, kind in enumerate(kinds):
        in_tok, out_tok, uncertainty = sample_node_shape(rng, kind, scenario)
        privacy = rng.choices([1, 2, 3], weights=[0.22, 0.46, 0.32])[0]
        risk = min(1.0, max(0.0, uncertainty * rng.uniform(0.75, 1.30)))
        deadline_weight = rng.uniform(0.65, 1.45)
        node = Node(idx, kind, in_tok, out_tok, uncertainty, privacy, risk, deadline_weight)
        proxy_phase = ((task_id * 131 + idx * 17) % 97) / 96.0
        alpha_token = min(
            0.98,
            max(0.02, 0.85 * uncertainty + 0.15 * (0.05 + 0.90 * proxy_phase)),
        )
        margin = 1.0 - alpha_token
        node.first_token_p1 = (1.0 + margin) / 2.0
        node.first_token_p2 = (1.0 - margin) / 2.0
        node.token_uncertainty = 1.0 - (node.first_token_p1 - node.first_token_p2)
        nodes.append(node)

    if graph_mode == "dependency_stress":
        _build_dependency_stress_graph(rng, nodes)
    else:
        for i in range(1, n):
            parent = rng.randint(0, i - 1)
            add_edge(nodes, parent, i)
            if i >= 3 and rng.random() < 0.42:
                extra_parent = rng.randint(0, i - 1)
                add_edge(nodes, extra_parent, i)
            if i >= 5 and rng.random() < 0.18:
                extra_parent = rng.randint(1, i - 1)
                add_edge(nodes, extra_parent, i)

    compute_graph_features(nodes)
    nominal = sum(0.05 + node.input_tokens / 2800.0 + node.output_tokens / 115.0 for node in nodes)
    deadline = nominal * rng.uniform(0.82, 1.22) * deadline_scale
    return Task(task_id, scenario, nodes, deadline)


def _build_dependency_stress_graph(rng: random.Random, nodes: list[Node]) -> None:
    """Construct a variable branch/merge DAG with a fixed 12-node budget."""
    layers = [[0, 1], [2, 3, 4], [5, 6, 7, 8], [9, 10], [11]]

    for dst in layers[1]:
        parents = rng.sample(layers[0], k=2 if rng.random() < 0.35 else 1)
        for src in parents:
            add_edge(nodes, src, dst)

    for dst in layers[2]:
        parent_count = 2 if rng.random() < 0.65 else 1
        for src in rng.sample(layers[1], k=parent_count):
            add_edge(nodes, src, dst)

    for dst in layers[3]:
        parent_count = 3 if rng.random() < 0.40 else 2
        for src in rng.sample(layers[2], k=parent_count):
            add_edge(nodes, src, dst)

    add_edge(nodes, 9, 11)
    add_edge(nodes, 10, 11)

    for layer_idx, layer in enumerate(layers[:-1]):
        next_layer = layers[layer_idx + 1]
        for src in layer:
            if not nodes[src].successors:
                add_edge(nodes, src, rng.choice(next_layer))


def sample_node_shape(rng: random.Random, kind: str, scenario: str) -> tuple[int, int, float]:
    scenario_risk = {"emergency": 0.12, "traffic": 0.06, "municipal_service": -0.04, "facility_inspection": 0.04}[scenario]
    if kind in {"intent", "extract"}:
        in_tok = rng.randint(40, 230)
        out_tok = rng.randint(8, 90)
        u = rng.uniform(0.12, 0.58)
    elif kind in {"retrieve", "tool_call"}:
        in_tok = rng.randint(80, 360)
        out_tok = rng.randint(70, 420)
        u = rng.uniform(0.20, 0.72)
    elif kind in {"plan", "generate"}:
        in_tok = rng.randint(260, 980)
        out_tok = rng.randint(280, 1250)
        u = rng.uniform(0.42, 0.95)
    else:
        in_tok = rng.randint(230, 850)
        out_tok = rng.randint(35, 320)
        u = rng.uniform(0.26, 0.78)
    return in_tok, out_tok, min(0.98, max(0.02, u + scenario_risk))


def add_edge(nodes: list[Node], src: int, dst: int) -> None:
    if src == dst:
        return
    if dst not in nodes[src].successors:
        nodes[src].successors.append(dst)
    if src not in nodes[dst].predecessors:
        nodes[dst].predecessors.append(src)


def compute_graph_features(nodes: list[Node]) -> None:
    longest = [0.0 for _ in nodes]
    downstream = [0.0 for _ in nodes]
    for node in reversed(nodes):
        own = 1.0 + 0.0015 * node.input_tokens + 0.0035 * node.output_tokens
        if node.successors:
            longest[node.idx] = own + max(longest[s] for s in node.successors)
            downstream[node.idx] = node.risk + sum(downstream[s] for s in node.successors) / len(node.successors)
        else:
            longest[node.idx] = own
            downstream[node.idx] = node.risk
    max_path = max(longest) if longest else 1.0
    max_down = max(downstream) if downstream else 1.0
    for node in nodes:
        node.criticality = longest[node.idx] / max_path
        node.downstream_risk = downstream[node.idx] / max_down
