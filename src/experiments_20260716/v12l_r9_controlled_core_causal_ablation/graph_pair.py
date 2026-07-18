from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Literal, Sequence


NODE_COUNT = 12
MAX_CORRELATION = 0.50
MIN_MEAN_ABSOLUTE_DIFFERENCE = 0.15
MAX_TOP3_OVERLAP_RATIO = 1.0 / 3.0

                                                                             
                                                                       
                                                                             
                                                                      
                                                              
ROLE_SWAP_A_EDGES: tuple[tuple[int, int], ...] = (
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (1, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 8),
    (4, 9),
    (4, 10),
    (5, 8),
    (5, 9),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 11),
    (10, 11),
)

                                                                           
                                                                              
                                         
ROLE_PERMUTATION: tuple[int, ...] = (
    2,
    3,
    0,
    1,
    6,
    7,
    4,
    5,
    10,
    8,
    9,
    11,
)
ROLE_SWAP_B_EDGES: tuple[tuple[int, int], ...] = tuple(
    sorted((ROLE_PERMUTATION[src], ROLE_PERMUTATION[dst]) for src, dst in ROLE_SWAP_A_EDGES)
)


class GraphPairAuditError(ValueError):
    """Raised when a proposed role-swap pair fails a frozen hard audit."""


@dataclass(frozen=True)
class GraphMember:
    edges: tuple[tuple[int, int], ...]
    exposure: tuple[float, ...]
    raw_exposure: tuple[float, ...]
    edge_count: int
    depth: int
    degree_multiset: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class GraphPairAudit:
    correlation: float
    mean_absolute_difference: float
    top3_a: tuple[int, ...]
    top3_b: tuple[int, ...]
    top3_overlap_count: int
    top3_overlap_ratio: float
    edge_count_equal: bool
    depth_equal: bool
    degree_multiset_equal: bool
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RoleSwapGraphPair:
    a: GraphMember
    b: GraphMember
    role_permutation: tuple[int, ...]
    audit: GraphPairAudit

    def member(self, name: Literal["A", "B"] | str) -> GraphMember:
        normalized = str(name).upper()
        if normalized == "A":
            return self.a
        if normalized == "B":
            return self.b
        raise ValueError(f"unknown graph-pair member: {name!r}")


def _nodes_by_id(nodes: Sequence[Any]) -> dict[int, Any]:
    if len(nodes) != NODE_COUNT:
        raise ValueError(f"role-swap DAG requires exactly {NODE_COUNT} nodes")
    indexed: dict[int, Any] = {}
    for position, node in enumerate(nodes):
        node_id = int(getattr(node, "idx", position))
        if node_id in indexed:
            raise ValueError(f"duplicate node id: {node_id}")
        indexed[node_id] = node
    expected = set(range(NODE_COUNT))
    if set(indexed) != expected:
        raise ValueError(
            "role-swap DAG requires node ids 0..11; "
            f"received {sorted(indexed)}"
        )
    return indexed


def _bounded(value: Any, low: float, high: float) -> float:
    numeric = float(value)
    return min(high, max(low, numeric))


def _descendant_weight(node: Any) -> float:
    """Return a bounded positive weight for a descendant's propagated impact."""

    risk = _bounded(getattr(node, "risk", 0.5), 0.0, 1.0)
    deadline_weight = _bounded(getattr(node, "deadline_weight", 1.0), 0.0, 2.0)
    input_tokens = max(0.0, float(getattr(node, "input_tokens", 0.0)))
    output_tokens = max(0.0, float(getattr(node, "output_tokens", 0.0)))
    token_weight = min((input_tokens + output_tokens) / 1800.0, 1.0)
    return 1.0 + 0.80 * risk + 0.25 * deadline_weight + 0.25 * token_weight


def _validate_edges(edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    normalized = tuple(sorted((int(src), int(dst)) for src, dst in edges))
    if len(normalized) != len(set(normalized)):
        raise ValueError("role-swap graph contains duplicate edges")
    for src, dst in normalized:
        if not (0 <= src < NODE_COUNT and 0 <= dst < NODE_COUNT):
            raise ValueError(f"edge outside 12-node budget: {(src, dst)}")
        if src >= dst:
            raise ValueError(
                "role-swap graph must retain the existing range(12) topological "
                f"order; invalid edge {(src, dst)}"
            )
    return normalized


def compute_weighted_exposure(
    nodes: Sequence[Any],
    edges: Iterable[tuple[int, int]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute weighted descendant exposure with directed path multiplicity.

    For source ``v``, every distinct path from ``v`` to descendant ``u`` adds
    one bounded weight for ``u``.  A small self term avoids an artificial zero
    for the sink.  Min-max normalization is performed within each graph.
    Unlike successor averaging, this score remains sensitive to branch count,
    merge count and the number of downstream paths.
    """

    indexed = _nodes_by_id(nodes)
    normalized_edges = _validate_edges(edges)
    successors: list[list[int]] = [[] for _ in range(NODE_COUNT)]
    for src, dst in normalized_edges:
        successors[src].append(dst)
    weights = [_descendant_weight(indexed[idx]) for idx in range(NODE_COUNT)]

    raw: list[float] = []
    for source in range(NODE_COUNT):
        path_count = [0.0] * NODE_COUNT
        path_count[source] = 1.0
        for node_id in range(source, NODE_COUNT):
            if path_count[node_id] == 0.0:
                continue
            for successor in successors[node_id]:
                path_count[successor] += path_count[node_id]
        descendant_total = sum(
            path_count[target] * weights[target]
            for target in range(NODE_COUNT)
            if target != source
        )
        raw.append(0.15 * weights[source] + descendant_total)

    low = min(raw)
    high = max(raw)
    if not high > low:
        raise GraphPairAuditError("weighted exposure is constant across all nodes")
    normalized = tuple((value - low) / (high - low) for value in raw)
    return normalized, tuple(raw)


def _graph_depth(edges: tuple[tuple[int, int], ...]) -> int:
    depth = [1] * NODE_COUNT
    for src, dst in edges:
        depth[dst] = max(depth[dst], depth[src] + 1)
    return max(depth)


def _degree_multiset(edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    indegree = [0] * NODE_COUNT
    outdegree = [0] * NODE_COUNT
    for src, dst in edges:
        outdegree[src] += 1
        indegree[dst] += 1
    return tuple(sorted(zip(indegree, outdegree)))


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 1e-12:
        raise GraphPairAuditError("cannot audit correlation for constant exposure")
    return numerator / denominator


def _top_nodes(values: Sequence[float], count: int = 3) -> tuple[int, ...]:
    return tuple(
        sorted(range(len(values)), key=lambda idx: (values[idx], -idx), reverse=True)[:count]
    )


def audit_graph_pair(a: GraphMember, b: GraphMember) -> GraphPairAudit:
    correlation = _pearson(a.exposure, b.exposure)
    mean_absolute_difference = sum(
        abs(left - right) for left, right in zip(a.exposure, b.exposure)
    ) / NODE_COUNT
    top3_a = _top_nodes(a.exposure)
    top3_b = _top_nodes(b.exposure)
    overlap_count = len(set(top3_a).intersection(top3_b))
    overlap_ratio = overlap_count / 3.0
    edge_count_equal = a.edge_count == b.edge_count
    depth_equal = a.depth == b.depth
    degree_multiset_equal = a.degree_multiset == b.degree_multiset

    failures: list[str] = []
    if correlation > MAX_CORRELATION:
        failures.append(
            f"exposure correlation {correlation:.6f} exceeds {MAX_CORRELATION:.2f}"
        )
    if mean_absolute_difference < MIN_MEAN_ABSOLUTE_DIFFERENCE:
        failures.append(
            "exposure mean absolute difference "
            f"{mean_absolute_difference:.6f} is below "
            f"{MIN_MEAN_ABSOLUTE_DIFFERENCE:.2f}"
        )
    if overlap_ratio > MAX_TOP3_OVERLAP_RATIO:
        failures.append(
            f"top-3 overlap ratio {overlap_ratio:.6f} exceeds "
            f"{MAX_TOP3_OVERLAP_RATIO:.6f}"
        )
    if not edge_count_equal:
        failures.append("A/B edge counts differ")
    if not depth_equal:
        failures.append("A/B graph depths differ")
    if not degree_multiset_equal:
        failures.append("A/B joint degree multisets differ")

    return GraphPairAudit(
        correlation=correlation,
        mean_absolute_difference=mean_absolute_difference,
        top3_a=top3_a,
        top3_b=top3_b,
        top3_overlap_count=overlap_count,
        top3_overlap_ratio=overlap_ratio,
        edge_count_equal=edge_count_equal,
        depth_equal=depth_equal,
        degree_multiset_equal=degree_multiset_equal,
        passed=not failures,
        failures=tuple(failures),
    )


def _member(nodes: Sequence[Any], edges: Iterable[tuple[int, int]]) -> GraphMember:
    normalized_edges = _validate_edges(edges)
    exposure, raw_exposure = compute_weighted_exposure(nodes, normalized_edges)
    return GraphMember(
        edges=normalized_edges,
        exposure=exposure,
        raw_exposure=raw_exposure,
        edge_count=len(normalized_edges),
        depth=_graph_depth(normalized_edges),
        degree_multiset=_degree_multiset(normalized_edges),
    )


def build_role_swap_pair(
    nodes: Sequence[Any],
    *,
    strict: bool = True,
) -> RoleSwapGraphPair:
    """Build and hard-audit the frozen 12-node A/B role-swap pair."""

    _nodes_by_id(nodes)
    a = _member(nodes, ROLE_SWAP_A_EDGES)
    b = _member(nodes, ROLE_SWAP_B_EDGES)
    audit = audit_graph_pair(a, b)
    if strict and not audit.passed:
        raise GraphPairAuditError("; ".join(audit.failures))
    return RoleSwapGraphPair(
        a=a,
        b=b,
        role_permutation=ROLE_PERMUTATION,
        audit=audit,
    )


def apply_role_swap_member(
    nodes: Sequence[Any],
    pair: RoleSwapGraphPair,
    member: Literal["A", "B"] | str,
    *,
    assign_legacy_graph_fields: bool = True,
) -> tuple[float, ...]:
    """Install one graph member on existing mutable Node objects.

    When ``assign_legacy_graph_fields`` is true, both ``criticality`` and
    ``downstream_risk`` receive the new exposure.  This intentionally makes
    the legacy ``0.5 * (criticality + downstream_risk)`` expression equal the
    audited V12H exposure while a dedicated environment field is introduced.
    """

    indexed = _nodes_by_id(nodes)
    selected = pair.member(member)
    for node in indexed.values():
        node.predecessors = []
        node.successors = []
    for src, dst in selected.edges:
        indexed[src].successors.append(dst)
        indexed[dst].predecessors.append(src)
    if assign_legacy_graph_fields:
        for node_id, exposure in enumerate(selected.exposure):
            indexed[node_id].criticality = float(exposure)
            indexed[node_id].downstream_risk = float(exposure)
    return selected.exposure


def self_test() -> None:
    @dataclass
    class _Node:
        idx: int
        input_tokens: int
        output_tokens: int
        risk: float
        deadline_weight: float
        predecessors: list[int]
        successors: list[int]
        criticality: float = 0.0
        downstream_risk: float = 0.0

    nodes = [
        _Node(
            idx=idx,
            input_tokens=100 + 31 * idx,
            output_tokens=80 + 47 * idx,
            risk=0.10 + 0.07 * (idx % 9),
            deadline_weight=0.70 + 0.05 * idx,
            predecessors=[],
            successors=[],
        )
        for idx in range(NODE_COUNT)
    ]
    pair = build_role_swap_pair(nodes)
    assert pair.audit.passed
    assert pair.a.edge_count == pair.b.edge_count == 18
    assert pair.a.depth == pair.b.depth == 4
    assert pair.a.degree_multiset == pair.b.degree_multiset
    exposure = apply_role_swap_member(nodes, pair, "B")
    assert all(
        abs(nodes[idx].criticality - exposure[idx]) < 1e-12
        and abs(nodes[idx].downstream_risk - exposure[idx]) < 1e-12
        for idx in range(NODE_COUNT)
    )
    assert sum(len(node.successors) for node in nodes) == 18


__all__ = [
    "GraphMember",
    "GraphPairAudit",
    "GraphPairAuditError",
    "MAX_CORRELATION",
    "MAX_TOP3_OVERLAP_RATIO",
    "MIN_MEAN_ABSOLUTE_DIFFERENCE",
    "NODE_COUNT",
    "ROLE_PERMUTATION",
    "ROLE_SWAP_A_EDGES",
    "ROLE_SWAP_B_EDGES",
    "RoleSwapGraphPair",
    "apply_role_swap_member",
    "audit_graph_pair",
    "build_role_swap_pair",
    "compute_weighted_exposure",
    "self_test",
]
