"""Small reliability-aware graph encoder for the A3-MVP pilot.

The encoder consumes only the bounded cached payload.  It performs directed
mean message passing in memory, pools node states to an APK representation,
and trains a binary graph classifier with NumPy.  No raw method signature or
call graph is reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .a3_mvp import A3_MVP_COMPONENTS, compute_reliability


@dataclass
class CachedGraph:
    x: np.ndarray
    source: np.ndarray
    target: np.ndarray
    edge_features: np.ndarray
    edge_type_ids: np.ndarray | None = None
    # Optional APK-level features used by the cache-only fusion ablation.
    # They are kept separate from node features so the graph-only variants
    # remain exactly identifiable.
    extra_features: np.ndarray | None = None
    reliability_values: np.ndarray | None = None


def _sigmoid(value):
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def build_cached_graph(
    payload: Mapping,
    reliability_by_node_id: Mapping[str, Mapping[str, float]],
    training_prior: Mapping[str, float],
    reliability_weights: Mapping[str, float],
    r_min: float,
    r_max: float,
    role_vocabulary_size: int,
    include_reliability: bool = True,
    reliability_components: tuple[str, ...] | None = None,
) -> CachedGraph:
    """Convert one cached payload to a graph input using train-only evidence."""

    active_components = tuple(reliability_components or A3_MVP_COMPONENTS)
    node_ids = list(payload.get("node_ids") or [])
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float64)
    if len(node_ids) == 0:
        base = np.zeros((0, 4), dtype=np.float64)
    if base.ndim != 2 or base.shape[0] != len(node_ids) or base.shape[1] != 4:
        raise ValueError("cached node_base_features must have shape [node_count, 4]")
    roles = list(payload.get("node_role_ids") or [])
    if len(roles) != len(node_ids):
        raise ValueError("cached node_role_ids length mismatch")
    role_features = np.zeros((len(node_ids), role_vocabulary_size), dtype=np.float64)
    reliability = np.zeros(len(node_ids), dtype=np.float64)
    for index, node_id in enumerate(node_ids):
        for role_id in roles[index]:
            role_id = int(role_id)
            if 0 <= role_id < role_vocabulary_size:
                role_features[index, role_id] = 1.0
        components = reliability_by_node_id.get(str(node_id), training_prior)
        if components is None:
            raise KeyError(f"no training reliability prior for node {node_id}")
        reliability[index] = compute_reliability(
            components,
            reliability_weights,
            r_min,
            r_max,
            active_components=active_components,
        )
    feature_parts = [base, role_features]
    if include_reliability:
        feature_parts.append(reliability[:, None])
    x = np.concatenate(feature_parts, axis=1)

    edge_index = payload.get("edge_index") or [[], []]
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    if source.shape != target.shape:
        raise ValueError("cached edge_index source/target length mismatch")
    if len(source) and (
        source.min() < 0
        or target.min() < 0
        or source.max() >= len(node_ids)
        or target.max() >= len(node_ids)
    ):
        raise ValueError("cached edge index is out of bounds")
    edge_features = np.asarray(
        payload.get("edge_features") or [], dtype=np.float64
    )
    if len(edge_features) != len(source):
        raise ValueError("cached edge_features length mismatch")
    edge_type_ids = np.asarray(
        payload.get("edge_type_ids") or [], dtype=np.uint64
    )
    if len(edge_type_ids) != len(source):
        raise ValueError("cached edge_type_ids length mismatch")
    return CachedGraph(
        x=x,
        source=source,
        target=target,
        edge_features=edge_features,
        edge_type_ids=edge_type_ids,
        reliability_values=reliability if include_reliability else None,
    )


class A3GraphClassifier:
    """A compact two-layer message-passing graph classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 24,
        seed: int = 0,
        extra_dim: int = 0,
    ):
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2")
        if input_dim < 1 or extra_dim < 0:
            raise ValueError("input_dim must be positive and extra_dim nonnegative")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.extra_dim = extra_dim
        self.rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(float(input_dim))
        self.w_in = self.rng.normal(0.0, scale, size=(input_dim, hidden_dim))
        self.b_in = np.zeros(hidden_dim, dtype=np.float64)
        self.w_layers = [
            self.rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, hidden_dim))
            for _ in range(2)
        ]
        self.b_layers = [np.zeros(hidden_dim, dtype=np.float64) for _ in range(2)]
        self.classifier_w = self.rng.normal(
            0.0,
            1.0 / np.sqrt(hidden_dim),
            size=(2 * hidden_dim + 6 + extra_dim,),
        )
        self.classifier_b = 0.0

    @staticmethod
    def _aggregate(h, source, target):
        aggregate = np.zeros_like(h)
        degree = np.zeros(h.shape[0], dtype=np.float64)
        if len(source):
            np.add.at(aggregate, target, h[source])
            np.add.at(degree, target, 1.0)
        mask = degree > 0.0
        if np.any(mask):
            aggregate[mask] /= degree[mask, None]
        return aggregate, degree

    def _forward(self, graph: CachedGraph):
        x = graph.x
        h0_pre = x @ self.w_in + self.b_in
        h = np.maximum(h0_pre, 0.0)
        layer_cache = []
        for weights, bias in zip(self.w_layers, self.b_layers):
            aggregate, degree = self._aggregate(h, graph.source, graph.target)
            pre = (h + aggregate) @ weights + bias
            layer_cache.append((h, aggregate, degree, pre))
            h = np.maximum(pre, 0.0)
        n = h.shape[0]
        if n:
            reliability_column = (
                graph.reliability_values
                if graph.reliability_values is not None
                else np.zeros(n, dtype=np.float64)
            )
            mean_h = h.mean(axis=0)
            max_h = h.max(axis=0)
            extras = np.asarray(
                [
                    np.log1p(n),
                    np.log1p(len(graph.source)),
                    float(reliability_column.mean()),
                    float(np.mean(graph.x[:, 1] > 0.0)),
                    float(graph.x[:, 0].mean()),
                    float(graph.x[:, 2].mean()),
                ],
                dtype=np.float64,
            )
        else:
            mean_h = np.zeros(self.hidden_dim)
            max_h = np.zeros(self.hidden_dim)
            extras = np.zeros(6, dtype=np.float64)
        if self.extra_dim:
            if graph.extra_features is None:
                raise ValueError("model expects graph extra_features")
            extra = np.asarray(graph.extra_features, dtype=np.float64)
            if extra.shape != (self.extra_dim,):
                raise ValueError(
                    f"graph extra_features must have shape ({self.extra_dim},), got {extra.shape}"
                )
            extras = np.concatenate([extras, extra])
        elif graph.extra_features is not None:
            raise ValueError("graph extra_features provided to a graph-only model")
        embedding = np.concatenate([mean_h, max_h, extras])
        probability = float(_sigmoid(embedding @ self.classifier_w + self.classifier_b))
        return probability, embedding, h0_pre, layer_cache, h

    def predict_proba(self, graphs: list[CachedGraph]) -> np.ndarray:
        return np.asarray([self._forward(graph)[0] for graph in graphs], dtype=np.float64)

    def fit(
        self,
        graphs: list[CachedGraph],
        labels: np.ndarray,
        learning_rate: float = 0.01,
        l2: float = 1e-4,
        epochs: int = 100,
    ) -> list[float]:
        if len(graphs) != len(labels) or not graphs:
            raise ValueError("graphs and labels must be non-empty and aligned")
        labels = np.asarray(labels, dtype=np.float64)
        history = []
        for _epoch in range(epochs):
            grad_w_in = np.zeros_like(self.w_in)
            grad_b_in = np.zeros_like(self.b_in)
            grad_w_layers = [np.zeros_like(value) for value in self.w_layers]
            grad_b_layers = [np.zeros_like(value) for value in self.b_layers]
            grad_classifier_w = np.zeros_like(self.classifier_w)
            grad_classifier_b = 0.0
            loss = 0.0
            for graph, label in zip(graphs, labels):
                probability, embedding, h0_pre, layer_cache, h_final = self._forward(graph)
                probability = min(1.0 - 1e-12, max(1e-12, probability))
                loss += -label * np.log(probability) - (1.0 - label) * np.log(1.0 - probability)
                d_logit = probability - label
                grad_classifier_w += d_logit * embedding
                grad_classifier_b += d_logit
                d_embedding = d_logit * self.classifier_w
                hidden = self.hidden_dim
                d_h = np.zeros_like(h_final)
                if len(h_final):
                    d_h += d_embedding[:hidden] / len(h_final)
                    argmax = np.argmax(h_final, axis=0)
                    d_h[argmax, np.arange(hidden)] += d_embedding[hidden:2 * hidden]
                for layer_index in range(len(self.w_layers) - 1, -1, -1):
                    h_in, _aggregate, degree, pre = layer_cache[layer_index]
                    d_pre = d_h * (pre > 0.0)
                    grad_w_layers[layer_index] += h_in.T @ d_pre
                    grad_b_layers[layer_index] += d_pre.sum(axis=0)
                    d_h_in = d_pre @ self.w_layers[layer_index].T
                    d_prev = d_h_in.copy()
                    if len(graph.source):
                        d_aggregate = d_h_in
                        valid = degree[graph.target] > 0.0
                        if np.any(valid):
                            np.add.at(
                                d_prev,
                                graph.source[valid],
                                d_aggregate[graph.target[valid]]
                                / degree[graph.target[valid], None],
                            )
                    d_h = d_prev
                d_h0_pre = d_h * (h0_pre > 0.0)
                grad_w_in += graph.x.T @ d_h0_pre
                grad_b_in += d_h0_pre.sum(axis=0)
            count = float(len(graphs))
            regularized = (
                np.sum(self.w_in ** 2)
                + sum(np.sum(value ** 2) for value in self.w_layers)
                + np.sum(self.classifier_w ** 2)
            )
            loss = loss / count + 0.5 * l2 * regularized
            history.append(float(loss))
            grad_w_in = grad_w_in / count + l2 * self.w_in
            grad_b_in = grad_b_in / count
            grad_classifier_w = grad_classifier_w / count + l2 * self.classifier_w
            grad_classifier_b /= count
            for index in range(len(self.w_layers)):
                grad_w_layers[index] = grad_w_layers[index] / count + l2 * self.w_layers[index]
                grad_b_layers[index] /= count
            gradients = [grad_w_in, grad_b_in, grad_classifier_w, grad_classifier_b]
            gradients.extend(grad_w_layers)
            gradients.extend(grad_b_layers)
            for gradient in gradients:
                if np.isscalar(gradient):
                    continue
                np.clip(gradient, -5.0, 5.0, out=gradient)
            self.w_in -= learning_rate * grad_w_in
            self.b_in -= learning_rate * grad_b_in
            self.classifier_w -= learning_rate * grad_classifier_w
            self.classifier_b -= learning_rate * grad_classifier_b
            for index in range(len(self.w_layers)):
                self.w_layers[index] -= learning_rate * grad_w_layers[index]
                self.b_layers[index] -= learning_rate * grad_b_layers[index]
        return history
