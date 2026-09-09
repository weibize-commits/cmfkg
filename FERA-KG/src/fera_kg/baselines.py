from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .benchmark import model_text


def _as_int32_sparse(matrix: Any) -> Any:
    """Keep sparse matrices compatible with estimators that require 32-bit indices."""
    from numpy import int32

    matrix = matrix.tocsr(copy=False)
    matrix.indices = matrix.indices.astype(int32, copy=False)
    matrix.indptr = matrix.indptr.astype(int32, copy=False)
    return matrix


class MajorityBaseline:
    def __init__(self) -> None:
        self.label = ""

    def fit(self, items: list[dict[str, Any]]) -> "MajorityBaseline":
        self.label = Counter(item["label"] for item in items).most_common(1)[0][0]
        return self

    def predict(self, items: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
        return [self.label] * len(items), [1.0] * len(items)


class ConditionalPriorBaseline:
    def __init__(self, field: str) -> None:
        self.field = field
        self.fallback = ""
        self.mapping: dict[str, str] = {}

    def fit(self, items: list[dict[str, Any]]) -> "ConditionalPriorBaseline":
        overall = Counter(item["label"] for item in items)
        self.fallback = overall.most_common(1)[0][0]
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for item in items:
            grouped[str(item[self.field])][item["label"]] += 1
        self.mapping = {value: counts.most_common(1)[0][0] for value, counts in grouped.items()}
        return self

    def predict(self, items: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
        labels = [self.mapping.get(str(item[self.field]), self.fallback) for item in items]
        return labels, [1.0] * len(items)


class TfidfLogisticBaseline:
    def __init__(self, *, c: float, seed: int, max_features: int = 80000) -> None:
        from numpy import float32
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                    dtype=float32,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed,
                        solver="liblinear",
                    )
                ),
            ),
        ])

    def fit(self, items: list[dict[str, Any]]) -> "TfidfLogisticBaseline":
        self.pipeline.fit([model_text(item) for item in items], [item["label"] for item in items])
        return self

    def predict(self, items: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
        texts = [model_text(item) for item in items]
        labels = self.pipeline.predict(texts).tolist()
        probabilities = self.pipeline.predict_proba(texts)
        confidence = probabilities.max(axis=1).tolist()
        return labels, confidence

    def predict_probabilities(self, items: list[dict[str, Any]]) -> tuple[list[str], Any]:
        texts = [model_text(item) for item in items]
        classifier = self.pipeline.named_steps["classifier"]
        return classifier.classes_.tolist(), self.pipeline.predict_proba(texts)


def graph_feature_dict(item: dict[str, Any], path_item: dict[str, Any]) -> dict[str, float]:
    """Create graph-path features without using the deleted target edge audit count."""
    features: dict[str, float] = {
        "bias": 1.0,
        f"relation={item['relation']}": 1.0,
        f"relation_group={item['relation_group']}": 1.0,
        f"head_type={item['head_type']}": 1.0,
        f"tail_type={item['tail_type']}": 1.0,
        f"type_pair={item['head_type']}->{item['tail_type']}": 1.0,
        "path_available": float(path_item["path_count"] > 0),
        "path_count": float(path_item["path_count"]),
        "passage_conditioned_path_count": float(path_item["passage_conditioned_path_count"]),
        "semantic_only_path_count": float(path_item["semantic_only_path_count"]),
        "local_typed_mentions": float(path_item["local_inventory"]["actual_typed_mentions"]),
        "head_in_actual_inventory": float(path_item["local_inventory"]["head_in_actual_inventory"]),
        "tail_in_actual_inventory": float(path_item["local_inventory"]["tail_in_actual_inventory"]),
    }
    paths = path_item.get("paths") or []
    if paths:
        features["shortest_hops"] = float(min(path["hops"] for path in paths))
        features["mean_hops"] = sum(float(path["hops"]) for path in paths) / len(paths)
        features["mean_semantic_edges"] = (
            sum(float(path["semantic_edges"]) for path in paths) / len(paths)
        )
    else:
        features["no_path"] = 1.0
    for path in paths:
        edges = [str(edge) for edge in path.get("edges") or []]
        nodes = path.get("nodes") or []
        path_kind = "passage" if path.get("passage_conditioned") else "semantic"
        features[f"path_kind={path_kind}"] = features.get(f"path_kind={path_kind}", 0.0) + 1.0
        features[f"edge_sequence={'|'.join(edges)}"] = (
            features.get(f"edge_sequence={'|'.join(edges)}", 0.0) + 1.0
        )
        node_types = [str(node.get("type", "")) for node in nodes]
        features[f"node_type_sequence={'|'.join(node_types)}"] = (
            features.get(f"node_type_sequence={'|'.join(node_types)}", 0.0) + 1.0
        )
        for edge in edges:
            features[f"edge={edge}"] = features.get(f"edge={edge}", 0.0) + 1.0
    return features


class GraphPathLogisticBaseline:
    def __init__(self, *, c: float, seed: int) -> None:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
                solver="liblinear",
            )
        )

    def fit(
        self,
        items: list[dict[str, Any]],
        path_items: list[dict[str, Any]],
    ) -> "GraphPathLogisticBaseline":
        matrix = self.vectorizer.fit_transform(
            [graph_feature_dict(item, path_item) for item, path_item in zip(items, path_items)]
        )
        matrix = _as_int32_sparse(matrix)
        self.classifier.fit(matrix, [item["label"] for item in items])
        return self

    def predict(
        self,
        items: list[dict[str, Any]],
        path_items: list[dict[str, Any]],
    ) -> tuple[list[str], list[float]]:
        matrix = self.vectorizer.transform(
            [graph_feature_dict(item, path_item) for item, path_item in zip(items, path_items)]
        )
        matrix = _as_int32_sparse(matrix)
        labels = self.classifier.predict(matrix).tolist()
        probabilities = self.classifier.predict_proba(matrix)
        return labels, probabilities.max(axis=1).tolist()

    def predict_probabilities(
        self,
        items: list[dict[str, Any]],
        path_items: list[dict[str, Any]],
    ) -> tuple[list[str], Any]:
        matrix = self.vectorizer.transform(
            [graph_feature_dict(item, path_item) for item, path_item in zip(items, path_items)]
        )
        matrix = _as_int32_sparse(matrix)
        return self.classifier.classes_.tolist(), self.classifier.predict_proba(matrix)


class GraphTextFusionBaseline:
    def __init__(self, *, c: float, seed: int, max_features: int = 80000) -> None:
        from numpy import float32
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        self.text_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
            dtype=float32,
        )
        self.graph_vectorizer = DictVectorizer(sparse=True)
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
                solver="liblinear",
            )
        )

    def fit(
        self,
        items: list[dict[str, Any]],
        path_items: list[dict[str, Any]],
    ) -> "GraphTextFusionBaseline":
        from scipy.sparse import hstack

        text_matrix = self.text_vectorizer.fit_transform([model_text(item) for item in items])
        graph_matrix = self.graph_vectorizer.fit_transform(
            [graph_feature_dict(item, path_item) for item, path_item in zip(items, path_items)]
        )
        matrix = _as_int32_sparse(hstack([text_matrix, graph_matrix], format="csr"))
        self.classifier.fit(matrix, [item["label"] for item in items])
        return self

    def predict(
        self,
        items: list[dict[str, Any]],
        path_items: list[dict[str, Any]],
    ) -> tuple[list[str], list[float]]:
        from scipy.sparse import hstack

        text_matrix = self.text_vectorizer.transform([model_text(item) for item in items])
        graph_matrix = self.graph_vectorizer.transform(
            [graph_feature_dict(item, path_item) for item, path_item in zip(items, path_items)]
        )
        matrix = _as_int32_sparse(hstack([text_matrix, graph_matrix], format="csr"))
        labels = self.classifier.predict(matrix).tolist()
        probabilities = self.classifier.predict_proba(matrix)
        return labels, probabilities.max(axis=1).tolist()

    def predict_probabilities(
        self,
        items: list[dict[str, Any]],
        path_items: list[dict[str, Any]],
    ) -> tuple[list[str], Any]:
        from scipy.sparse import hstack

        text_matrix = self.text_vectorizer.transform([model_text(item) for item in items])
        graph_matrix = self.graph_vectorizer.transform(
            [graph_feature_dict(item, path_item) for item, path_item in zip(items, path_items)]
        )
        matrix = _as_int32_sparse(hstack([text_matrix, graph_matrix], format="csr"))
        return self.classifier.classes_.tolist(), self.classifier.predict_proba(matrix)
