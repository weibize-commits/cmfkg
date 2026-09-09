from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LABELS = [
    "SUPPORTED",
    "NOT_SUPPORTED",
    "INSUFFICIENT_CONTEXT",
    "ENTITY_OR_TYPE_ERROR",
]


class EvidenceConditionedPathReasoner(nn.Module):
    """RED-GNN-style propagation over target-edge-deleted enumerated paths."""

    def __init__(
        self,
        *,
        edge_vocab_size: int,
        type_vocab_size: int,
        hidden_dim: int,
        text_dim: int,
        max_hops: int,
        evidence_conditioned: bool,
        evidence_path_attention: bool = False,
        output_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops
        self.evidence_conditioned = evidence_conditioned
        self.evidence_path_attention = evidence_path_attention
        self.output_cross_attention = output_cross_attention
        self.edge_embedding = nn.Embedding(edge_vocab_size, hidden_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(type_vocab_size, hidden_dim, padding_idx=0)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.graph_meta_projection = nn.Linear(6, hidden_dim)
        self.path_feature_projection = nn.Linear(3, hidden_dim)
        self.initial = nn.Linear(hidden_dim * 3, hidden_dim)
        self.message = nn.Linear(hidden_dim * 5, hidden_dim)
        self.graph_gate = nn.Linear(hidden_dim * 5, 1)
        self.evidence_strength = (
            nn.Parameter(torch.tensor(0.10033535)) if evidence_conditioned else None
        )
        # Ascend torch-npu 2.3 does not implement aten::_thnn_fused_gru_cell.
        # Keep a GRU-style gated state update in explicit primitive operations.
        self.update_candidate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.path_score = nn.Linear(hidden_dim * (3 if output_cross_attention else 2), 1)
        self.no_path = nn.Parameter(torch.zeros(hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Keep every shared layer identically initialized in the no-ECRP and
        # ECRP models. The ECRP-only interaction layer is allocated last.
        self.evidence_gate = (
            nn.Linear(hidden_dim * 6, 1) if evidence_conditioned else None
        )
        self.evidence_path_score = (
            nn.Linear(hidden_dim * 4, 1)
            if evidence_conditioned and evidence_path_attention
            else None
        )
        self.evidence_path_strength = (
            nn.Parameter(torch.tensor(0.10033535))
            if evidence_conditioned and evidence_path_attention
            else None
        )
        if self.evidence_gate is not None:
            nn.init.zeros_(self.evidence_gate.weight)
            nn.init.zeros_(self.evidence_gate.bias)
            if self.evidence_path_score is not None:
                nn.init.zeros_(self.evidence_path_score.weight)
                nn.init.zeros_(self.evidence_path_score.bias)

    def forward(self, graph: dict[str, Tensor], text_state: Tensor) -> tuple[Tensor, Tensor]:
        edges = graph["edge_ids"]
        node_types = graph["node_type_ids"]
        hop_mask = graph["hop_mask"].bool()
        path_mask = graph["path_mask"].bool()
        batch_size, path_count, _ = edges.shape
        query = self.edge_embedding(graph["query_relation_id"])
        head_type = self.type_embedding(graph["head_type_id"])
        tail_type = self.type_embedding(graph["tail_type_id"])
        text = torch.tanh(self.text_projection(text_state))
        graph_meta = torch.tanh(self.graph_meta_projection(graph["graph_features"].float()))
        query_paths = query[:, None, :].expand(-1, path_count, -1)
        text_paths = text[:, None, :].expand(-1, path_count, -1)
        meta_paths = graph_meta[:, None, :].expand(-1, path_count, -1)
        source_type = self.type_embedding(node_types[:, :, 0])
        endpoints = (head_type + tail_type)[:, None, :].expand(-1, path_count, -1)
        state = torch.tanh(self.initial(torch.cat([source_type, query_paths, endpoints], dim=-1)))

        for hop in range(self.max_hops):
            edge = self.edge_embedding(edges[:, :, hop])
            source = self.type_embedding(node_types[:, :, hop])
            target = self.type_embedding(node_types[:, :, hop + 1])
            structural = torch.cat([state, edge, query_paths, source, target], dim=-1)
            message = torch.tanh(self.message(structural))
            gate = torch.sigmoid(self.graph_gate(structural))
            if self.evidence_conditioned:
                compatibility = torch.sigmoid(
                    self.evidence_gate(
                        torch.cat(
                            [
                                message,
                                query_paths,
                                text_paths,
                                meta_paths,
                                message * text_paths,
                                torch.abs(message - text_paths),
                            ],
                            dim=-1,
                        )
                    )
                )
                # The zero-initialized compatibility layer starts at 0.5, so
                # propagation exactly matches no-ECRP while retaining a
                # nonzero gradient for learning path-evidence interactions.
                strength = torch.tanh(self.evidence_strength)
                gate = gate * (1.0 + strength * (2.0 * compatibility - 1.0))
            gated_message = message * gate
            update_input = torch.cat([gated_message, state], dim=-1)
            candidate = torch.tanh(self.update_candidate(update_input))
            update_gate = torch.sigmoid(self.update_gate(update_input))
            updated = (1.0 - update_gate) * state + update_gate * candidate
            state = torch.where(hop_mask[:, :, hop, None], updated, state)

        state = self.layer_norm(
            state + torch.tanh(self.path_feature_projection(graph["path_features"].float()))
        )
        score_inputs = [state, query_paths]
        if self.output_cross_attention:
            score_inputs.append(text_paths)
        scores = self.path_score(torch.cat(score_inputs, dim=-1)).squeeze(-1)
        if self.evidence_path_score is not None:
            evidence_scores = self.evidence_path_score(
                torch.cat(
                    [
                        state,
                        text_paths,
                        state * text_paths,
                        torch.abs(state - text_paths),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            scores = scores + torch.tanh(self.evidence_path_strength) * evidence_scores
        scores = scores.masked_fill(~path_mask, -1.0e4)
        weights = torch.softmax(scores, dim=1)
        weights = weights * path_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        pooled = torch.sum(weights[:, :, None] * state, dim=1)
        has_path = path_mask.any(dim=1, keepdim=True)
        pooled = torch.where(has_path, pooled, self.no_path[None, :].expand(batch_size, -1))
        return pooled, weights


class FERAKGModel(nn.Module):
    def __init__(
        self,
        *,
        checkpoint: str,
        family: str,
        edge_vocab_size: int,
        type_vocab_size: int,
        tokenizer_size: int,
        hidden_dim: int = 192,
        max_hops: int = 3,
        dropout: float = 0.15,
        text_pooling: str = "cls",
        marker_token_ids: Optional[dict[str, int]] = None,
        class_weights: Optional[Tensor] = None,
        cer_margin: float = 0.20,
        cer_weight: float = 0.50,
        evidence_path_attention: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        valid = {
            "text_only",
            "path_text",
            "path_text_hierarchical",
            "graph_only",
            "graph_hierarchical",
            "late_fusion",
            "output_cross_fusion",
            "no_ecrp",
            "fera_kg",
            "fera_flat",
            "graph_residual",
        }
        if family not in valid:
            raise ValueError(f"unknown family: {family}")
        self.family = family
        self.hierarchical = family in {
            "path_text_hierarchical",
            "graph_hierarchical",
            "no_ecrp",
            "fera_kg",
        }
        self.graph_without_text = family in {"graph_only", "graph_hierarchical"}
        if text_pooling not in {"cls", "mean", "learned_mix", "entity_pair"}:
            raise ValueError(f"unknown text pooling: {text_pooling}")
        self.text_pooling = text_pooling
        self.marker_token_ids = marker_token_ids or {}
        self.cer_margin = cer_margin
        self.cer_weight = cer_weight
        self.text_encoder: Optional[nn.Module]
        if self.graph_without_text:
            self.text_encoder = None
            text_dim = hidden_dim
        else:
            self.text_encoder = AutoModel.from_pretrained(checkpoint)
            self.text_encoder.resize_token_embeddings(tokenizer_size)
            text_dim = int(self.text_encoder.config.hidden_size)
        self.pooling_logit = nn.Parameter(torch.tensor(0.0)) if text_pooling == "learned_mix" else None
        self.entity_pair_pooler = (
            nn.Sequential(nn.Linear(text_dim * 5, text_dim), nn.GELU(), nn.LayerNorm(text_dim))
            if text_pooling == "entity_pair"
            else None
        )
        self.graph_reasoner: Optional[EvidenceConditionedPathReasoner]
        if family in {"text_only", "path_text", "path_text_hierarchical"}:
            self.graph_reasoner = None
        else:
            self.graph_reasoner = EvidenceConditionedPathReasoner(
                edge_vocab_size=edge_vocab_size,
                type_vocab_size=type_vocab_size,
                hidden_dim=hidden_dim,
                text_dim=text_dim,
                max_hops=max_hops,
                evidence_conditioned=family in {"fera_kg", "fera_flat"},
                evidence_path_attention=evidence_path_attention,
                output_cross_attention=family == "output_cross_fusion",
            )
        if family in {"text_only", "path_text", "path_text_hierarchical"}:
            fusion_dim = text_dim
        elif self.graph_without_text or family == "graph_residual":
            fusion_dim = hidden_dim
        elif family == "output_cross_fusion":
            fusion_dim = text_dim + hidden_dim * 3
        else:
            fusion_dim = text_dim + hidden_dim
        self.cross_text = nn.Linear(text_dim, hidden_dim) if family == "output_cross_fusion" else None
        self.cross_graph = nn.Linear(hidden_dim, hidden_dim) if family == "output_cross_fusion" else None
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_fusion = (
            nn.Sequential(
                nn.Linear(text_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            if family == "graph_residual"
            else None
        )
        self.residual_gate_logit = (
            nn.Parameter(torch.tensor(-2.944439))
            if family == "graph_residual"
            else None
        )
        self.flat_head = nn.Linear(hidden_dim, len(LABELS))
        self.judgeability_head = nn.Linear(hidden_dim, 1)
        self.support_head = nn.Linear(hidden_dim, 1)
        self.reason_head = nn.Linear(hidden_dim, 1)
        if class_weights is None:
            class_weights = torch.ones(len(LABELS), dtype=torch.float32)
        self.register_buffer("class_weights", class_weights.float())

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self.text_encoder is None:
            return torch.zeros(input_ids.shape[0], self.fusion[0].in_features, device=input_ids.device)
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # SikuRoBERTa is distributed as a masked-language-model checkpoint and
        # does not provide a pretrained pooler. Use the pretrained CLS state
        # instead of a randomly initialized pooler output.
        cls_state = output.last_hidden_state[:, 0]
        if self.text_pooling == "cls":
            return cls_state
        mask = attention_mask[:, :, None].to(output.last_hidden_state.dtype)
        mean_state = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if self.text_pooling == "mean":
            return mean_state
        if self.text_pooling == "entity_pair":
            def marker_state(name: str) -> Tensor:
                token_id = int(self.marker_token_ids[name])
                marker_mask = (input_ids == token_id)[:, :, None].to(output.last_hidden_state.dtype)
                state = (output.last_hidden_state * marker_mask).sum(dim=1)
                count = marker_mask.sum(dim=1)
                state = state / count.clamp_min(1.0)
                return torch.where(count > 0, state, cls_state)

            relation_state = marker_state("[REL]")
            head_state = marker_state("[H]")
            tail_state = marker_state("[T]")
            return self.entity_pair_pooler(
                torch.cat([cls_state, mean_state, relation_state, head_state, tail_state], dim=-1)
            )
        mixing_weight = torch.sigmoid(self.pooling_logit)
        return mixing_weight * cls_state + (1.0 - mixing_weight) * mean_state

    def _features(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        graph: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        if self.graph_without_text:
            text = torch.zeros(input_ids.shape[0], self.graph_reasoner.hidden_dim, device=input_ids.device)
        else:
            text = self.encode_text(input_ids, attention_mask)
        if self.graph_reasoner is None:
            fused_input, path_weights = text, torch.empty(input_ids.shape[0], 0, device=input_ids.device)
        else:
            graph_state, path_weights = self.graph_reasoner(graph, text)
            if self.graph_without_text:
                fused_input = graph_state
            elif self.family == "graph_residual":
                graph_features = self.fusion(graph_state)
                residual_features = self.residual_fusion(torch.cat([text, graph_state], dim=-1))
                residual_gate = 0.25 * torch.sigmoid(self.residual_gate_logit)
                return graph_features + residual_gate * residual_features, path_weights
            elif self.family == "output_cross_fusion":
                text_cross = torch.tanh(self.cross_text(text))
                graph_cross = torch.tanh(self.cross_graph(graph_state))
                fused_input = torch.cat(
                    [text, graph_state, text_cross * graph_cross, torch.abs(text_cross - graph_cross)],
                    dim=-1,
                )
            else:
                fused_input = torch.cat([text, graph_state], dim=-1)
        return self.fusion(fused_input), path_weights

    def _probabilities(self, features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.hierarchical:
            logits = self.flat_head(features)
            return torch.softmax(logits, dim=-1), {"flat_logits": logits}
        judgeable_logit = self.judgeability_head(features).squeeze(-1)
        support_logit = self.support_head(features).squeeze(-1)
        reason_logit = self.reason_head(features).squeeze(-1)
        judgeable = torch.sigmoid(judgeable_logit)
        support = torch.sigmoid(support_logit)
        insufficient = torch.sigmoid(reason_logit)
        probabilities = torch.stack(
            [
                judgeable * support,
                judgeable * (1.0 - support),
                (1.0 - judgeable) * insufficient,
                (1.0 - judgeable) * (1.0 - insufficient),
            ],
            dim=-1,
        )
        return probabilities, {
            "judgeability_logit": judgeable_logit,
            "support_logit": support_logit,
            "reason_logit": reason_logit,
        }

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        graph: dict[str, Tensor],
        labels: Optional[Tensor] = None,
        sample_weights: Optional[Tensor] = None,
        counterfactual_input_ids: Optional[Tensor] = None,
        counterfactual_attention_mask: Optional[Tensor] = None,
        counterfactual_mask: Optional[Tensor] = None,
    ) -> dict[str, Any]:
        features, path_weights = self._features(input_ids, attention_mask, graph)
        probabilities, auxiliary = self._probabilities(features)
        output: dict[str, Any] = {
            "probabilities": probabilities,
            "path_weights": path_weights,
            **auxiliary,
        }
        if self.residual_gate_logit is not None:
            output["residual_gate"] = 0.25 * torch.sigmoid(self.residual_gate_logit)
        if labels is None:
            return output
        # Compute logarithms in FP32 even when the encoder runs under AMP.
        # The previous 1e-8 floor underflowed to zero in float16, which could
        # produce -inf values and eventually a NaN counterfactual loss.
        log_probabilities = torch.log(probabilities.float().clamp_min(1.0e-8))
        per_item_loss = F.nll_loss(
            log_probabilities,
            labels,
            weight=self.class_weights,
            reduction="none",
        )
        if sample_weights is None:
            loss = per_item_loss.mean()
        else:
            normalized_weights = sample_weights.float().clamp_min(0.0)
            loss = (per_item_loss * normalized_weights).sum() / normalized_weights.sum().clamp_min(1.0e-8)
        output["classification_loss"] = loss.detach()
        if (
            self.family in {"fera_kg", "fera_flat"}
            and self.cer_weight > 0.0
            and counterfactual_input_ids is not None
            and counterfactual_attention_mask is not None
            and counterfactual_mask is not None
            and bool(counterfactual_mask.any())
        ):
            cf_features, _ = self._features(
                counterfactual_input_ids,
                counterfactual_attention_mask,
                graph,
            )
            cf_probabilities, _ = self._probabilities(cf_features)
            positive_score = log_probabilities[:, 0]
            counterfactual_score = torch.log(
                cf_probabilities[:, 0].float().clamp_min(1.0e-8)
            )
            ranking = F.relu(self.cer_margin - positive_score + counterfactual_score)
            active = counterfactual_mask.bool()
            ranking = ranking[active]
            if sample_weights is None:
                ranking = ranking.mean()
            else:
                ranking_weights = sample_weights.float()[active].clamp_min(0.0)
                ranking = (ranking * ranking_weights).sum() / ranking_weights.sum().clamp_min(1.0e-8)
            loss = loss + self.cer_weight * ranking
            output["counterfactual_loss"] = ranking.detach()
        output["loss"] = loss
        return output
