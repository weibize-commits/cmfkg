from __future__ import annotations

import unittest

try:
    import torch
    from fera_kg.deep_models import EvidenceConditionedPathReasoner
except ModuleNotFoundError:  # The lightweight local audit environment omits PyTorch.
    torch = None
    EvidenceConditionedPathReasoner = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the local audit environment")
class EvidenceConditioningTests(unittest.TestCase):
    def test_zero_compatibility_layer_matches_no_ecrp_at_initialization(self) -> None:
        arguments = {
            "edge_vocab_size": 12,
            "type_vocab_size": 9,
            "hidden_dim": 8,
            "text_dim": 16,
            "max_hops": 3,
        }
        torch.manual_seed(17)
        baseline = EvidenceConditionedPathReasoner(
            **arguments,
            evidence_conditioned=False,
        ).eval()
        torch.manual_seed(17)
        conditioned = EvidenceConditionedPathReasoner(
            **arguments,
            evidence_conditioned=True,
        ).eval()

        graph = {
            "edge_ids": torch.tensor([[[2, 3, 0], [4, 0, 0]], [[5, 6, 7], [0, 0, 0]]]),
            "node_type_ids": torch.tensor(
                [[[2, 3, 4, 0], [2, 5, 0, 0]], [[3, 4, 5, 6], [0, 0, 0, 0]]]
            ),
            "hop_mask": torch.tensor(
                [[[True, True, False], [True, False, False]], [[True, True, True], [False, False, False]]]
            ),
            "path_mask": torch.tensor([[True, True], [True, False]]),
            "path_features": torch.tensor(
                [[[1.0, 0.5, 0.67], [0.0, 0.33, 0.33]], [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]
            ),
            "graph_features": torch.tensor(
                [[0.4, 0.5, 0.5, 0.2, 1.0, 1.0], [0.2, 1.0, 0.0, 0.1, 1.0, 0.0]]
            ),
            "query_relation_id": torch.tensor([2, 5]),
            "head_type_id": torch.tensor([2, 3]),
            "tail_type_id": torch.tensor([4, 6]),
        }
        text = torch.randn(2, 16)

        baseline_state, baseline_weights = baseline(graph, text)
        conditioned_state, conditioned_weights = conditioned(graph, text)
        self.assertTrue(torch.allclose(baseline_state, conditioned_state, atol=1.0e-7))
        self.assertTrue(torch.allclose(baseline_weights, conditioned_weights, atol=1.0e-7))

        torch.manual_seed(17)
        conditioned_attention = EvidenceConditionedPathReasoner(
            **arguments,
            evidence_conditioned=True,
            evidence_path_attention=True,
        ).eval()
        with torch.no_grad():
            conditioned_attention.evidence_path_score.weight.fill_(0.05)
        _, learned_weights = conditioned_attention(graph, text)
        self.assertFalse(torch.allclose(baseline_weights[0], learned_weights[0], atol=1.0e-7))


if __name__ == "__main__":
    unittest.main()
