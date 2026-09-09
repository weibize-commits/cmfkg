from fera_kg.entity_identity import (
    add_identity_feature_records,
    add_identity_features,
    fit_identity_lexicon,
    normalize_source_title,
    normalize_surface,
)

import pandas as pd


def test_normalization_is_conservative() -> None:
    assert normalize_surface(" 千 金方 ") == "千金方"
    assert normalize_source_title("23 粥谱（二种）.txt") == "粥谱"


def test_training_lexicon_excludes_identity_errors() -> None:
    rows = [
        {
            "label": "SUPPORTED",
            "head_text": "生姜",
            "head_type": "食材",
            "tail_text": "胃经",
            "tail_type": "归经",
            "source_name": "01 千金方.txt",
            "book": "千金方",
        },
        {
            "label": "ENTITY_OR_TYPE_ERROR",
            "head_text": "千金方",
            "head_type": "食疗方",
            "tail_text": "生姜",
            "tail_type": "病症",
            "source_name": "01 千金方.txt",
            "book": "千金方",
        },
    ]
    lexicon = fit_identity_lexicon(rows)
    assert lexicon.surface_type_counts["生姜"]["食材"] == 1
    assert lexicon.surface_type_counts["生姜"]["病症"] == 0
    assert "千金方" in lexicon.source_titles


def test_identity_features_detect_type_conflict_and_source_title() -> None:
    train = [
        {
            "label": "SUPPORTED",
            "head_text": "生姜",
            "head_type": "食材",
            "tail_text": "胃经",
            "tail_type": "归经",
            "source_name": "千金方.txt",
            "book": "千金方",
        }
    ]
    row = {
        "passage_text": "《千金方》曰，生姜入胃经。",
        "head_text": "千金方",
        "head_type": "食疗方",
        "tail_text": "生姜",
        "tail_type": "病症",
    }
    frame = add_identity_features(pd.DataFrame({"base": [1.0]}), [row], fit_identity_lexicon(train))
    assert frame.loc[0, "head__identity_known_source_title"] == 1.0
    assert frame.loc[0, "tail__identity_seen_only_as_other_type"] == 1.0


def test_precomputed_feature_records_must_align() -> None:
    frame = pd.DataFrame({"base": [1.0]})
    result = add_identity_feature_records(frame, [{"identity": 0.5}])
    assert result.loc[0, "identity"] == 0.5
