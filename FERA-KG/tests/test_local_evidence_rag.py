from __future__ import annotations

from fera_kg.local_evidence_rag import omission_variant, split_segments


def test_split_segments_preserves_offsets() -> None:
    text = "第一句。\n第二句；第三句。"
    segments = split_segments(text)
    assert [segment[2] for segment in segments] == ["第一句。", "第二句；", "第三句。"]
    for start, end, content in segments:
        assert text[start:end] == content


def test_omission_variant_preserves_endpoints_and_length() -> None:
    passage = "粥方使用生姜，可以温中。"
    item = {
        "passage_text": passage,
        "head_start": 0,
        "head_end": 2,
        "tail_start": 4,
        "tail_end": 6,
    }
    variant = omission_variant(item)
    assert len(variant["passage_text"]) == len(passage)
    assert variant["passage_text"][0:2] == "粥方"
    assert variant["passage_text"][4:6] == "生姜"
    assert "□" in variant["passage_text"]
