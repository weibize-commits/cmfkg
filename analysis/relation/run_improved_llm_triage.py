from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "paper_revision" / "relation_improvement"
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
RUN_DIR = EXPERIMENT_DIR / "llm_runs"
PREPARE_PATH = EXPERIMENT_DIR / "analysis" / "prepare_and_tune_local.py"

MODEL = "DeepSeek-v3.2-DP-cntrain09"
BASE_URL = "https://maas.bit.edu.cn/v1-openai"
API_KEY_ENV = "BIT_GPUSTACK_API_KEY"
PROMPT_VERSION = "cmfkg_relation_triage_v2_20260731"
CHUNK_SIZE = 12
CONCURRENCY = 6
MAX_TOKENS = 1800


RELATION_DEFINITIONS: dict[str, str] = {
    "hasNature": "食材具有某种食性（寒、凉、平、温或热）",
    "hasFlavor": "食材具有某种食味（酸、苦、甘、辛或咸）",
    "entersmeridian": "食材归入某经",
    "hasEffect": "食材具有某种功效",
    "treats": "食材治疗或缓解某病症",
    "containsIngredient": "食疗方含有某食材",
    "usesCookingMethod": "食疗方使用某烹饪方法",
    "treatedByRecipe": "病症由某食疗方治疗或缓解",
    "contraindicates": "食材存在某种食用禁忌",
    "affectsOrgan": "病症涉及或影响某脏腑",
    "correspondsToOrgan": "归经对应某脏腑",
    "recordedInText": "实体在某古籍中有记载",
    "authoredBy": "古籍由某医家所著",
    "recommendsIngredient": "养生原则推荐某食材",
}


SYSTEM_PROMPT = """你是中医古籍关系标注复核员。你的任务是复现专家标注规范，而不是判断现代临床真伪。
只能依据每个候选给出的局部证据、实体边界、实体类型和关系方向作判断，不得补充外部知识。
古籍短句、标题接续、条目式列举和轻度OCR噪声可以构成结构性证据；但无归属关系的远距离共现不能算作证据。
你必须为每个候选返回一个决定，不得遗漏。输出必须是单个JSON对象，不要输出Markdown。"""


USER_TEMPLATE = """请逐一判断下列候选关系。

标签定义：
- supported：局部证据直接表达关系，或条目结构、配方列举、属性列举清楚地把尾实体归属于头实体。
- plausible：古文省略、标题接续或轻度OCR噪声使表达不完整，但局部结构仍较可能支持该关系，且没有明显的其他归属对象。
- unsupported：只是共现、关系方向不符、实体属于其他条目，或证据不足。

注意：
1. “甘、温”“入脾经”等紧邻食材条目的属性表达可判为supported。
2. 配方标题后的原料或制法列表可支持containsIngredient或usesCookingMethod。
3. 不要因为文字残缺就自动否定；若结构仍可辨认，可判为plausible。
4. 若实体边界看起来异常，但给定标注结构仍明确连接这两个实体，只评价关系连接，不要自行改实体。
5. 每个candidate_id必须恰好返回一次。

候选：
{candidate_blocks}

严格返回：
{{"decisions":[{{"candidate_id":"B001","label":"supported","evidence":"不超过30字的原文片段","confidence":0.90}}]}}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def safe_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def compact_entity(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_char": int(entity["start_char"]),
        "end_char": int(entity["end_char"]),
        "text": str(entity["text"]),
        "type": str(entity["type"]),
    }


def candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    head = candidate["head"]
    tail = candidate["tail"]
    return (
        str(candidate["sample_id"]),
        int(head["start_char"]),
        int(head["end_char"]),
        safe_text(head["text"]),
        str(head["type"]),
        str(candidate["relation"]),
        int(tail["start_char"]),
        int(tail["end_char"]),
        safe_text(tail["text"]),
        str(tail["type"]),
    )


def candidate_digest(candidate: dict[str, Any]) -> str:
    payload = json.dumps(candidate_key(candidate), ensure_ascii=False, separators=(",", ":"))
    return sha256_text(payload)[:20]


def batch_digest(candidates: list[dict[str, Any]]) -> str:
    payload = "|".join(candidate_digest(candidate) for candidate in candidates)
    return sha256_text(payload)[:20]


def make_prompt(candidates: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    blocks: list[str] = []
    candidate_map: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = f"B{index:03d}"
        candidate_map[candidate_id] = candidate
        head = candidate["head"]
        tail = candidate["tail"]
        relation = str(candidate["relation"])
        blocks.append(
            "\n".join(
                [
                    f"[{candidate_id}]",
                    f"证据：{safe_text(candidate.get('evidence_text'))[:500]}",
                    f"头实体：{safe_text(head['text'])}（{head['type']}）",
                    f"关系：{relation}（{RELATION_DEFINITIONS.get(relation, relation)}）",
                    f"尾实体：{safe_text(tail['text'])}（{tail['type']}）",
                ]
            )
        )
    return USER_TEMPLATE.format(candidate_blocks="\n\n".join(blocks)), candidate_map


def extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError(f"Cannot parse JSON object: {text[:300]}")


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "supported": "supported",
        "support": "supported",
        "yes": "supported",
        "true": "supported",
        "支持": "supported",
        "plausible": "plausible",
        "uncertain": "plausible",
        "可能": "plausible",
        "不确定": "plausible",
        "unsupported": "unsupported",
        "not_supported": "unsupported",
        "no": "unsupported",
        "false": "unsupported",
        "不支持": "unsupported",
    }
    return mapping.get(text, "unsupported")


def parse_decisions(
    parsed: dict[str, Any],
    candidate_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    values = parsed.get("decisions")
    if not isinstance(values, list):
        raise ValueError("decisions must be a list")
    parsed_by_id: dict[str, dict[str, Any]] = {}
    unknown_ids: list[str] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        candidate_id = str(value.get("candidate_id") or value.get("id") or "").strip().upper()
        match = re.fullmatch(r"B0*(\d+)", candidate_id)
        if match:
            candidate_id = f"B{int(match.group(1)):03d}"
        if candidate_id not in candidate_map:
            if candidate_id:
                unknown_ids.append(candidate_id)
            continue
        try:
            confidence = float(value.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        parsed_by_id[candidate_id] = {
            "label": normalize_label(value.get("label")),
            "evidence": safe_text(value.get("evidence"))[:100],
            "confidence": max(0.0, min(1.0, confidence)),
        }
    missing_ids = sorted(set(candidate_map) - set(parsed_by_id))
    decisions: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_map.items():
        decision = parsed_by_id.get(
            candidate_id,
            {"label": "unsupported", "evidence": "", "confidence": 0.0},
        )
        decisions.append(
            {
                "candidate_id": candidate_id,
                "candidate_digest": candidate_digest(candidate),
                "sample_id": str(candidate["sample_id"]),
                "head": compact_entity(candidate["head"]),
                "relation": str(candidate["relation"]),
                "tail": compact_entity(candidate["tail"]),
                "evidence_text": str(candidate.get("evidence_text") or ""),
                **decision,
            }
        )
    return decisions, unknown_ids, missing_ids


def call_batch(
    client: OpenAI,
    split: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt, candidate_map = make_prompt(candidates)
    digest = batch_digest(candidates)
    prompt_sha = sha256_text(SYSTEM_PROMPT + "\n" + prompt)
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                timeout=180,
                stream=False,
            )
            choice = response.choices[0]
            raw = str(choice.message.content or "")
            parsed = extract_json_object(raw)
            decisions, unknown_ids, missing_ids = parse_decisions(parsed, candidate_map)
            usage = getattr(response, "usage", None)
            return {
                "status": "success",
                "split": split,
                "batch_digest": digest,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_sha,
                "candidate_count": len(candidates),
                "decisions": decisions,
                "unknown_candidate_ids": unknown_ids,
                "missing_candidate_ids": missing_ids,
                "requested_model": MODEL,
                "returned_model": str(getattr(response, "model", "") or ""),
                "response_id": str(getattr(response, "id", "") or ""),
                "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "raw_content": raw,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at_utc": utc_now(),
            }
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(2**attempt, 12))
    assert last_error is not None
    return {
        "status": "error",
        "split": split,
        "batch_digest": digest,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha,
        "candidate_count": len(candidates),
        "requested_model": MODEL,
        "error_type": type(last_error).__name__,
        "error": str(last_error)[:1000],
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": utc_now(),
    }


def existing_successes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    successes: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if (
            record.get("status") == "success"
            and record.get("batch_digest")
            and record.get("prompt_version") == PROMPT_VERSION
            and record.get("returned_model") == MODEL
            and not (record.get("missing_candidate_ids") or [])
            and len(record.get("decisions") or [])
            == int(record.get("candidate_count") or -1)
        ):
            successes[str(record["batch_digest"])] = record
    return successes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing environment variable {API_KEY_ENV}")

    candidates = read_jsonl(DATA_DIR / f"{args.split}_candidates.jsonl")
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_sample[str(candidate["sample_id"])].append(candidate)
    batches: list[list[dict[str, Any]]] = []
    for sample_id in sorted(by_sample):
        sample_candidates = sorted(by_sample[sample_id], key=candidate_key)
        for start in range(0, len(sample_candidates), CHUNK_SIZE):
            batches.append(sample_candidates[start : start + CHUNK_SIZE])
    if args.limit_batches is not None:
        batches = batches[: args.limit_batches]

    run_path = RUN_DIR / args.split / "responses.jsonl"
    decision_path = OUTPUT_DIR / f"deepseek_triage_{args.split}_decisions.jsonl"
    manifest_path = RUN_DIR / args.split / "run_manifest.json"
    completed = existing_successes(run_path)
    pending = [batch for batch in batches if batch_digest(batch) not in completed]
    write_lock = threading.Lock()
    local_state = threading.local()

    def worker(batch: list[dict[str, Any]]) -> dict[str, Any]:
        client = getattr(local_state, "client", None)
        if client is None:
            client = OpenAI(api_key=api_key, base_url=BASE_URL)
            local_state.client = client
        result = call_batch(client, args.split, batch)
        append_jsonl(run_path, result, write_lock)
        return result

    if pending:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [executor.submit(worker, batch) for batch in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                print(
                    f"{args.split} {index}/{len(pending)} "
                    f"{result['status']} {result['batch_digest']}",
                    flush=True,
                )

    completed = existing_successes(run_path)
    all_decisions: list[dict[str, Any]] = []
    for batch in batches:
        record = completed.get(batch_digest(batch))
        if record is None:
            continue
        all_decisions.extend(record.get("decisions") or [])
    write_jsonl(decision_path, all_decisions)

    run_records = read_jsonl(run_path) if run_path.exists() else []
    accepted_records = [
        record
        for record in completed.values()
        if record.get("returned_model") == MODEL
    ]
    manifest = {
        "status": "complete" if len(accepted_records) == len(batches) else "incomplete",
        "split": args.split,
        "provider": "Beijing Institute of Technology GPUStack",
        "base_url": BASE_URL,
        "requested_model": MODEL,
        "returned_models": dict(
            Counter(str(record.get("returned_model") or "") for record in accepted_records)
        ),
        "api_key_source": f"environment variable {API_KEY_ENV}; value not stored",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT + "\n" + USER_TEMPLATE),
        "chunk_size": CHUNK_SIZE,
        "concurrency": CONCURRENCY,
        "total_candidates": len(candidates),
        "total_batches": len(batches),
        "successful_batches": len(accepted_records),
        "error_records": sum(1 for record in run_records if record.get("status") == "error"),
        "decision_count": len(all_decisions),
        "label_distribution": dict(Counter(row["label"] for row in all_decisions)),
        "missing_decisions_filled_as_unsupported": sum(
            len(record.get("missing_candidate_ids") or []) for record in accepted_records
        ),
        "prompt_tokens": sum(int(record.get("prompt_tokens") or 0) for record in accepted_records),
        "completion_tokens": sum(
            int(record.get("completion_tokens") or 0) for record in accepted_records
        ),
        "candidate_path": str(
            (DATA_DIR / f"{args.split}_candidates.jsonl").relative_to(ROOT)
        ),
        "candidate_sha256": sha256_file(DATA_DIR / f"{args.split}_candidates.jsonl"),
        "response_path": str(run_path.relative_to(ROOT)),
        "response_sha256": sha256_file(run_path) if run_path.exists() else "",
        "decision_path": str(decision_path.relative_to(ROOT)),
        "decision_sha256": sha256_file(decision_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "generated_at_utc": utc_now(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
