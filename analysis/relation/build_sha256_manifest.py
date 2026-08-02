from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "paper_revision" / "relation_improvement"
MANIFEST = PACKAGE / "sha256_manifest.txt"
INCLUDED = [
    PACKAGE / "analysis",
    PACKAGE / "configs",
    PACKAGE / "data",
    PACKAGE / "human_revalidation",
    PACKAGE / "llm_runs",
    PACKAGE / "models",
    PACKAGE / "outputs",
    PACKAGE / "predictions",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    files: set[Path] = {PACKAGE / "EXPERIMENT_REPORT_ZH.md"}
    for directory in INCLUDED:
        if directory.exists():
            files.update(path for path in directory.rglob("*") if path.is_file())
    files.discard(MANIFEST)
    lines = [
        f"{sha256(path)}\t{path.stat().st_size}\t{path.relative_to(ROOT).as_posix()}"
        for path in sorted(files)
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"files={len(lines)}")
    print(MANIFEST)


if __name__ == "__main__":
    main()
