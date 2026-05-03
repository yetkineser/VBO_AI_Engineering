"""
scorer.py — analyze test_*.jsonl files and emit a side-by-side comparison.

Usage:
    python scorer.py                      # picks up everything in ./results/
    python scorer.py results/*.jsonl      # explicit list

Metrics computed per model:
  - smoke_ok           : 6 sorudan kaçında non-empty cevap geldi (no tool errors)
  - cite_rate          : "[Source: ...]" pattern'ini içeren cevap oranı
  - hallu_cite         : real file_name'leri tanıyan listeye uymayan citation sayısı
  - lang_consistency   : ascii-only cevap oranı (yanlış dil halüsinasyonu yakalar)
  - avg_latency_s      : tüm cevapların ortalama süresi
  - memory_ok          : 3 memory turunun hepsi error'sız bitti mi
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REAL_FILES = {
    "employee_handbook.docx",
    "it_security.pdf",
    "leave_policy.docx",
    "offboarding_checklist.txt",
    "onboarding_guide.docx",
    "performance_review.docx",
    "recruitment_policy.docx",
    "travel_expense_policy.docx",
}

CITE_RE = re.compile(r"\[Source:\s*([^\]\n]+?)\s*\]", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_ascii_clean(s: str) -> bool:
    """Heuristic: cevabın >=85%'i ASCII range'inde mi? (Tay/Cyrillic halüsinasyonu yakalar)"""
    if not s:
        return True
    ascii_count = sum(1 for c in s if ord(c) < 128)
    return ascii_count / len(s) >= 0.85


def score_one(records: list[dict]) -> dict:
    smoke = [r for r in records if r["section"] == "smoke"]
    memory = [r for r in records if r["section"] == "memory"]

    smoke_ok = sum(1 for r in smoke if r.get("ok") and r.get("answer"))
    memory_ok = sum(1 for r in memory if r.get("ok"))

    cited = 0
    hallu_cite = 0
    lang_clean = 0
    for r in smoke + memory:
        ans = r.get("answer", "")
        cites = CITE_RE.findall(ans)
        if cites:
            cited += 1
            for cite in cites:
                head = cite.split()[0].strip()  # strip optional "p.4"
                if head not in REAL_FILES:
                    hallu_cite += 1
        if is_ascii_clean(ans):
            lang_clean += 1

    n = len(smoke) + len(memory)
    latencies = [r.get("latency_s", 0) for r in smoke + memory]
    return {
        "smoke_ok": f"{smoke_ok}/{len(smoke)}",
        "memory_ok": f"{memory_ok}/{len(memory)}",
        "cite_rate": f"{cited}/{n}",
        "hallu_cite": hallu_cite,
        "lang_consistency": f"{lang_clean}/{n}",
        "avg_latency_s": round(sum(latencies) / max(1, len(latencies)), 2),
        "max_latency_s": round(max(latencies, default=0), 2),
    }


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(Path(__file__).parent.joinpath("results").glob("test_*.jsonl"))

    if not paths:
        print("No test_*.jsonl files found. Run `python main.py test --model X` first.")
        sys.exit(1)

    rows = {}
    for p in paths:
        model = p.stem.replace("test_", "")
        rows[model] = score_one(load_jsonl(p))

    cols = ["smoke_ok", "memory_ok", "cite_rate", "hallu_cite", "lang_consistency", "avg_latency_s", "max_latency_s"]
    width = max(len(m) for m in rows)
    header = f"{'metric':22s} | " + " | ".join(f"{m:>{width}s}" for m in rows)
    print(header)
    print("-" * len(header))
    for c in cols:
        line = f"{c:22s} | " + " | ".join(f"{str(rows[m][c]):>{width}s}" for m in rows)
        print(line)


if __name__ == "__main__":
    main()
