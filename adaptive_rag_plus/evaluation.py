"""Per-task metric synthesis for AdaptiveRAG+.

This module computes:
  - QA metrics (EM, F1, Acc) for trivia / nq / hotpotqa cells, mirroring
    :mod:`scripts.run_pipelines_4x3` so existing 3-way numbers stay
    consistent and reproducible.
  - ROUGE-L for class-D cells (gov_report / qmsum) using
    :class:`rouge_score.rouge_scorer.RougeScorer`.
  - Step / Time efficiency stats for every cell (including the
    LongBench summary column).

It also assembles the final 4-pipeline x 4-dataset-group table with
per-group, task-specific subcolumns (QA: EM/F1/Acc/Step/Time;
LongBench: ROUGE-L/Step/Time) and the longbench macro-average over
gov_report and qmsum.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .contracts import (
    DATASET_GROUP_SUBCOLS,
    DATASET_GROUPS,
    LONGBENCH_D_DATASETS,
    PIPELINES,
    QA_DATASETS,
    QA_SUBCOLS,
    D_SUBCOLS,
)


logger = logging.getLogger(__name__)


# ----------------- ROUGE-L (class D) ----------------- #

def _get_rouge_scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l_for_pairs(pairs: List[Tuple[str, List[str]]]) -> float:
    """Mean ROUGE-L F1 across (prediction, references) pairs.

    For each example, ROUGE-L is taken as the max F1 across reference
    answers (LongBench reports per-reference scoring this way).
    """
    if not pairs:
        return 0.0
    scorer = _get_rouge_scorer()
    total = 0.0
    n = 0
    for pred, refs in pairs:
        best = 0.0
        for ref in refs:
            r = scorer.score(ref, pred)["rougeL"].fmeasure
            if r > best:
                best = r
        total += best
        n += 1
    return total / n if n else 0.0


def evaluate_d_predictions_jsonl(predictions_path: Path) -> Dict[str, float]:
    """Compute ROUGE-L (and aggregate efficiency) for a global pipeline output.

    Expects ``predictions.jsonl`` from
    :func:`adaptive_rag_plus.global_inference.run_global_pipeline_for_dataset`
    and looks for a sibling ``traces.jsonl`` for efficiency stats.
    """
    pairs: List[Tuple[str, List[str]]] = []
    n = 0
    with Path(predictions_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append((row.get("prediction", ""), row.get("references", []) or []))
            n += 1
    rouge_l = compute_rouge_l_for_pairs(pairs)

    traces_path = Path(predictions_path).parent / "traces.jsonl"
    avg_step, avg_time = _aggregate_traces(traces_path) if traces_path.exists() else (None, None)

    return {
        "ROUGE-L": rouge_l,
        "Step": avg_step if avg_step is not None else 1.0,
        "Time": avg_time,
        "count": n,
    }


def _aggregate_traces(traces_path: Path) -> Tuple[Optional[float], Optional[float]]:
    n = 0
    step_sum = 0.0
    time_sum = 0.0
    with Path(traces_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            step_sum += float(t.get("step_count", 0.0))
            time_sum += float(t.get("latency_seconds", 0.0))
            n += 1
    if not n:
        return None, None
    return step_sum / n, time_sum / n


# ----------------- QA helpers (mirror run_pipelines_4x3) ----------------- #

_NORM_REGEX_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize_qa(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _NORM_REGEX_ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _qa_answer_extract(potentially_cot: str) -> str:
    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]
    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    m = cot_regex.match(potentially_cot)
    if m:
        out = m.group(1)
        if out.endswith("."):
            out = out[:-1]
        return out
    return potentially_cot


def _calc_qa_acc(prediction: str, ground_truths: List[str]) -> int:
    p = _normalize_qa(prediction)
    for gt in ground_truths:
        if _normalize_qa(gt) in p:
            return 1
    return 0


# ----------------- Aggregation ----------------- #

def aggregate_longbench(
    per_d_metrics: Dict[str, Dict[str, Optional[float]]],
) -> Dict[str, Optional[float]]:
    """Macro-average ROUGE-L / Step / Time across gov_report and qmsum."""
    keys = ("ROUGE-L", "Step", "Time")
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [per_d_metrics[d].get(k) for d in LONGBENCH_D_DATASETS if d in per_d_metrics]
        vals = [v for v in vals if v is not None]
        out[k] = (sum(vals) / len(vals)) if vals else None
    return out


# ----------------- Table emitters ----------------- #

def _fmt(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def emit_grouped_csv(
    metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    out_path: Path,
) -> None:
    """Emit a grouped CSV with per-dataset-group, task-specific subcolumns."""
    out_path = Path(out_path)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        header1 = ["pipeline"]
        header2 = [""]
        for ds in DATASET_GROUPS:
            subs = DATASET_GROUP_SUBCOLS[ds]
            header1.extend([ds] + [""] * (len(subs) - 1))
            header2.extend(list(subs))
        w.writerow(header1)
        w.writerow(header2)
        for pipe in PIPELINES:
            row = [pipe]
            for ds in DATASET_GROUPS:
                subs = DATASET_GROUP_SUBCOLS[ds]
                cell = metrics.get(pipe, {}).get(ds, {}) or {}
                for s in subs:
                    row.append(_fmt(cell.get(s)))
            w.writerow(row)


def emit_grouped_md(
    metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    out_path: Path,
) -> None:
    """Emit a grouped Markdown table mirroring the CSV structure."""
    lines: List[str] = []
    head_cells = ["pipeline"]
    for ds in DATASET_GROUPS:
        subs = DATASET_GROUP_SUBCOLS[ds]
        head_cells.append(f"{ds} ({'/'.join(subs)})")
    lines.append("| " + " | ".join(head_cells) + " |")
    lines.append("|" + "---|" * len(head_cells))
    for pipe in PIPELINES:
        row_cells = [pipe]
        for ds in DATASET_GROUPS:
            subs = DATASET_GROUP_SUBCOLS[ds]
            cell = metrics.get(pipe, {}).get(ds, {}) or {}
            row_cells.append(" / ".join(_fmt(cell.get(s)) for s in subs))
        lines.append("| " + " | ".join(row_cells) + " |")
    Path(out_path).write_text("\n".join(lines) + "\n")


def emit_provenance(
    metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    extra: dict,
    out_path: Path,
) -> None:
    payload = {"metrics": metrics, "extra": extra}
    Path(out_path).write_text(json.dumps(payload, indent=2, default=str))
