"""Data, routing, and metric contracts for AdaptiveRAG+.

This module is the single source of truth for the AdaptiveRAG+ schema.

1. Routing label space (A/B/C/D)
   A = no retrieval        (paper: TriviaQA)
   B = single-step         (paper: Natural Questions)
   C = multi-step          (paper: HotpotQA)
   D = global retrieval    (this work: LongBench gov_report and qmsum)

2. Class-D labelling policy (deterministic):
   any record whose ``dataset`` field is ``gov_report`` or ``qmsum`` is
   labelled ``D``. Label is derived from dataset identity, not query text.

3. Per-task metric schema:
   - QA datasets (A/B/C columns): EM, F1, Acc, Step, Time
   - LongBench (D column):        ROUGE-L, Step, Time
   The orchestrator emits task-specific column groups so QA cells and the D
   cell never share a column.

4. Aggregation rules for the ``longbench`` dataset group:
   - ROUGE-L(longbench) = mean(ROUGE-L_gov_report, ROUGE-L_qmsum)
   - Step(longbench)    = mean(Step_gov_report,    Step_qmsum)
   - Time(longbench)    = mean(Time_gov_report,    Time_qmsum)

All consumers (ingestion, indexing, inference, evaluation, orchestration)
import from this module so that downstream changes happen in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


ROUTING_LABELS = ("A", "B", "C", "D")

ROUTING_LABEL_TO_PIPELINE: Dict[str, str] = {
    "A": "nor_qa",
    "B": "oner_qa",
    "C": "ircot_qa",
    "D": "global_qa",
}

PIPELINE_TO_ROUTING_LABEL: Dict[str, str] = {v: k for k, v in ROUTING_LABEL_TO_PIPELINE.items()}


QA_DATASETS = ("trivia", "nq", "hotpotqa")
LONGBENCH_D_DATASETS = ("gov_report", "qmsum")
DATASET_GROUPS = ("trivia", "nq", "hotpotqa", "longbench")


PIPELINES = ("nor_qa", "oner_qa", "ircot_qa", "adaptive-rag", "adaptive-rag+")


QA_SUBCOLS = ("EM", "F1", "Acc", "Step", "Time")
D_SUBCOLS = ("ROUGE-L", "Step", "Time")

DATASET_GROUP_SUBCOLS: Dict[str, tuple] = {
    "trivia":    QA_SUBCOLS,
    "nq":        QA_SUBCOLS,
    "hotpotqa":  QA_SUBCOLS,
    "longbench": D_SUBCOLS,
}


@dataclass
class NormalizedRecord:
    """Common record format shared across A/B/C/D pipelines.

    Fields:
      qid              stable per-record id (LongBench `_id` for D, dataset id for A/B/C)
      query            user query text (LongBench `input` for D, question for A/B/C)
      context          full source context (LongBench `context` for D, optional for A/B/C)
      references       list of gold reference strings (LongBench `answers` for D)
      dataset          source dataset name (`trivia`, `nq`, `hotpotqa`, `gov_report`, `qmsum`)
      routing_label    one of ROUTING_LABELS; for D this is forced to "D" by policy
      target_metric_type
                       "qa" for A/B/C datasets, "rouge" for class-D datasets
      length           optional original length annotation (LongBench `length`)
      language         optional language tag from LongBench
      all_classes      optional pass-through from LongBench
      raw              optional pass-through of original source record
    """

    qid: str
    query: str
    context: str
    references: List[str]
    dataset: str
    routing_label: str
    target_metric_type: str
    length: Optional[int] = None
    language: Optional[str] = None
    all_classes: Optional[List[str]] = None
    raw: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EfficiencyTrace:
    """Per-query efficiency telemetry emitted by every pipeline.

    Fields are deliberately uniform so downstream reporting can compare
    pipelines fairly on Step / Time across A/B/C/D.
    """

    qid: str
    pipeline: str
    routed_label: Optional[str] = None
    retrieval_calls: int = 0
    step_count: float = 0.0
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredictionRecord:
    """Final answer record (one per query) for any pipeline.

    Stored alongside an EfficiencyTrace.
    """

    qid: str
    dataset: str
    pipeline: str
    routed_label: Optional[str]
    prediction: str
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def label_for_dataset(dataset: str) -> str:
    """Deterministic class-D labelling policy.

    Any LongBench `gov_report` / `qmsum` record is labelled D. All other
    datasets fall back to None and inherit the existing 3-way classifier
    output.
    """
    if dataset in LONGBENCH_D_DATASETS:
        return "D"
    return None


def metric_type_for_dataset(dataset: str) -> str:
    if dataset in LONGBENCH_D_DATASETS:
        return "rouge"
    if dataset in QA_DATASETS:
        return "qa"
    raise ValueError(f"Unknown dataset for metric routing: {dataset}")


def subcols_for_group(group: str) -> tuple:
    if group not in DATASET_GROUP_SUBCOLS:
        raise KeyError(f"Unknown dataset group: {group}")
    return DATASET_GROUP_SUBCOLS[group]
