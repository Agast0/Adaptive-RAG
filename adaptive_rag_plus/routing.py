"""4-way (A/B/C/D) routing data preparation and dispatcher.

Responsibilities:
  - Build the augmented training / validation / predict corpora that
    extend the existing 3-way training corpus with class-D examples drawn
    from LongBench gov_report + qmsum (deterministic labelling per
    :func:`adaptive_rag_plus.contracts.label_for_dataset`).
  - Provide a route dispatcher that maps a 4-way classifier prediction
    JSON (``dict_id_pred_results.json``) to per-dataset answer files
    produced by either the existing A/B/C pipelines or the new D pipeline.

Split hygiene:
  * Class-D records are partitioned by ``_id`` into disjoint
    train / valid / test slices using a fixed random seed; the test
    slice is the only one ever used for reporting metrics.
  * Existing A/B/C train/valid/predict files are not modified; the
    augmented files are written to a ``..._plus`` sibling directory so
    the original 3-way artifacts remain reproducible.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .contracts import (
    LONGBENCH_D_DATASETS,
    NormalizedRecord,
    ROUTING_LABELS,
    label_for_dataset,
)
from .longbench import load_normalized as load_longbench_normalized


logger = logging.getLogger(__name__)


DEFAULT_BASE_DATA_DIR = Path("classifier") / "data" / "musique_hotpot_wiki2_nq_tqa_sqd"
DEFAULT_PLUS_DATA_DIR = Path("classifier") / "data" / "musique_hotpot_wiki2_nq_tqa_sqd_plus"
DEFAULT_LLM_TAG = "gpt"  # which sub-tree of the existing data to extend


# ----------------- Class-D split policy ----------------- #

def split_d_records(
    records: List[NormalizedRecord],
    *,
    test_size: int,
    train_size: int,
    valid_size: int,
    seed: int = 1234,
) -> Tuple[List[NormalizedRecord], List[NormalizedRecord], List[NormalizedRecord]]:
    """Partition D records into disjoint (train, valid, test) by _id.

    Test slice is taken first (so test sample identity is stable across
    train/valid resizing); train/valid are sampled from the remainder.
    """
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r.qid)
    rng.shuffle(pool)

    test = pool[:test_size]
    rest = pool[test_size:]
    train = rest[:train_size]
    valid = rest[train_size:train_size + valid_size]

    used = {r.qid for r in test} | {r.qid for r in train} | {r.qid for r in valid}
    assert len(used) == len(test) + len(train) + len(valid), "split_d_records produced overlapping ids"
    return train, valid, test


def to_classifier_record(rec: NormalizedRecord) -> dict:
    """Convert a normalized D record to the existing classifier row schema.

    The existing classifier files use:
      {answer, answer_description, dataset_name, id, question}
    """
    return {
        "answer": rec.routing_label,
        "answer_description": "global",
        "dataset_name": rec.dataset,
        "id": rec.qid,
        "question": rec.query,
    }


# ----------------- Augmented training/predict file builders ----------------- #

def _read_json_list(path: Path) -> list:
    return json.loads(Path(path).read_text())


def _write_json_list(rows: list, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=4))
    return path


def build_plus_corpus(
    *,
    base_dir: Path = DEFAULT_BASE_DATA_DIR,
    plus_dir: Path = DEFAULT_PLUS_DATA_DIR,
    llm_tag: str = DEFAULT_LLM_TAG,
    train_per_dataset: int = 60,
    valid_per_dataset: int = 20,
    test_per_dataset: int = 40,
    seed: int = 1234,
    longbench_cache_dir: Optional[Path] = None,
) -> dict:
    """Produce the 4-way classifier corpus by appending D rows to existing data.

    The existing files used by the 3-way pipeline are:
      ``base_dir/<llm_tag>/binary_silver/train.json``
      ``base_dir/<llm_tag>/silver/valid.json``      (when present)
      ``base_dir/predict.json``                     (test queries to label)

    The augmented versions are written under ``plus_dir`` with the same
    layout. Any missing source file is treated as an empty list.

    Args:
        train_per_dataset / valid_per_dataset / test_per_dataset: counts
            applied to *each* D dataset (gov_report, qmsum) so the final
            class-D contribution is roughly 2x these numbers.

    Returns a dict describing the produced files and split sizes.
    """
    base_dir = Path(base_dir)
    plus_dir = Path(plus_dir)

    base_train = base_dir / llm_tag / "binary_silver" / "train.json"
    base_valid = base_dir / llm_tag / "silver" / "valid.json"  # may be absent
    base_predict = base_dir / "predict.json"

    plus_train = plus_dir / llm_tag / "binary_silver" / "train.json"
    plus_valid = plus_dir / llm_tag / "silver" / "valid.json"
    plus_predict = plus_dir / "predict.json"

    train_rows = _read_json_list(base_train) if base_train.exists() else []
    valid_rows = _read_json_list(base_valid) if base_valid.exists() else []
    predict_rows = _read_json_list(base_predict) if base_predict.exists() else []

    info = {
        "base_train": str(base_train),
        "base_valid": str(base_valid),
        "base_predict": str(base_predict),
        "plus_train": str(plus_train),
        "plus_valid": str(plus_valid),
        "plus_predict": str(plus_predict),
        "added_train": 0,
        "added_valid": 0,
        "added_test": 0,
        "splits_per_dataset": {},
        "test_qids_per_dataset": {},
    }

    for ds in LONGBENCH_D_DATASETS:
        recs = load_longbench_normalized(
            ds,
            cache_dir=longbench_cache_dir or Path("raw_data") / "longbench",
        )
        for r in recs:
            assert label_for_dataset(r.dataset) == "D", \
                f"label policy invariant violated: {r.dataset}/{r.qid}"
        d_train, d_valid, d_test = split_d_records(
            recs,
            test_size=test_per_dataset,
            train_size=train_per_dataset,
            valid_size=valid_per_dataset,
            seed=seed,
        )

        train_rows.extend(to_classifier_record(r) for r in d_train)
        valid_rows.extend(to_classifier_record(r) for r in d_valid)
        predict_rows.extend(to_classifier_record(r) for r in d_test)

        info["added_train"] += len(d_train)
        info["added_valid"] += len(d_valid)
        info["added_test"] += len(d_test)
        info["splits_per_dataset"][ds] = {
            "train": len(d_train),
            "valid": len(d_valid),
            "test": len(d_test),
        }
        info["test_qids_per_dataset"][ds] = [r.qid for r in d_test]

    _write_json_list(train_rows, plus_train)
    _write_json_list(valid_rows, plus_valid)
    _write_json_list(predict_rows, plus_predict)

    info["plus_train_total"] = len(train_rows)
    info["plus_valid_total"] = len(valid_rows)
    info["plus_predict_total"] = len(predict_rows)

    logger.info(
        "Built 4-way corpus: train=%d valid=%d predict=%d (added D: train=%d valid=%d test=%d)",
        info["plus_train_total"], info["plus_valid_total"], info["plus_predict_total"],
        info["added_train"], info["added_valid"], info["added_test"],
    )
    return info


# ----------------- 4-way routing dispatcher ----------------- #

def load_classifier_predictions(predict_results_file: Path) -> Dict[str, str]:
    """Read a classifier ``dict_id_pred_results.json`` and return qid -> label."""
    raw = json.loads(Path(predict_results_file).read_text())
    out: Dict[str, str] = {}
    for qid, info in raw.items():
        pred = info.get("prediction") if isinstance(info, dict) else info
        if pred not in ROUTING_LABELS:
            logger.warning("dropping unknown classifier label %r for qid=%s", pred, qid)
            continue
        out[qid] = pred
    return out


def routing_summary(qid_to_label: Dict[str, str]) -> Dict[str, int]:
    counts = {lab: 0 for lab in ROUTING_LABELS}
    for v in qid_to_label.values():
        counts[v] = counts.get(v, 0) + 1
    return counts
