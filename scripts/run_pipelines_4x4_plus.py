#!/usr/bin/env python
"""AdaptiveRAG+ orchestrator: 5 pipelines x 4 dataset groups (GPT).

Pipelines (rows):
    nor_qa, oner_qa, ircot_qa, adaptive-rag, adaptive-rag+

  * `adaptive-rag`  uses the existing 3-way (A/B/C) classifier and routes
    every query to one of the existing baselines. For QA columns this
    reuses the original 3-way adaptive output under
    `predictions/classifier/...`. For the LongBench column we run the
    same 3-way classifier on the D test slice (it has no `D` label, so
    every D query is routed to A/B/C) and read the corresponding D
    baseline output.
  * `adaptive-rag+` uses the new 4-way (A/B/C/D) classifier built in this
    work and adds the global retrieval pipeline for class D.

Dataset groups (cols):
    trivia, nq, hotpotqa, longbench
    (`longbench` is a macro-average over LongBench gov_report + qmsum;
     each baseline pipeline is evaluated on both D datasets independently
     before aggregation.)

Subcolumns are task-specific:
    QA datasets (trivia/nq/hotpotqa) : EM | F1 | Acc | Step | Time
    longbench   (class-D)            : ROUGE-L | Step | Time

Modes:
    --smoke-test   tiny D test slice (default 4 records per D dataset),
                   tiny classifier sweep, reuses 3-way QA artifacts.
    --full-run     full LongBench D test slice, reporting classifier
                   sweep, runs missing 3-way QA artifacts.

Prerequisite (indexing is now decoupled and resumable):
    python scripts/build_longbench_index.py
    # default indexes BOTH gov_report and qmsum (200 records each).
    # Re-running is safe: it resumes from the last successful checkpoint
    # if a previous build was interrupted (e.g. by an OpenAI rate limit).
This orchestrator now only LOADS the prebuilt indexes; if either
gov_report or qmsum index is missing, Phase 0 aborts with a clear
remediation pointer.

Outputs (under --out-dir, default ./reports/<timestamp>_plus_<mode>):
    pipeline_results_grouped.csv
    pipeline_results_grouped.md
    pipeline_results_provenance.json
    pipeline_results_notes.md
    smoke_test_report.md (smoke-test only)
    classifier_sweep_summary.json
    classifier_corpus_info.json
    coverage.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=_REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass

from adaptive_rag_plus import (  # noqa: E402
    baselines_d,
    contracts,
    evaluation,
    global_index,
    global_inference,
    longbench,
    routing,
)


REPO_ROOT = _REPO_ROOT


# ----------------- Constants ----------------- #

QA_DATASETS = list(contracts.QA_DATASETS)            # trivia, nq, hotpotqa
D_DATASETS = list(contracts.LONGBENCH_D_DATASETS)    # gov_report, qmsum
DATASET_GROUPS = list(contracts.DATASET_GROUPS)      # + 'longbench'
PIPELINES = list(contracts.PIPELINES)                # nor_qa, oner_qa, ircot_qa, adaptive-rag+

GPT_BM25 = {"oner_qa": 6, "ircot_qa": 3}
PROMPT_SET = 1
DISTRACTOR_COUNT = 1


# ----------------- Logging ----------------- #

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def section(msg: str) -> None:
    bar = "=" * (len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


# ----------------- 3-way QA artifact paths (reused) ----------------- #

def base_pred_dir(pipeline: str, dataset: str) -> Path:
    if pipeline == "nor_qa":
        sub = f"nor_qa_gpt_{dataset}____prompt_set_1"
    elif pipeline == "oner_qa":
        sub = (
            f"oner_qa_gpt_{dataset}____prompt_set_1"
            f"___bm25_retrieval_count__{GPT_BM25['oner_qa']}"
            f"___distractor_count__{DISTRACTOR_COUNT}"
        )
    elif pipeline == "ircot_qa":
        sub = (
            f"ircot_qa_gpt_{dataset}____prompt_set_1"
            f"___bm25_retrieval_count__{GPT_BM25['ircot_qa']}"
            f"___distractor_count__{DISTRACTOR_COUNT}"
        )
    else:
        raise ValueError(pipeline)
    return REPO_ROOT / "predictions" / "test" / sub


def base_eval_path(pipeline: str, dataset: str) -> Path:
    return base_pred_dir(pipeline, dataset) / f"evaluation_metrics__{dataset}_to_{dataset}__test_subsampled.json"


def base_pred_path(pipeline: str, dataset: str) -> Path:
    return base_pred_dir(pipeline, dataset) / f"prediction__{dataset}_to_{dataset}__test_subsampled.json"


def base_time_path(pipeline: str, dataset: str) -> Path:
    return base_pred_dir(pipeline, dataset) / f"prediction__{dataset}_to_{dataset}__test_subsampled_time_taken.txt"


def base_step_path(pipeline: str, dataset: str) -> Optional[Path]:
    if pipeline != "ircot_qa":
        return None
    return base_pred_dir(pipeline, dataset) / "stepNum.json"


def processed_test_path(dataset: str) -> Path:
    return REPO_ROOT / "processed_data" / dataset / "test_subsampled.jsonl"


# ----------------- D baseline artifact paths (new) ----------------- #

def d_pipeline_dir(pipeline: str, dataset: str) -> Path:
    return REPO_ROOT / "predictions" / "test" / f"{pipeline}_gpt_{dataset}"


# ----------------- QA Acc helpers (mirror 4x3) ----------------- #

_NORM_REGEX_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize_qa(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _NORM_REGEX_ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _qa_extract(potentially_cot: str) -> str:
    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]
    m = re.match(".* answer is:? (.*)\\.?", potentially_cot)
    if m:
        out = m.group(1)
        if out.endswith("."):
            out = out[:-1]
        return out
    return potentially_cot


def _qa_acc(prediction: str, gts: List[str]) -> int:
    p = _normalize_qa(prediction)
    return int(any(_normalize_qa(g) in p for g in gts))


def _load_id_to_gt(processed_path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with processed_path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["question_id"]] = d["answers_objects"][0]["spans"]
    return out


def _qa_acc_for_predictions(prediction_file: Path, dataset: str) -> Tuple[float, int]:
    if dataset == "hotpotqa":
        return _qa_acc_hotpot_official(prediction_file)
    return _qa_acc_substring(prediction_file, dataset)


def _qa_acc_substring(prediction_file: Path, dataset: str) -> Tuple[float, int]:
    id_to_pred = json.loads(prediction_file.read_text())
    id_to_gt = _load_id_to_gt(processed_test_path(dataset))
    correct = 0
    total = 0
    for qid, gt in id_to_gt.items():
        if qid not in id_to_pred:
            continue
        pred = id_to_pred[qid]
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        pred = _qa_extract(str(pred))
        correct += _qa_acc(pred, gt)
        total += 1
    return (correct / total if total else 0.0), total


def _qa_acc_hotpot_official(prediction_file: Path) -> Tuple[float, int]:
    id_to_pred = json.loads(prediction_file.read_text())
    qids = list(id_to_pred.keys())
    raw = json.loads((REPO_ROOT / "raw_data" / "hotpotqa" / "hotpot_dev_distractor_v1.json").read_text())
    filtered = [d for d in raw if d["_id"] in set(qids)]

    tmp_dir = REPO_ROOT / ".temp"
    tmp_dir.mkdir(exist_ok=True)
    gt_path = tmp_dir / uuid.uuid4().hex
    pr_path = tmp_dir / uuid.uuid4().hex
    out_path = tmp_dir / uuid.uuid4().hex
    gt_path.write_text(json.dumps(filtered))

    answer_dict = {}
    for qid, p in id_to_pred.items():
        if isinstance(p, list):
            p = " ".join(str(x) for x in p) if len(p) > 1 else str(p[0]) if p else ""
        answer_dict[qid] = str(p)
    pr_path.write_text(json.dumps({
        "answer": answer_dict,
        "sp": {qid: [["", 0]] for qid in id_to_pred},
    }))

    rel_gt = os.path.join("..", "..", str(gt_path.relative_to(REPO_ROOT)))
    rel_pr = os.path.join("..", "..", str(pr_path.relative_to(REPO_ROOT)))
    rel_out = os.path.join("..", "..", str(out_path.relative_to(REPO_ROOT)))
    cmd = (
        f"cd official_evaluation/hotpotqa ; "
        f"python hotpot_evaluate_v1.py {rel_pr} {rel_gt} > {rel_out}"
    )
    rc = subprocess.call(cmd, shell=True, cwd=str(REPO_ROOT))
    if rc != 0 or not out_path.exists():
        gt_path.unlink(missing_ok=True)
        pr_path.unlink(missing_ok=True)
        raise RuntimeError("Official hotpotqa eval failed")
    metrics = eval(out_path.read_text().strip())
    gt_path.unlink(missing_ok=True)
    pr_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return float(metrics["acc"]), len(id_to_pred)


# ----------------- Phase: D baselines + global pipeline ----------------- #

def run_all_d_pipelines(
    *,
    test_per_dataset: int,
    seed: int,
    embed_model: str,
    gen_model: str,
    chunk_chars: int,
    chunk_overlap: int,
    num_clusters: int,
    summary_max_tokens: int,
    gen_max_tokens: int,
    n_ircot_steps: int,
    skip_existing: bool,
    inter_request_pause: float,
) -> Dict[str, Dict[str, Path]]:
    """Run nor/oner/ircot/global_qa on each D dataset's test slice.

    Returns a {pipeline: {dataset: out_dir}} mapping for downstream
    evaluation.
    """
    section("Class-D pipelines on LongBench (gov_report, qmsum)")
    plus_corpus_info_path = REPO_ROOT / "classifier" / "data" / "musique_hotpot_wiki2_nq_tqa_sqd_plus" / "corpus_info.json"
    if not plus_corpus_info_path.exists():
        raise RuntimeError(
            "Plus corpus info missing; run the classifier-corpus step first."
        )
    info = json.loads(plus_corpus_info_path.read_text())
    test_qids = info["test_qids_per_dataset"]

    d_records_by_dataset: Dict[str, List[contracts.NormalizedRecord]] = {}
    for ds in D_DATASETS:
        all_recs = longbench.load_normalized(ds)
        wanted = set(test_qids.get(ds, []))
        recs = [r for r in all_recs if r.qid in wanted]
        recs = recs[:test_per_dataset] if test_per_dataset else recs
        d_records_by_dataset[ds] = recs
        log(f"  D test records selected for {ds}: {len(recs)}")

    out_paths: Dict[str, Dict[str, Path]] = {p: {} for p in ("nor_qa", "oner_qa", "ircot_qa", "global_qa")}

    def _has_complete_output(out_dir: Path, expected_n: int) -> bool:
        preds = out_dir / "predictions.jsonl"
        traces = out_dir / "traces.jsonl"
        if not (preds.exists() and traces.exists()):
            return False
        try:
            n_p = sum(1 for _ in preds.open())
            n_t = sum(1 for _ in traces.open())
        except FileNotFoundError:
            return False
        return n_p == expected_n and n_t == expected_n and expected_n > 0

    for ds in D_DATASETS:
        records = d_records_by_dataset[ds]
        for pipeline in ("nor_qa", "oner_qa", "ircot_qa"):
            out_dir = d_pipeline_dir(pipeline, ds)
            preds_file = out_dir / "predictions.jsonl"
            if skip_existing and _has_complete_output(out_dir, len(records)):
                log(f"  reuse: {pipeline} on {ds} ({preds_file.relative_to(REPO_ROOT)})")
            else:
                log(f"  run  : {pipeline} on {ds} -> {out_dir.relative_to(REPO_ROOT)}")
                baselines_d.run_baseline_for_dataset(
                    pipeline, records, out_dir,
                    embed_model=embed_model, gen_model=gen_model,
                    gen_max_tokens=gen_max_tokens,
                    chunk_chars=chunk_chars, chunk_overlap=chunk_overlap,
                    n_steps=n_ircot_steps,
                )
            out_paths[pipeline][ds] = out_dir

    for ds in D_DATASETS:
        records = d_records_by_dataset[ds]
        # Indexing is no longer performed inline. The index must already
        # have been built (and any retries already absorbed) by:
        #   python scripts/build_longbench_index.py [--dataset <ds>]
        # We only LOAD the cached, fully-complete index here.
        if not global_index.index_exists(ds):
            raise RuntimeError(
                f"Global index for {ds!r} not found. Build it first with:\n"
                f"    python scripts/build_longbench_index.py --dataset {ds}\n"
                f"(default invocation indexes both gov_report and qmsum)."
            )
        index = global_index.GlobalIndex.load(ds)
        log(f"  loaded prebuilt index for {ds}: chunks={index.manifest.n_chunks} "
            f"clusters={index.manifest.n_clusters}")
        # Sanity-check the index covers every test record's qid.
        index_qids = {c.get("source_qid") for c in index.chunks}
        missing = [r.qid for r in records if r.qid not in index_qids]
        if missing:
            raise RuntimeError(
                f"Prebuilt index for {ds!r} is missing chunks for {len(missing)} test "
                f"record(s) (e.g. {missing[:3]}). Re-run scripts/build_longbench_index.py "
                f"so that every test qid is covered (defaults index all 200 records)."
            )
        out_dir = d_pipeline_dir("global_qa", ds)
        preds_file = out_dir / "predictions.jsonl"
        if skip_existing and _has_complete_output(out_dir, len(records)):
            log(f"  reuse: global_qa on {ds} ({preds_file.relative_to(REPO_ROOT)})")
        else:
            log(f"  run  : global_qa on {ds} -> {out_dir.relative_to(REPO_ROOT)}")
            global_inference.run_global_pipeline_for_dataset(
                records, index, out_dir,
                embed_model=embed_model, gen_model=gen_model,
                gen_max_tokens=gen_max_tokens,
            )
        out_paths["global_qa"][ds] = out_dir

    return out_paths


# ----------------- Phase: classifier 4-way sweep ----------------- #

def classifier_sweep_plus(
    *,
    epochs: List[int],
    base_model: str = "t5-small",
    batch: int = 8,
    sweep_root_name: str = "sweep_plus",
    plus_data_dir: str = "musique_hotpot_wiki2_nq_tqa_sqd_plus",
    llm_tag: str = "gpt",
) -> Tuple[Path, Path, dict]:
    """Sweep epochs of the 4-way classifier and return (best_dir, predict_file, summary)."""
    section(f"4-way classifier sweep epochs={epochs} base={base_model}")
    classifier_dir = REPO_ROOT / "classifier"
    sweep_root = classifier_dir / "outputs" / plus_data_dir / "model" / base_model / llm_tag / sweep_root_name
    sweep_root.mkdir(parents=True, exist_ok=True)
    sweep_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    sweep_dir = sweep_root / sweep_id
    sweep_dir.mkdir()

    train_file = f"./data/{plus_data_dir}/{llm_tag}/binary_silver/train.json"
    valid_file = f"./data/{plus_data_dir}/{llm_tag}/silver/valid.json"
    predict_file = f"./data/{plus_data_dir}/predict.json"

    summary = {"sweep_dir": str(sweep_dir.relative_to(REPO_ROOT)), "runs": []}
    for ep in epochs:
        ep_out = sweep_dir / f"epoch_{ep}"
        ep_out.mkdir(parents=True, exist_ok=True)
        log(f"  >>> training epoch={ep} -> {ep_out.relative_to(REPO_ROOT)}")
        t0 = time.time()
        run([
            "python", "run_classifier_plus.py",
            "--model_name_or_path", base_model,
            "--train_file", train_file,
            "--question_column", "question",
            "--answer_column", "answer",
            "--learning_rate", "3e-5",
            "--max_seq_length", "384",
            "--doc_stride", "128",
            "--per_device_train_batch_size", str(batch),
            "--output_dir", str(ep_out),
            "--overwrite_cache",
            "--train_column", "train",
            "--do_train",
            "--num_train_epochs", str(ep),
        ], cwd=classifier_dir)
        train_secs = time.time() - t0

        valid_dir = ep_out / "valid"
        valid_dir.mkdir(exist_ok=True)
        run([
            "python", "run_classifier_plus.py",
            "--model_name_or_path", str(ep_out),
            "--validation_file", valid_file,
            "--question_column", "question",
            "--answer_column", "answer",
            "--max_seq_length", "384",
            "--doc_stride", "128",
            "--per_device_eval_batch_size", "32",
            "--output_dir", str(valid_dir),
            "--overwrite_cache",
            "--val_column", "validation",
            "--do_eval",
        ], cwd=classifier_dir)
        valid_metric_file = valid_dir / "final_eval_results.json"
        valid_metrics = json.loads(valid_metric_file.read_text()) if valid_metric_file.exists() else {}
        valid_acc = valid_metrics.get("final_acc_score")

        predict_dir = ep_out / "predict"
        predict_dir.mkdir(exist_ok=True)
        run([
            "python", "run_classifier_plus.py",
            "--model_name_or_path", str(ep_out),
            "--validation_file", predict_file,
            "--question_column", "question",
            "--answer_column", "answer",
            "--max_seq_length", "384",
            "--doc_stride", "128",
            "--per_device_eval_batch_size", "32",
            "--output_dir", str(predict_dir),
            "--overwrite_cache",
            "--val_column", "validation",
            "--do_eval",
        ], cwd=classifier_dir)

        summary["runs"].append({
            "epoch": ep,
            "train_seconds": train_secs,
            "checkpoint_dir": str(ep_out.relative_to(REPO_ROOT)),
            "valid_dir": str(valid_dir.relative_to(REPO_ROOT)),
            "predict_dir": str(predict_dir.relative_to(REPO_ROOT)),
            "valid_final_acc_score": valid_acc,
        })
        log(f"  <<< epoch={ep} valid_final_acc_score={valid_acc}")

    runs_with_metric = [r for r in summary["runs"] if r["valid_final_acc_score"] is not None]
    if not runs_with_metric:
        raise RuntimeError("No 4-way classifier sweep run produced a validation metric")
    best = max(runs_with_metric, key=lambda r: r["valid_final_acc_score"])
    summary["best_epoch"] = best["epoch"]
    summary["best_valid_acc"] = best["valid_final_acc_score"]
    log(f"  *** best epoch={best['epoch']} valid_final_acc_score={best['valid_final_acc_score']}")
    best_predict = REPO_ROOT / best["predict_dir"] / "dict_id_pred_results.json"
    return Path(best["checkpoint_dir"]), best_predict, summary


# ----------------- Phase: 4-way postprocess + adaptive evaluation ----------------- #

def run_postprocess_plus(best_predict: Path) -> Path:
    section("4-way postprocess (route A/B/C/D to per-dataset adaptive predictions)")
    rel = best_predict.relative_to(REPO_ROOT)
    run([
        "python", "classifier/postprocess/predict_complexity_on_classification_results_plus.py",
        "gpt",
        "--classification_result_file", str(REPO_ROOT / rel),
    ], cwd=REPO_ROOT)
    parts = rel.parts
    idx = parts.index("model")
    after_model = parts[idx + 1 : -2]
    return REPO_ROOT / "predictions" / "classifier_plus" / Path(*after_model)


# ----------------- adaptive-rag (3-way) helpers ----------------- #

THREE_WAY_DATA_DIR = "musique_hotpot_wiki2_nq_tqa_sqd"


def _discover_3way_artifacts(
    *,
    base_model: str = "t5-small",
    llm_tag: str = "flan_t5_xl",
) -> Optional[dict]:
    """Find the latest 3-way classifier sweep and select the best epoch.

    Returns a dict with ``checkpoint_dir``, ``predict_results_file``, and
    ``adaptive_root`` (under ``predictions/classifier/``) or ``None`` if
    no usable sweep is on disk.
    """
    sweep_root = (
        REPO_ROOT / "classifier" / "outputs" / THREE_WAY_DATA_DIR / "model"
        / base_model / llm_tag / "sweep"
    )
    if not sweep_root.exists():
        return None
    sweep_dirs = sorted([p for p in sweep_root.iterdir() if p.is_dir()])
    if not sweep_dirs:
        return None
    latest = sweep_dirs[-1]

    best = None
    for ep_dir in sorted(latest.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("epoch_"):
            continue
        valid_metric = ep_dir / "valid" / "final_eval_results.json"
        if not valid_metric.exists():
            continue
        try:
            acc = json.loads(valid_metric.read_text()).get("final_acc_score")
        except Exception:
            continue
        if acc is None:
            continue
        if best is None or acc > best["valid_acc"]:
            best = {
                "epoch": int(ep_dir.name.split("_", 1)[1]),
                "valid_acc": float(acc),
                "checkpoint_dir": ep_dir,
            }
    if best is None:
        return None

    predict_file = best["checkpoint_dir"] / "predict" / "dict_id_pred_results.json"
    adaptive_root = (
        REPO_ROOT / "predictions" / "classifier" / base_model / llm_tag / "sweep"
        / latest.name / f"epoch_{best['epoch']}"
    )
    return {
        "sweep_dir": latest,
        "best_epoch": best["epoch"],
        "best_valid_acc": best["valid_acc"],
        "checkpoint_dir": best["checkpoint_dir"],
        "predict_results_file": predict_file if predict_file.exists() else None,
        "adaptive_root": adaptive_root if adaptive_root.exists() else None,
    }


def _build_d_predict_file(
    d_test_qids_per_dataset: Dict[str, List[str]],
    out_path: Path,
    longbench_cache_dir: Optional[Path] = None,
) -> int:
    """Materialise a 3-way-compatible predict.json containing only D test
    queries. Each row uses ``answer="D"`` (placeholder; the 3-way
    classifier only outputs A/B/C, so the recorded accuracy field is
    meaningless for these rows and we don't use it)."""
    rows: List[dict] = []
    for ds, qids in d_test_qids_per_dataset.items():
        wanted = set(qids)
        recs = longbench.load_normalized(
            ds, cache_dir=longbench_cache_dir or REPO_ROOT / "raw_data" / "longbench"
        )
        for r in recs:
            if r.qid in wanted:
                rows.append({
                    "answer": "D",
                    "answer_description": "global",
                    "dataset_name": r.dataset,
                    "id": r.qid,
                    "question": r.query,
                })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=4))
    return len(rows)


def run_3way_classifier_on_d_test(
    checkpoint_dir: Path,
    d_test_qids_per_dataset: Dict[str, List[str]],
    out_dir: Path,
) -> Path:
    """Run the existing 3-way classifier on the D test slice and return the
    path to its ``dict_id_pred_results.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier_dir = REPO_ROOT / "classifier"
    predict_input = out_dir / "d_predict.json"
    n = _build_d_predict_file(d_test_qids_per_dataset, predict_input)
    log(f"  3-way D predict file: {predict_input.relative_to(REPO_ROOT)} (n={n})")

    rel_input = os.path.relpath(predict_input, classifier_dir)
    rel_ckpt = os.path.relpath(checkpoint_dir, classifier_dir)
    rel_out = os.path.relpath(out_dir, classifier_dir)

    run([
        "python", "run_classifier.py",
        "--model_name_or_path", rel_ckpt,
        "--validation_file", rel_input,
        "--question_column", "question",
        "--answer_column", "answer",
        "--max_seq_length", "384",
        "--doc_stride", "128",
        "--per_device_eval_batch_size", "32",
        "--output_dir", rel_out,
        "--overwrite_cache",
        "--val_column", "validation",
        "--do_eval",
    ], cwd=classifier_dir)

    pred_file = out_dir / "dict_id_pred_results.json"
    if not pred_file.exists():
        raise RuntimeError(f"3-way classifier did not produce {pred_file}")
    return pred_file


def _d_pipeline_pred_jsonl_to_qid_pred(out_dir: Path) -> Dict[str, str]:
    """Read a D pipeline's predictions.jsonl into qid -> prediction text."""
    out: Dict[str, str] = {}
    pf = out_dir / "predictions.jsonl"
    with pf.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row.get("qid"))] = row.get("prediction", "")
    return out


def _d_pipeline_trace_to_qid_step_time(out_dir: Path) -> Dict[str, Tuple[float, float]]:
    """Read a D pipeline's traces.jsonl into qid -> (step_count, latency)."""
    out: Dict[str, Tuple[float, float]] = {}
    tf = out_dir / "traces.jsonl"
    with tf.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row.get("qid"))] = (
                float(row.get("step_count", 0.0)),
                float(row.get("latency_seconds", 0.0)),
            )
    return out


def assemble_3way_adaptive_d_predictions(
    classification_pred_file: Path,
    d_test_qids_per_dataset: Dict[str, List[str]],
    d_paths: Dict[str, Dict[str, Path]],
    out_root: Path,
) -> Dict[str, Path]:
    """Assemble per-D-dataset adaptive predictions for the 3-way row.

    For each D qid the 3-way classifier emits A/B/C; we route to the
    corresponding D baseline pipeline output. Returns
    ``{dataset: per_dataset_dir}``; each dir contains ``<ds>.json`` and
    ``<ds>_option.json`` (mirrors the 4-way postprocess output).
    """
    cls_preds = json.loads(classification_pred_file.read_text())
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_per_ds: Dict[str, Path] = {}

    label_to_pipeline = {"A": "nor_qa", "B": "oner_qa", "C": "ircot_qa"}

    for ds, qids in d_test_qids_per_dataset.items():
        ds_dir = out_root / ds
        ds_dir.mkdir(parents=True, exist_ok=True)

        baseline_lookups: Dict[str, Dict[str, str]] = {}
        baseline_steps: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for pipe in ("nor_qa", "oner_qa", "ircot_qa"):
            base_dir = d_paths[pipe][ds]
            baseline_lookups[pipe] = _d_pipeline_pred_jsonl_to_qid_pred(base_dir)
            baseline_steps[pipe] = _d_pipeline_trace_to_qid_step_time(base_dir)

        ds_pred: Dict[str, str] = {}
        ds_opt: Dict[str, dict] = {}
        for qid in qids:
            info = cls_preds.get(qid)
            if not info:
                log(f"  WARNING: 3-way classifier missing qid={qid} ({ds})")
                continue
            opt = info.get("prediction") if isinstance(info, dict) else info
            if opt not in label_to_pipeline:
                log(f"  WARNING: 3-way classifier returned unsupported label {opt!r} for qid={qid}")
                continue
            pipe = label_to_pipeline[opt]
            pred = baseline_lookups[pipe].get(qid, "")
            step, _ = baseline_steps[pipe].get(qid, (0.0, 0.0))
            ds_pred[qid] = pred
            ds_opt[qid] = {"prediction": pred, "option": opt, "stepNum": step}

        (ds_dir / f"{ds}.json").write_text(json.dumps(ds_pred, indent=4, sort_keys=True))
        (ds_dir / f"{ds}_option.json").write_text(json.dumps(ds_opt, indent=4, sort_keys=True))
        out_per_ds[ds] = ds_dir
    return out_per_ds


def collect_adaptive_rag_3way_qa_metrics(
    adaptive_root_3way: Path, d_paths: Dict[str, Dict[str, Path]]
) -> Dict[str, dict]:
    """Read existing 3-way adaptive QA outputs (predictions/classifier/...).
    Mirrors :func:`collect_adaptive_qa_metrics` but pointed at the 3-way
    output tree."""
    out: Dict[str, dict] = {}
    for ds in QA_DATASETS:
        ds_dir = adaptive_root_3way / ds
        ds_pred_file = ds_dir / f"{ds}.json"
        opt_file = ds_dir / f"{ds}_option.json"
        if not ds_pred_file.exists():
            log(f"  WARNING: adaptive-rag (3-way) QA pred missing for {ds}: {ds_pred_file}")
            out[ds] = {}
            continue
        ds_pred = json.loads(ds_pred_file.read_text())
        opts = json.loads(opt_file.read_text()) if opt_file.exists() else {}

        em, f1, acc = _eval_adaptive_qa_em_f1_acc(ds, ds_pred)
        avg_step, avg_time = _adaptive_step_time(ds, opts, d_paths)
        out[ds] = {
            "EM": em, "F1": f1, "Acc": acc, "Step": avg_step, "Time": avg_time,
            "count": len(ds_pred),
            "_provenance": {
                "predictions_file": str(ds_pred_file.relative_to(REPO_ROOT)),
                "option_file": str(opt_file.relative_to(REPO_ROOT)),
                "step_time_source": "weighted from base pipeline outputs by routed option (A/B/C only)",
                "qa_eval": "hotpot_official" if ds == "hotpotqa" else "evaluate_predictions(em,f1,acc)",
            },
        }
    return out


def collect_adaptive_rag_3way_d_metrics(
    adaptive_root_3way_d: Dict[str, Path],
    d_paths: Dict[str, Dict[str, Path]],
) -> Dict[str, dict]:
    """Compute ROUGE-L / Step / Time for the 3-way row's longbench cell.

    For each D dataset we read the per-qid prediction file produced by
    :func:`assemble_3way_adaptive_d_predictions` and score it via the
    same ROUGE-L definition used for the 4-way row, then read per-qid
    Step/Time from the routed baseline pipeline's traces.
    """
    per_d: Dict[str, dict] = {}
    for ds, ds_dir in adaptive_root_3way_d.items():
        ds_pred_file = ds_dir / f"{ds}.json"
        opt_file = ds_dir / f"{ds}_option.json"
        if not ds_pred_file.exists():
            log(f"  WARNING: adaptive-rag (3-way) D pred missing for {ds}: {ds_pred_file}")
            per_d[ds] = {}
            continue
        id_to_pred = json.loads(ds_pred_file.read_text())
        opts = json.loads(opt_file.read_text()) if opt_file.exists() else {}

        all_recs = longbench.load_normalized(ds)
        id_to_refs = {r.qid: r.references for r in all_recs}
        pairs: List[Tuple[str, List[str]]] = []
        for qid, pred in id_to_pred.items():
            refs = id_to_refs.get(qid, [])
            if not refs:
                continue
            pairs.append((pred or "", refs))
        rouge_l = evaluation.compute_rouge_l_for_pairs(pairs)

        baseline_steps: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for pipe in ("nor_qa", "oner_qa", "ircot_qa"):
            baseline_steps[pipe] = _d_pipeline_trace_to_qid_step_time(d_paths[pipe][ds])
        label_to_pipeline = {"A": "nor_qa", "B": "oner_qa", "C": "ircot_qa"}

        steps: List[float] = []
        times: List[float] = []
        for qid, info in opts.items():
            opt = info.get("option")
            if opt not in label_to_pipeline:
                continue
            step, lat = baseline_steps[label_to_pipeline[opt]].get(qid, (0.0, 0.0))
            steps.append(step)
            times.append(lat)
        avg_step = (sum(steps) / len(steps)) if steps else None
        avg_time = (sum(times) / len(times)) if times else None
        per_d[ds] = {
            "ROUGE-L": rouge_l,
            "Step": avg_step,
            "Time": avg_time,
            "count": len(id_to_pred),
            "_provenance": {
                "predictions_file": str(ds_pred_file.relative_to(REPO_ROOT)),
                "option_file": str(opt_file.relative_to(REPO_ROOT)),
                "metric": "rougeL_max_over_refs_then_mean",
                "step_time_source": "from D baseline traces.jsonl, weighted by routed option (A/B/C)",
            },
        }
    out = {"_per_d": per_d, "longbench": evaluation.aggregate_longbench(per_d)}
    out["longbench"]["_provenance"] = {
        "aggregation": "macro-average over " + ",".join(D_DATASETS),
        "per_dataset": per_d,
    }
    return out


# ----------------- Metric collection ----------------- #

def collect_qa_baseline_metrics() -> Dict[str, Dict[str, dict]]:
    """Reuse existing 3-way QA artifacts for trivia/nq/hotpotqa columns."""
    out: Dict[str, Dict[str, dict]] = {}
    for pipe in ("nor_qa", "oner_qa", "ircot_qa"):
        out[pipe] = {}
        for ds in QA_DATASETS:
            ev_path = base_eval_path(pipe, ds)
            tm_path = base_time_path(pipe, ds)
            pr_path = base_pred_path(pipe, ds)
            if not (ev_path.exists() and tm_path.exists() and pr_path.exists()):
                log(f"  WARNING: missing 3-way QA artifact for {pipe}/{ds}; cell will be NA")
                out[pipe][ds] = {}
                continue
            ev = json.loads(ev_path.read_text())
            total_t = float(tm_path.read_text().strip())
            n = ev.get("count") or 0
            avg_t = (total_t / n) if n else None
            if pipe == "ircot_qa":
                sn = json.loads(base_step_path(pipe, ds).read_text())
                avg_step = sum(sn.values()) / len(sn)
            elif pipe == "oner_qa":
                avg_step = 1.0
            else:
                avg_step = 0.0
            acc, _ = _qa_acc_for_predictions(pr_path, ds)
            out[pipe][ds] = {
                "EM": ev.get("em"),
                "F1": ev.get("f1"),
                "Acc": acc,
                "Step": avg_step,
                "Time": avg_t,
                "count": n,
                "_provenance": {
                    "eval_metrics_file": str(ev_path.relative_to(REPO_ROOT)),
                    "prediction_file": str(pr_path.relative_to(REPO_ROOT)),
                    "time_file": str(tm_path.relative_to(REPO_ROOT)),
                    "step_file": str(base_step_path(pipe, ds).relative_to(REPO_ROOT)) if pipe == "ircot_qa" else None,
                    "acc_method": "hotpot_official" if ds == "hotpotqa" else "substring_normalized",
                },
            }
    return out


def collect_d_metrics(d_paths: Dict[str, Dict[str, Path]]) -> Dict[str, Dict[str, dict]]:
    """Compute ROUGE-L (+ Step/Time) per (pipeline, D-dataset) and aggregate longbench."""
    out: Dict[str, Dict[str, dict]] = {p: {} for p in ("nor_qa", "oner_qa", "ircot_qa", "global_qa")}
    per_d_per_pipe: Dict[str, Dict[str, dict]] = {p: {} for p in out}
    for pipe in out:
        for ds in D_DATASETS:
            preds = d_paths[pipe][ds] / "predictions.jsonl"
            metrics = evaluation.evaluate_d_predictions_jsonl(preds)
            metrics["_provenance"] = {
                "predictions_file": str(preds.relative_to(REPO_ROOT)),
                "traces_file": str((d_paths[pipe][ds] / "traces.jsonl").relative_to(REPO_ROOT)),
                "metric": "rougeL_max_over_refs_then_mean",
            }
            per_d_per_pipe[pipe][ds] = metrics
        out[pipe]["_per_d"] = per_d_per_pipe[pipe]
        out[pipe]["longbench"] = evaluation.aggregate_longbench(per_d_per_pipe[pipe])
        out[pipe]["longbench"]["_provenance"] = {
            "aggregation": "macro-average over " + ",".join(D_DATASETS),
            "per_dataset": {ds: per_d_per_pipe[pipe][ds] for ds in D_DATASETS},
        }
    return out


def collect_adaptive_qa_metrics(adaptive_root: Path, d_paths: Dict[str, Dict[str, Path]]) -> Dict[str, dict]:
    """For adaptive-rag+ on QA datasets: read per-dataset adaptive predictions
    + use base time/step weighted by routed option (A/B/C/D)."""
    out: Dict[str, dict] = {}
    for ds in QA_DATASETS:
        eval_file_dir = adaptive_root / ds
        ds_pred_file = eval_file_dir / f"{ds}.json"
        opt_file = eval_file_dir / f"{ds}_option.json"
        if not ds_pred_file.exists():
            log(f"  WARNING: adaptive QA pred missing for {ds}: {ds_pred_file}")
            out[ds] = {}
            continue
        ds_pred = json.loads(ds_pred_file.read_text())
        opts = json.loads(opt_file.read_text()) if opt_file.exists() else {}

        em, f1, acc = _eval_adaptive_qa_em_f1_acc(ds, ds_pred)
        avg_step, avg_time = _adaptive_step_time(ds, opts, d_paths)
        out[ds] = {
            "EM": em, "F1": f1, "Acc": acc, "Step": avg_step, "Time": avg_time,
            "count": len(ds_pred),
            "_provenance": {
                "predictions_file": str(ds_pred_file.relative_to(REPO_ROOT)),
                "option_file": str(opt_file.relative_to(REPO_ROOT)),
                "step_time_source": "weighted from base pipeline outputs by routed option",
                "qa_eval": "hotpot_official" if ds == "hotpotqa" else "evaluate_predictions(em,f1,acc)",
            },
        }
    return out


def _eval_adaptive_qa_em_f1_acc(dataset: str, id_to_pred: Dict[str, str]) -> Tuple[float, float, float]:
    """Use evaluate.evaluate_by_dicts to get EM/F1/Acc for a per-dataset adaptive output."""
    from evaluate import evaluate_by_dicts  # type: ignore

    id_to_gt = _load_id_to_gt(processed_test_path(dataset))
    if dataset == "hotpotqa":
        acc, _ = _qa_acc_hotpot_official_inline(id_to_pred)
    else:
        acc = None  # filled below
    em_total = 0.0
    f1_total = 0.0
    n = 0
    for qid, gt in id_to_gt.items():
        if qid not in id_to_pred:
            continue
        pred = id_to_pred[qid]
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        pred = _qa_extract(str(pred))
        from evaluate import SquadAnswerEmF1Metric  # type: ignore
        m = SquadAnswerEmF1Metric()
        m(pred, gt)
        # The repo's metric returns a dict {em, f1, count}, but older
        # variants returned a (em, f1) tuple. Handle both for safety.
        out = m.get_metric()
        if isinstance(out, dict):
            em_per = float(out.get("em", 0.0))
            f1_per = float(out.get("f1", 0.0))
        else:
            em_per, f1_per = out  # type: ignore[misc]
        em_total += em_per
        f1_total += f1_per
        if dataset != "hotpotqa":
            if acc is None:
                acc = 0.0
            acc += _qa_acc(pred, gt)
        n += 1
    em = em_total / n if n else 0.0
    f1 = f1_total / n if n else 0.0
    if dataset == "hotpotqa":
        return em, f1, acc
    return em, f1, (acc / n) if n else 0.0


def _qa_acc_hotpot_official_inline(id_to_pred: Dict[str, str]) -> Tuple[float, int]:
    qids = list(id_to_pred.keys())
    raw = json.loads((REPO_ROOT / "raw_data" / "hotpotqa" / "hotpot_dev_distractor_v1.json").read_text())
    filtered = [d for d in raw if d["_id"] in set(qids)]
    tmp_dir = REPO_ROOT / ".temp"
    tmp_dir.mkdir(exist_ok=True)
    gt_path = tmp_dir / uuid.uuid4().hex
    pr_path = tmp_dir / uuid.uuid4().hex
    out_path = tmp_dir / uuid.uuid4().hex
    gt_path.write_text(json.dumps(filtered))
    answer_dict = {}
    for qid, p in id_to_pred.items():
        if isinstance(p, list):
            p = " ".join(str(x) for x in p) if len(p) > 1 else str(p[0]) if p else ""
        answer_dict[qid] = str(p)
    pr_path.write_text(json.dumps({"answer": answer_dict, "sp": {qid: [["", 0]] for qid in id_to_pred}}))
    rel_gt = os.path.join("..", "..", str(gt_path.relative_to(REPO_ROOT)))
    rel_pr = os.path.join("..", "..", str(pr_path.relative_to(REPO_ROOT)))
    rel_out = os.path.join("..", "..", str(out_path.relative_to(REPO_ROOT)))
    cmd = (
        f"cd official_evaluation/hotpotqa ; "
        f"python hotpot_evaluate_v1.py {rel_pr} {rel_gt} > {rel_out}"
    )
    rc = subprocess.call(cmd, shell=True, cwd=str(REPO_ROOT))
    if rc != 0 or not out_path.exists():
        gt_path.unlink(missing_ok=True)
        pr_path.unlink(missing_ok=True)
        raise RuntimeError("Official hotpotqa eval failed")
    metrics = eval(out_path.read_text().strip())
    gt_path.unlink(missing_ok=True)
    pr_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return float(metrics["acc"]), len(id_to_pred)


def _adaptive_step_time(
    ds: str, opts: Dict[str, dict], d_paths: Dict[str, Dict[str, Path]]
) -> Tuple[Optional[float], Optional[float]]:
    """Weighted Step/Time using base pipeline timings for the routed option.

    For QA datasets, A/B/C times come from the existing 3-way artifacts;
    D never routes for QA datasets, so D path is unused here. Step values
    follow the postprocess script (A=0, B=1, C=ircot stepNum, D=1).
    """
    total_step = 0.0
    total_time = 0.0
    n = 0
    for qid, info in opts.items():
        opt = info.get("option")
        step = info.get("stepNum", 0)
        total_step += float(step)
        if opt == "A":
            src = "nor_qa"
        elif opt == "B":
            src = "oner_qa"
        elif opt == "C":
            src = "ircot_qa"
        else:
            n += 1
            continue
        t_path = base_time_path(src, ds)
        ev_path = base_eval_path(src, ds)
        try:
            total_t = float(t_path.read_text().strip())
            cnt = json.loads(ev_path.read_text()).get("count") or 0
            if cnt:
                total_time += total_t / cnt
        except FileNotFoundError:
            pass
        n += 1
    return (total_step / n) if n else None, (total_time / n) if n else None


def collect_adaptive_d_metrics(adaptive_root: Path) -> Dict[str, dict]:
    """Adaptive metrics for class-D columns: ROUGE-L from per-dataset
    adaptive prediction files (which deterministically equal the global
    pipeline output when classifier routes D correctly)."""
    per_d: Dict[str, dict] = {}
    for ds in D_DATASETS:
        ds_pred_file = adaptive_root / ds / f"{ds}.json"
        opt_file = adaptive_root / ds / f"{ds}_option.json"
        if not ds_pred_file.exists():
            log(f"  WARNING: adaptive D pred missing for {ds}: {ds_pred_file}")
            per_d[ds] = {}
            continue
        id_to_pred = json.loads(ds_pred_file.read_text())
        all_recs = longbench.load_normalized(ds)
        id_to_refs = {r.qid: r.references for r in all_recs}
        pairs = []
        for qid, pred in id_to_pred.items():
            refs = id_to_refs.get(qid, [])
            if not refs:
                continue
            pairs.append((pred or "", refs))
        rouge_l = evaluation.compute_rouge_l_for_pairs(pairs)
        opts = json.loads(opt_file.read_text()) if opt_file.exists() else {}
        steps = [float(v.get("stepNum", 0)) for v in opts.values()]
        avg_step = (sum(steps) / len(steps)) if steps else None
        per_d[ds] = {
            "ROUGE-L": rouge_l,
            "Step": avg_step,
            "Time": None,  # adaptive D time captured below from D pipeline traces if available
            "count": len(id_to_pred),
            "_provenance": {
                "predictions_file": str(ds_pred_file.relative_to(REPO_ROOT)),
                "option_file": str(opt_file.relative_to(REPO_ROOT)),
            },
        }
    out = {"_per_d": per_d, "longbench": evaluation.aggregate_longbench(per_d)}
    out["longbench"]["_provenance"] = {
        "aggregation": "macro-average over " + ",".join(D_DATASETS),
        "per_dataset": per_d,
    }
    return out


# ----------------- Driver ----------------- #

def main() -> int:
    _setup_logging()

    parser = argparse.ArgumentParser(description="AdaptiveRAG+ orchestrator")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, nargs="+", default=None)
    parser.add_argument("--d-test-per-dataset", type=int, default=None,
                        help="override the D test slice size per dataset")
    parser.add_argument("--d-train-per-dataset", type=int, default=None,
                        help="override the D classifier train slice size per dataset")
    parser.add_argument("--d-valid-per-dataset", type=int, default=None,
                        help="override the D classifier valid slice size per dataset")
    parser.add_argument("--num-clusters", type=int, default=None,
                        help="(DEPRECATED) indexing hyperparameter; configure when "
                             "running scripts/build_longbench_index.py instead. "
                             "Kept for backward compatibility but unused here.")
    parser.add_argument("--chunk-chars", type=int, default=global_index.DEFAULT_CHUNK_CHARS,
                        help="char window for D baselines (must match index when relevant).")
    parser.add_argument("--chunk-overlap", type=int, default=global_index.DEFAULT_CHUNK_OVERLAP,
                        help="char overlap for D baselines.")
    parser.add_argument("--summary-max-tokens", type=int, default=global_index.DEFAULT_SUMMARY_MAX_TOKENS,
                        help="(DEPRECATED) configure on scripts/build_longbench_index.py; "
                             "unused by this orchestrator.")
    parser.add_argument("--gen-max-tokens", type=int, default=baselines_d.DEFAULT_GEN_MAX_TOKENS)
    parser.add_argument("--n-ircot-steps", type=int, default=baselines_d.DEFAULT_IRCOT_STEPS)
    parser.add_argument("--embed-model", type=str, default=global_index.DEFAULT_EMBED_MODEL)
    parser.add_argument("--gen-model", type=str, default=global_index.DEFAULT_COMPLETION_MODEL)
    parser.add_argument("--inter-request-pause", type=float,
                        default=global_index.DEFAULT_INTER_REQUEST_PAUSE,
                        help="(DEPRECATED for indexing) configure on "
                             "scripts/build_longbench_index.py instead; the "
                             "orchestrator no longer indexes inline.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-classifier-sweep", action="store_true",
                        help="reuse existing 4-way classifier predict file")
    parser.add_argument("--classifier-predict-file", type=str, default=None,
                        help="path to dict_id_pred_results.json from a prior 4-way sweep best epoch")
    parser.add_argument("--skip-adaptive-rag-3way", action="store_true",
                        help="skip the original 3-way adaptive-rag row entirely")
    parser.add_argument("--adaptive-rag-3way-ckpt", type=str, default=None,
                        help="explicit 3-way classifier checkpoint dir to use for D inference "
                             "(default: auto-discover under classifier/outputs/.../sweep)")
    parser.add_argument("--adaptive-rag-3way-adaptive-root", type=str, default=None,
                        help="explicit 3-way adaptive root for QA cells (default: auto-discover "
                             "under predictions/classifier/.../sweep)")
    args = parser.parse_args()

    if args.smoke_test == args.full_run:
        parser.error("Pick exactly one of --smoke-test or --full-run")
    mode = "smoke" if args.smoke_test else "full"

    if args.epochs is None:
        args.epochs = [1, 2] if mode == "smoke" else [10, 15, 20, 25]
    if args.d_test_per_dataset is None:
        args.d_test_per_dataset = 4 if mode == "smoke" else 40
    if args.d_train_per_dataset is None:
        args.d_train_per_dataset = 20 if mode == "smoke" else 80
    if args.d_valid_per_dataset is None:
        args.d_valid_per_dataset = 6 if mode == "smoke" else 20
    if args.num_clusters is None:
        args.num_clusters = 4 if mode == "smoke" else 8

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_plus_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    section(f"AdaptiveRAG+ orchestration ({mode}) -> {out_dir}")

    notes: List[str] = [
        f"# AdaptiveRAG+ run notes ({mode})",
        f"- timestamp: {datetime.now().isoformat()}",
        f"- out_dir: {out_dir}",
        f"- d test/train/valid per dataset: {args.d_test_per_dataset}/{args.d_train_per_dataset}/{args.d_valid_per_dataset}",
        f"- chunk_chars/chunk_overlap/num_clusters: {args.chunk_chars}/{args.chunk_overlap}/{args.num_clusters}",
        f"- embed_model={args.embed_model} gen_model={args.gen_model}",
    ]

    section("Phase 0: preflight checks (LongBench data + prebuilt indexes)")
    longbench.download_longbench_jsonl("gov_report")
    longbench.download_longbench_jsonl("qmsum")
    missing_indexes = [ds for ds in D_DATASETS if not global_index.index_exists(ds)]
    if missing_indexes:
        cmd = "python scripts/build_longbench_index.py"
        if missing_indexes != D_DATASETS:
            cmd += " " + " ".join(f"--dataset {d}" for d in missing_indexes)
        raise SystemExit(
            "\nMissing LongBench global index(es) for: "
            + ", ".join(missing_indexes)
            + "\nIndexing is now decoupled from this orchestrator and is fully "
            + "resumable. Build the indexes first, then re-run this script:\n\n"
            + f"    {cmd}\n\n"
            + "If a previous build was interrupted (rate limit, kill, etc.), the "
            + "above command will pick up exactly where it left off. "
            + "Run with --status to inspect progress."
        )
    for ds in D_DATASETS:
        st = global_index.index_build_status(ds)
        log(f"  index[{ds}] OK: chunks={st.get('n_chunks')} clusters={st.get('n_clusters_actual')}")

    section("Phase 1: data and 4-way classifier corpus")
    corpus_info = routing.build_plus_corpus(
        train_per_dataset=args.d_train_per_dataset,
        valid_per_dataset=args.d_valid_per_dataset,
        test_per_dataset=args.d_test_per_dataset,
        seed=args.seed,
    )
    plus_data_dir = REPO_ROOT / "classifier" / "data" / "musique_hotpot_wiki2_nq_tqa_sqd_plus"
    (plus_data_dir / "corpus_info.json").write_text(json.dumps(corpus_info, indent=2))
    (out_dir / "classifier_corpus_info.json").write_text(json.dumps(corpus_info, indent=2))

    section("Phase 2: D pipelines (nor/oner/ircot/global) over LongBench D test slice")
    d_paths = run_all_d_pipelines(
        test_per_dataset=args.d_test_per_dataset,
        seed=args.seed,
        embed_model=args.embed_model,
        gen_model=args.gen_model,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        num_clusters=args.num_clusters,
        summary_max_tokens=args.summary_max_tokens,
        gen_max_tokens=args.gen_max_tokens,
        n_ircot_steps=args.n_ircot_steps,
        skip_existing=args.skip_existing,
        inter_request_pause=args.inter_request_pause,
    )

    section("Phase 3: 4-way classifier sweep")
    if args.skip_classifier_sweep:
        if not args.classifier_predict_file:
            raise SystemExit("--skip-classifier-sweep requires --classifier-predict-file")
        best_predict = Path(args.classifier_predict_file).resolve()
        if not best_predict.exists():
            raise SystemExit(f"Missing classifier predict file: {best_predict}")
        sweep_summary = {"reused_predict_file": str(best_predict.relative_to(REPO_ROOT))}
    else:
        _, best_predict, sweep_summary = classifier_sweep_plus(epochs=args.epochs)
    (out_dir / "classifier_sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2, default=str))
    notes.append(f"- classifier sweep epochs: {args.epochs}")
    notes.append(f"- best 4-way classifier predict file: {best_predict}")

    section("Phase 4: 4-way postprocess")
    adaptive_root = run_postprocess_plus(best_predict)
    notes.append(f"- adaptive output root: {adaptive_root.relative_to(REPO_ROOT)}")

    section("Phase 4b: adaptive-rag (3-way) inference on D test slice")
    three_way_info: dict = {"used": False}
    adaptive_root_3way: Optional[Path] = None
    adaptive_root_3way_d: Dict[str, Path] = {}
    if args.skip_adaptive_rag_3way:
        log("  --skip-adaptive-rag-3way: skipping 3-way row (will be reported as NA)")
    else:
        if args.adaptive_rag_3way_adaptive_root:
            adaptive_root_3way = Path(args.adaptive_rag_3way_adaptive_root).resolve()
            log(f"  using explicit 3-way adaptive root: {adaptive_root_3way}")
        if args.adaptive_rag_3way_ckpt:
            three_way_ckpt = Path(args.adaptive_rag_3way_ckpt).resolve()
        else:
            disc = _discover_3way_artifacts()
            if disc is None:
                log("  no existing 3-way classifier sweep found; 3-way row will be NA")
                three_way_ckpt = None
            else:
                three_way_ckpt = disc["checkpoint_dir"]
                three_way_info.update({
                    "discovered_sweep_dir": str(disc["sweep_dir"].relative_to(REPO_ROOT)),
                    "best_epoch": disc["best_epoch"],
                    "best_valid_acc": disc["valid_acc"] if "valid_acc" in disc else disc.get("best_valid_acc"),
                })
                if adaptive_root_3way is None and disc.get("adaptive_root") is not None:
                    adaptive_root_3way = disc["adaptive_root"]
                    log(f"  discovered 3-way adaptive root: {adaptive_root_3way.relative_to(REPO_ROOT)}")
                if three_way_ckpt is not None:
                    log(f"  discovered 3-way classifier ckpt:   {three_way_ckpt.relative_to(REPO_ROOT)} (epoch={disc['best_epoch']})")

        d_test_qids_per_dataset = {
            ds: list(corpus_info["test_qids_per_dataset"].get(ds, []))[: args.d_test_per_dataset]
            for ds in D_DATASETS
        }

        if three_way_ckpt is not None:
            three_way_d_dir = out_dir / "adaptive_rag_3way" / "classifier_d_inference"
            three_way_pred_file = run_3way_classifier_on_d_test(
                three_way_ckpt, d_test_qids_per_dataset, three_way_d_dir
            )
            three_way_info["d_classifier_predict_file"] = str(three_way_pred_file.relative_to(REPO_ROOT))

            three_way_adaptive_d_root = out_dir / "adaptive_rag_3way" / "adaptive_d"
            adaptive_root_3way_d = assemble_3way_adaptive_d_predictions(
                three_way_pred_file, d_test_qids_per_dataset, d_paths, three_way_adaptive_d_root
            )
            three_way_info["adaptive_d_root"] = str(three_way_adaptive_d_root.relative_to(REPO_ROOT))
            three_way_info["used"] = True
        if adaptive_root_3way is not None:
            three_way_info["adaptive_root_qa"] = str(adaptive_root_3way.relative_to(REPO_ROOT))
            notes.append(f"- adaptive-rag (3-way) QA root: {adaptive_root_3way.relative_to(REPO_ROOT)}")
        else:
            notes.append("- adaptive-rag (3-way) QA root: NOT FOUND (cells reported as NA)")

    section("Phase 5: metric collection (QA + ROUGE-L)")
    qa_baseline_metrics = collect_qa_baseline_metrics()
    d_metrics_per_pipe = collect_d_metrics(d_paths)
    adaptive_qa_metrics = collect_adaptive_qa_metrics(adaptive_root, d_paths)
    adaptive_d_metrics = collect_adaptive_d_metrics(adaptive_root)

    if adaptive_root_3way is not None:
        adaptive_rag_3way_qa = collect_adaptive_rag_3way_qa_metrics(adaptive_root_3way, d_paths)
    else:
        adaptive_rag_3way_qa = {ds: {} for ds in QA_DATASETS}
    if adaptive_root_3way_d:
        adaptive_rag_3way_d = collect_adaptive_rag_3way_d_metrics(adaptive_root_3way_d, d_paths)
    else:
        adaptive_rag_3way_d = {"_per_d": {}, "longbench": {}}

    metrics: Dict[str, Dict[str, dict]] = {
        "nor_qa": {ds: qa_baseline_metrics["nor_qa"].get(ds, {}) for ds in QA_DATASETS},
        "oner_qa": {ds: qa_baseline_metrics["oner_qa"].get(ds, {}) for ds in QA_DATASETS},
        "ircot_qa": {ds: qa_baseline_metrics["ircot_qa"].get(ds, {}) for ds in QA_DATASETS},
        "adaptive-rag": {ds: adaptive_rag_3way_qa.get(ds, {}) for ds in QA_DATASETS},
        "adaptive-rag+": {ds: adaptive_qa_metrics.get(ds, {}) for ds in QA_DATASETS},
    }
    metrics["nor_qa"]["longbench"] = d_metrics_per_pipe["nor_qa"]["longbench"]
    metrics["oner_qa"]["longbench"] = d_metrics_per_pipe["oner_qa"]["longbench"]
    metrics["ircot_qa"]["longbench"] = d_metrics_per_pipe["ircot_qa"]["longbench"]
    metrics["adaptive-rag"]["longbench"] = adaptive_rag_3way_d.get("longbench", {})
    metrics["adaptive-rag+"]["longbench"] = adaptive_d_metrics["longbench"]

    section("Phase 6: emit artifacts")
    csv_path = out_dir / "pipeline_results_grouped.csv"
    md_path = out_dir / "pipeline_results_grouped.md"
    prov_path = out_dir / "pipeline_results_provenance.json"
    evaluation.emit_grouped_csv(metrics, csv_path)
    evaluation.emit_grouped_md(metrics, md_path)
    evaluation.emit_provenance(metrics, {
        "mode": mode,
        "args": vars(args),
        "d_artifact_paths": {p: {ds: str(d.relative_to(REPO_ROOT)) for ds, d in m.items()} for p, m in d_paths.items()},
        "classifier_sweep": sweep_summary,
        "adaptive_root": str(adaptive_root.relative_to(REPO_ROOT)),
        "corpus_info": corpus_info,
        "d_metrics_per_pipeline": {p: d_metrics_per_pipe[p] for p in d_metrics_per_pipe},
        "adaptive_d_per_dataset": adaptive_d_metrics,
        "adaptive_rag_3way": three_way_info,
        "adaptive_rag_3way_d_per_dataset": adaptive_rag_3way_d,
    }, prov_path)
    coverage = {
        "qa_baseline_present": {pipe: {ds: base_eval_path(pipe, ds).exists() for ds in QA_DATASETS}
                                for pipe in ("nor_qa", "oner_qa", "ircot_qa")},
        "d_artifacts_present": {pipe: {ds: (d_paths[pipe][ds] / "predictions.jsonl").exists() for ds in D_DATASETS}
                                for pipe in ("nor_qa", "oner_qa", "ircot_qa", "global_qa")},
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))

    notes.append("- artifacts:")
    for p in [csv_path, md_path, prov_path, out_dir / "coverage.json", out_dir / "classifier_sweep_summary.json", out_dir / "classifier_corpus_info.json"]:
        notes.append(f"  - {p.relative_to(REPO_ROOT)}")
    notes.append("")
    notes.append("## Caveats and conventions")
    notes.append("- QA columns (trivia/nq/hotpotqa) reuse existing 3-way 500-sample test artifacts.")
    notes.append("- Acc for nq/trivia uses substring-after-normalization (matches evaluate_final_acc.evaluate_by_dicts).")
    notes.append("- Acc for hotpotqa uses official hotpot_evaluate_v1 (`acc` field).")
    notes.append("- LongBench D column uses ROUGE-L (max over references then mean across queries).")
    notes.append("- `longbench` cell is the macro-average of (gov_report, qmsum) for ROUGE-L/Step/Time.")
    notes.append("- Class-D label is deterministically D for any LongBench gov_report/qmsum query.")
    notes.append("- D baseline pipelines operate per-record on the LongBench `context` field "
                 "(no Wikipedia / Elasticsearch retrieval) since LongBench ships its own document.")
    notes.append("- AdaptiveRAG+ adaptive output lives under `predictions/classifier_plus/` "
                 "to avoid colliding with the original 3-way `predictions/classifier/` artifacts.")
    notes.append("- The `adaptive-rag` row uses the existing 3-way classifier:")
    notes.append("    - QA cells reuse `predictions/classifier/.../<sweep>/<best_epoch>/`.")
    notes.append("    - LongBench cell runs the 3-way classifier on the D test slice and routes")
    notes.append("      A/B/C predictions to the corresponding D baseline output (no D path).")
    notes.append("    - Pass `--skip-adaptive-rag-3way` to omit this row (cells become NA).")
    if mode == "smoke":
        notes.append("- This is a SMOKE run with reduced D test slice, classifier sweep, and cluster count.")
    (out_dir / "pipeline_results_notes.md").write_text("\n".join(notes) + "\n")

    if mode == "smoke":
        smoke = [
            "# AdaptiveRAG+ smoke test report",
            f"- mode: smoke",
            f"- epochs swept: {args.epochs}",
            f"- best epoch: {sweep_summary.get('best_epoch')}",
            f"- best valid acc: {sweep_summary.get('best_valid_acc')}",
            f"- d test per dataset: {args.d_test_per_dataset}",
            "",
            "## Coverage",
            json.dumps(coverage, indent=2),
            "",
            "## Output artifacts",
        ]
        for p in [csv_path, md_path, prov_path,
                  out_dir / "pipeline_results_notes.md",
                  out_dir / "classifier_sweep_summary.json",
                  out_dir / "classifier_corpus_info.json",
                  out_dir / "coverage.json"]:
            smoke.append(f"- {p.relative_to(REPO_ROOT)}: exists={p.exists()}")
        smoke.append("")
        smoke.append("## Matrix completeness check")
        ok = True
        for pipe in PIPELINES:
            for ds in DATASET_GROUPS:
                cell = metrics.get(pipe, {}).get(ds, {}) or {}
                subs = contracts.DATASET_GROUP_SUBCOLS[ds]
                missing = [s for s in subs if cell.get(s) is None]
                if missing:
                    ok = False
                smoke.append(f"- {pipe} / {ds}: missing={missing}")
        smoke.append("")
        smoke.append(f"## Acceptance: {'PASS' if ok else 'PARTIAL (see missing)'}")
        (out_dir / "smoke_test_report.md").write_text("\n".join(smoke) + "\n")

    section("Done")
    print(f"Results written under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
