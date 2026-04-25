#!/usr/bin/env python
"""Orchestration script for the 4-pipeline x 3-dataset (GPT) Adaptive-RAG report.

Pipelines (rows): nor_qa, oner_qa, ircot_qa, adaptive-rag
Datasets (cols):  trivia, nq, hotpotqa
Backend:          GPT (gpt-3.5-turbo-instruct, configured in base_configs/*_gpt_*.jsonnet)

Modes:
  --smoke-test  mini sweep (epochs 1,2) and skips base pipeline runs when artifacts exist
  --full-run    paper-aligned sweep (epochs 10,15,20,25); will execute missing base runs

Outputs (under --out-dir, default ./reports/<timestamp>):
  pipeline_results_grouped.csv
  pipeline_results_grouped.md
  pipeline_results_provenance.json
  pipeline_results_notes.md
  smoke_test_report.md (smoke-test mode only)
  classifier_sweep_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import string
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINES = ["nor_qa", "oner_qa", "ircot_qa", "adaptive-rag"]
DATASETS = ["trivia", "nq", "hotpotqa"]
GPT_BM25 = {"oner_qa": 6, "ircot_qa": 3}
PROMPT_SET = 1
DISTRACTOR_COUNT = 1


# ----------------- Pretty logging ----------------- #

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def section(msg: str) -> None:
    bar = "=" * (len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)


# ----------------- Filesystem helpers ----------------- #

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


def base_prediction_path(pipeline: str, dataset: str) -> Path:
    return base_pred_dir(pipeline, dataset) / f"prediction__{dataset}_to_{dataset}__test_subsampled.json"


def base_time_path(pipeline: str, dataset: str) -> Path:
    return base_pred_dir(pipeline, dataset) / f"prediction__{dataset}_to_{dataset}__test_subsampled_time_taken.txt"


def base_step_path(pipeline: str, dataset: str) -> Optional[Path]:
    if pipeline != "ircot_qa":
        return None
    return base_pred_dir(pipeline, dataset) / "stepNum.json"


def processed_test_path(dataset: str) -> Path:
    return REPO_ROOT / "processed_data" / dataset / "test_subsampled.jsonl"


# ----------------- Subprocess helpers ----------------- #

def run(cmd: List[str], cwd: Optional[Path] = None, env_overrides: Optional[dict] = None,
        check: bool = True) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


# ----------------- Coverage / preflight ----------------- #

def preflight(args) -> Dict[str, dict]:
    section("Preflight coverage audit")
    coverage = {}
    for pipe in ["nor_qa", "oner_qa", "ircot_qa"]:
        for ds in DATASETS:
            ev = base_eval_path(pipe, ds)
            tm = base_time_path(pipe, ds)
            pr = base_prediction_path(pipe, ds)
            present = ev.exists() and tm.exists() and pr.exists()
            log(f"  {pipe:9s} {ds:9s} {'OK' if present else 'MISSING'}: {ev.relative_to(REPO_ROOT)}")
            coverage[f"{pipe}|{ds}"] = {"present": present, "eval": str(ev), "time": str(tm), "pred": str(pr)}

    hotpot_raw = REPO_ROOT / "raw_data" / "hotpotqa" / "hotpot_dev_distractor_v1.json"
    log(f"  hotpot raw distractor: {'OK' if hotpot_raw.exists() else 'MISSING'} ({hotpot_raw.relative_to(REPO_ROOT)})")
    coverage["raw_data_hotpotqa"] = {"present": hotpot_raw.exists(), "path": str(hotpot_raw)}

    for ds in DATASETS:
        proc = processed_test_path(ds)
        ok = proc.exists()
        log(f"  processed test {ds}: {'OK' if ok else 'MISSING'} ({proc.relative_to(REPO_ROOT)})")
        coverage[f"processed|{ds}"] = {"present": ok, "path": str(proc)}

    return coverage


# ----------------- Base pipeline runs (only if missing) ----------------- #

def run_base_pipeline(pipe: str, ds: str, llm_port: int) -> None:
    log(f"Running base pipeline {pipe} on {ds} via run_retrieval_test.sh ...")
    run(["bash", "run_retrieval_test.sh", pipe, "gpt", ds, str(llm_port)], cwd=REPO_ROOT)


def ensure_base_runs(coverage: Dict[str, dict], allow_run: bool, llm_port: int) -> List[str]:
    section("Base pipeline completion (only missing runs)")
    ran = []
    for pipe in ["nor_qa", "oner_qa", "ircot_qa"]:
        for ds in DATASETS:
            key = f"{pipe}|{ds}"
            if coverage[key]["present"]:
                log(f"  reuse: {pipe} / {ds}")
                continue
            if not allow_run:
                log(f"  SKIP run (smoke or skip-existing): {pipe} / {ds}")
                continue
            run_base_pipeline(pipe, ds, llm_port)
            coverage[key]["present"] = True
            ran.append(key)
    return ran


# ----------------- Acc computation ----------------- #

# Re-implement the exact normalization + acc + SquadAnswerEmF1 logic that
# evaluate_final_acc.py uses, but applied directly to base prediction files.

_NORM_REGEX_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _NORM_REGEX_ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def _answer_extract(potentially_cot: str) -> str:
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


def _calc_acc(prediction: str, ground_truths: List[str]) -> int:
    p = _normalize_answer(prediction)
    for gt in ground_truths:
        if _normalize_answer(gt) in p:
            return 1
    return 0


def _load_id_to_gt(processed_path: Path) -> Dict[str, List[str]]:
    out = {}
    with processed_path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["question_id"]] = d["answers_objects"][0]["spans"]
    return out


def compute_acc_for_predictions(prediction_file: Path, dataset: str) -> Tuple[float, int]:
    """Compute Acc (substring-after-normalization) for any prediction file.

    For hotpotqa we additionally invoke the official hotpot_evaluate_v1.py to
    obtain the official `acc`. For nq/trivia we use the same acc definition as
    evaluate_final_acc.evaluate_by_dicts.
    """
    if dataset == "hotpotqa":
        return _compute_acc_hotpot_official(prediction_file)
    return _compute_acc_substring(prediction_file, dataset)


def _compute_acc_substring(prediction_file: Path, dataset: str) -> Tuple[float, int]:
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
        pred = _answer_extract(str(pred))
        correct += _calc_acc(pred, gt)
        total += 1
    return (correct / total if total else 0.0), total


def _compute_acc_hotpot_official(prediction_file: Path) -> Tuple[float, int]:
    """Wrap the official HotpotQA evaluator (which reports its own acc field)."""
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

    metrics = eval(out_path.read_text().strip())  # output is a python dict literal
    gt_path.unlink(missing_ok=True)
    pr_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return float(metrics["acc"]), len(id_to_pred)


# ----------------- Metric assembly for base rows ----------------- #

def collect_base_metrics() -> Dict[str, Dict[str, dict]]:
    section("Collect base pipeline metrics (EM/F1/Acc/Step/Time)")
    out: Dict[str, Dict[str, dict]] = {}
    for pipe in ["nor_qa", "oner_qa", "ircot_qa"]:
        out[pipe] = {}
        for ds in DATASETS:
            ev_path = base_eval_path(pipe, ds)
            tm_path = base_time_path(pipe, ds)
            pr_path = base_prediction_path(pipe, ds)
            ev = json.loads(ev_path.read_text())
            total_t = float(tm_path.read_text().strip())
            n = ev.get("count") or 0
            avg_t = total_t / n if n else None
            if pipe == "ircot_qa":
                sn = json.loads(base_step_path(pipe, ds).read_text())
                avg_step = sum(sn.values()) / len(sn)
            elif pipe == "oner_qa":
                avg_step = 1.0
            else:
                avg_step = 0.0
            log(f"  {pipe}/{ds}: computing Acc ...")
            acc, _ = compute_acc_for_predictions(pr_path, ds)
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
            log(f"    EM={out[pipe][ds]['EM']:.3f} F1={out[pipe][ds]['F1']:.3f} "
                f"Acc={out[pipe][ds]['Acc']:.3f} Step={out[pipe][ds]['Step']:.2f} "
                f"Time/q={avg_t:.3f}s")
    return out


# ----------------- Classifier epoch sweep ----------------- #

def classifier_sweep(epochs: List[int], llm_name: str = "flan_t5_xl",
                     dataset_name: str = "musique_hotpot_wiki2_nq_tqa_sqd",
                     base_model: str = "t5-small", batch: int = 8) -> Tuple[Path, Path, dict]:
    section(f"Classifier epoch sweep: {epochs} (base={base_model}, llm_label={llm_name})")
    classifier_dir = REPO_ROOT / "classifier"
    sweep_root = classifier_dir / "outputs" / dataset_name / "model" / base_model / llm_name / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    sweep_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    sweep_dir = sweep_root / sweep_id
    sweep_dir.mkdir()

    train_file = f"./data/{dataset_name}/{llm_name}/binary_silver/train.json"
    valid_file = f"./data/{dataset_name}/{llm_name}/silver/valid.json"
    predict_file = f"./data/{dataset_name}/predict.json"

    summary = {"sweep_dir": str(sweep_dir.relative_to(REPO_ROOT)), "runs": []}
    for ep in epochs:
        ep_out = sweep_dir / f"epoch_{ep}"
        ep_out.mkdir(parents=True, exist_ok=True)
        log(f"  >>> training epoch={ep} -> {ep_out.relative_to(REPO_ROOT)}")
        t0 = time.time()
        run([
            "python", "run_classifier.py",
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
            "python", "run_classifier.py",
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
            "python", "run_classifier.py",
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

        run_info = {
            "epoch": ep,
            "train_seconds": train_secs,
            "checkpoint_dir": str(ep_out.relative_to(REPO_ROOT)),
            "valid_dir": str(valid_dir.relative_to(REPO_ROOT)),
            "predict_dir": str(predict_dir.relative_to(REPO_ROOT)),
            "valid_final_acc_score": valid_acc,
        }
        summary["runs"].append(run_info)
        log(f"  <<< epoch={ep} valid_final_acc_score={valid_acc}")

    runs_with_metric = [r for r in summary["runs"] if r["valid_final_acc_score"] is not None]
    if not runs_with_metric:
        raise RuntimeError("No classifier sweep run produced a validation metric")
    best = max(runs_with_metric, key=lambda r: r["valid_final_acc_score"])
    summary["best_epoch"] = best["epoch"]
    summary["best_valid_acc"] = best["valid_final_acc_score"]
    log(f"  *** best epoch={best['epoch']} valid_final_acc_score={best['valid_final_acc_score']}")

    best_predict_results = REPO_ROOT / best["predict_dir"] / "dict_id_pred_results.json"
    return Path(best["checkpoint_dir"]), best_predict_results, summary


# ----------------- Adaptive postprocess + final eval ----------------- #

def run_adaptive(best_predict_results: Path) -> Tuple[Path, dict]:
    section("Adaptive postprocess + final evaluation")

    rel_predict_results = best_predict_results.relative_to(REPO_ROOT)
    run([
        "python", "classifier/postprocess/predict_complexity_on_classification_results.py",
        "gpt",
        "--classification_result_file", str(REPO_ROOT / rel_predict_results),
    ], cwd=REPO_ROOT)

    parts = rel_predict_results.parts
    idx = parts.index("model")
    after_model = parts[idx + 1:-2]
    adaptive_root = REPO_ROOT / "predictions" / "classifier" / Path(*after_model)
    log(f"  adaptive output root: {adaptive_root.relative_to(REPO_ROOT)}")

    run(["python", "evaluate_final_acc.py"], cwd=REPO_ROOT,
        env_overrides={"ADAPTIVE_BASE_PRED_PATH": str(adaptive_root) + os.sep})

    metrics = {}
    for ds in DATASETS:
        eval_file = adaptive_root / ds / "eval_metic_result_acc.json"
        opt_file = adaptive_root / ds / f"{ds}_option.json"
        if not eval_file.exists():
            log(f"  WARNING: missing adaptive eval file for {ds}: {eval_file}")
            metrics[ds] = {"EM": None, "F1": None, "Acc": None, "Step": None, "Time": None,
                            "_provenance": {"eval_file": str(eval_file.relative_to(REPO_ROOT)),
                                            "option_file": str(opt_file.relative_to(REPO_ROOT))}}
            continue
        m = json.loads(eval_file.read_text())
        opts = json.loads(opt_file.read_text()) if opt_file.exists() else {}

        total_step = 0.0
        total_time = 0.0
        n = 0
        for qid, info in opts.items():
            opt = info.get("option")
            step = info.get("stepNum", 0)
            total_step += float(step)
            if opt == "A":
                src_pipe = "nor_qa"
            elif opt == "B":
                src_pipe = "oner_qa"
            elif opt == "C":
                src_pipe = "ircot_qa"
            else:
                continue
            t_path = base_time_path(src_pipe, ds)
            ev_path = base_eval_path(src_pipe, ds)
            try:
                total_t = float(t_path.read_text().strip())
                cnt = json.loads(ev_path.read_text()).get("count") or 0
                if cnt:
                    total_time += total_t / cnt
            except FileNotFoundError:
                pass
            n += 1
        avg_step = total_step / n if n else None
        avg_time = total_time / n if n else None

        metrics[ds] = {
            "EM": m.get("em"),
            "F1": m.get("f1"),
            "Acc": m.get("acc"),
            "Step": avg_step,
            "Time": avg_time,
            "count": m.get("count"),
            "_provenance": {
                "eval_file": str(eval_file.relative_to(REPO_ROOT)),
                "option_file": str(opt_file.relative_to(REPO_ROOT)),
                "step_time_source": "weighted from base pipeline outputs by routed option",
            },
        }
        log(f"  adaptive/{ds}: EM={metrics[ds]['EM']} F1={metrics[ds]['F1']} "
            f"Acc={metrics[ds]['Acc']} Step={avg_step} Time/q={avg_time}")
    return adaptive_root, metrics


# ----------------- Output emitters ----------------- #

def fmt(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


SUBCOLS = ["EM", "F1", "Acc", "Step", "Time"]


def emit_csv(metrics: dict, out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        header1 = ["pipeline"] + sum(([ds] + [""] * (len(SUBCOLS) - 1) for ds in DATASETS), [])
        header2 = [""] + sum(([s for s in SUBCOLS] for _ in DATASETS), [])
        w.writerow(header1)
        w.writerow(header2)
        for pipe in PIPELINES:
            row = [pipe]
            for ds in DATASETS:
                cell = metrics.get(pipe, {}).get(ds, {})
                for s in SUBCOLS:
                    row.append(fmt(cell.get(s)))
            w.writerow(row)


def emit_md(metrics: dict, out_path: Path) -> None:
    lines = []
    top = "| pipeline | " + " | ".join(f"{ds} ({'/'.join(SUBCOLS)})" for ds in DATASETS) + " |"
    sep = "|" + "---|" * (1 + len(DATASETS))
    lines.append(top)
    lines.append(sep)
    for pipe in PIPELINES:
        cells = [pipe]
        for ds in DATASETS:
            cell = metrics.get(pipe, {}).get(ds, {})
            sub = " / ".join(fmt(cell.get(s)) for s in SUBCOLS)
            cells.append(sub)
        lines.append("| " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def emit_provenance(metrics: dict, out_path: Path) -> None:
    prov = {}
    for pipe in PIPELINES:
        prov[pipe] = {}
        for ds in DATASETS:
            cell = metrics.get(pipe, {}).get(ds, {})
            prov[pipe][ds] = {
                "metrics": {s: cell.get(s) for s in SUBCOLS},
                "count": cell.get("count"),
                "provenance": cell.get("_provenance"),
            }
    out_path.write_text(json.dumps(prov, indent=2, default=str))


def emit_notes(notes: List[str], out_path: Path) -> None:
    out_path.write_text("\n".join(notes) + "\n")


# ----------------- Driver ----------------- #

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="mini sweep + reuse base predictions when available")
    parser.add_argument("--full-run", action="store_true",
                        help="full sweep + run any missing base predictions")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="reuse existing base prediction artifacts when present (default true)")
    parser.add_argument("--llm-port", type=int, default=8010,
                        help="port for runner.py LLM server (only used when running base pipelines)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory; defaults to ./reports/<timestamp>_<mode>")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                        help="override epoch sweep list")
    parser.add_argument("--skip-classifier-sweep", action="store_true",
                        help="reuse existing classifier predict file (must pass --classifier-predict-file)")
    parser.add_argument("--classifier-predict-file", type=str, default=None,
                        help="path to dict_id_pred_results.json from a prior sweep best epoch")
    args = parser.parse_args()

    if args.smoke_test == args.full_run:
        parser.error("Pick exactly one of --smoke-test or --full-run")
    mode = "smoke" if args.smoke_test else "full"

    if args.epochs is None:
        args.epochs = [1, 2] if mode == "smoke" else [10, 15, 20, 25]

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    section(f"Adaptive-RAG 4x3 orchestration ({mode}) -> {out_dir}")

    notes: List[str] = [f"# Pipeline run notes ({mode})", f"- timestamp: {datetime.now().isoformat()}", f"- out_dir: {out_dir}"]

    coverage = preflight(args)
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))

    allow_base_runs = (mode == "full")
    ran = ensure_base_runs(coverage, allow_run=allow_base_runs, llm_port=args.llm_port)
    notes.append(f"- base pipeline runs executed this session: {ran or 'none (reused existing)'}")

    base_metrics = collect_base_metrics()

    if args.skip_classifier_sweep:
        if not args.classifier_predict_file:
            raise SystemExit("--skip-classifier-sweep requires --classifier-predict-file")
        best_predict = Path(args.classifier_predict_file).resolve()
        if not best_predict.exists():
            raise SystemExit(f"Missing classifier predict file: {best_predict}")
        sweep_summary = {"reused_predict_file": str(best_predict.relative_to(REPO_ROOT))}
    else:
        _, best_predict, sweep_summary = classifier_sweep(args.epochs)

    (out_dir / "classifier_sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2, default=str))
    notes.append(f"- classifier sweep epochs: {args.epochs}")
    notes.append(f"- best classifier predict file: {best_predict}")

    adaptive_root, adaptive_metrics = run_adaptive(best_predict)
    notes.append(f"- adaptive output root: {adaptive_root.relative_to(REPO_ROOT)}")

    metrics = {**base_metrics, "adaptive-rag": adaptive_metrics}

    csv_path = out_dir / "pipeline_results_grouped.csv"
    md_path = out_dir / "pipeline_results_grouped.md"
    prov_path = out_dir / "pipeline_results_provenance.json"
    emit_csv(metrics, csv_path)
    emit_md(metrics, md_path)
    emit_provenance(metrics, prov_path)
    notes.append("- artifacts:")
    for p in [csv_path, md_path, prov_path]:
        notes.append(f"  - {p.relative_to(REPO_ROOT)}")

    notes.append("")
    notes.append("## Caveats")
    notes.append("- Base pipeline rows (nor_qa, oner_qa, ircot_qa) reuse existing 500-sample test artifacts.")
    notes.append("- Acc for nq/trivia uses substring-after-normalization (matches evaluate_final_acc.evaluate_by_dicts).")
    notes.append("- Acc for hotpotqa uses official hotpot_evaluate_v1 (`acc` field).")
    notes.append("- Adaptive-rag Step is the average routed step across qids; Time is the average per-question time of the chosen base pipeline.")
    if mode == "smoke":
        notes.append("- This is a SMOKE run with mini classifier sweep; rerun with --full-run for paper-aligned numbers.")
    emit_notes(notes, out_dir / "pipeline_results_notes.md")

    if mode == "smoke":
        smoke = ["# Smoke test report",
                 f"- mode: smoke",
                 f"- epochs swept: {args.epochs}",
                 f"- best epoch: {sweep_summary.get('best_epoch')}",
                 f"- best valid acc: {sweep_summary.get('best_valid_acc')}",
                 "",
                 "## Coverage",
                 ]
        for k, v in coverage.items():
            smoke.append(f"- {k}: present={v.get('present')}")
        smoke.append("")
        smoke.append("## Output artifacts")
        for p in [csv_path, md_path, prov_path, out_dir / "pipeline_results_notes.md", out_dir / "classifier_sweep_summary.json", out_dir / "coverage.json"]:
            smoke.append(f"- {p.relative_to(REPO_ROOT)}: exists={p.exists()}")
        smoke.append("")
        smoke.append("## Matrix completeness check")
        ok = True
        for pipe in PIPELINES:
            for ds in DATASETS:
                cell = metrics.get(pipe, {}).get(ds, {})
                missing = [s for s in SUBCOLS if cell.get(s) is None]
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
