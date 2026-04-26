"""4-way (A/B/C/D) variants of the classifier postprocess helpers.

D-routed predictions live in ``predictions/test/global_qa_<model>_<dataset>/``
as a JSONL file (``predictions.jsonl``) produced by
``adaptive_rag_plus.global_inference.run_global_pipeline_for_dataset``.
We index that file by qid and read each row's ``prediction`` field for the
adaptive output assembly.
"""

from __future__ import annotations

import json
import os
from typing import Dict


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)
    print(path)


def load_jsonl_qid_to_prediction(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid"))
            out[qid] = row.get("prediction", "")
    return out


def load_jsonl_qid_to_step(path: str) -> Dict[str, float]:
    """Read a D pipeline's traces.jsonl into qid -> step_count.

    Used so a D-dataset query routed to ircot (C) by the 4-way classifier
    gets a real per-qid step count from the D ircot baseline trace,
    rather than relying on `predictions/test/ircot_qa_<model>/total/stepNum.json`
    which only covers the QA datasets.
    """
    out: Dict[str, float] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row.get("qid"))] = float(row.get("step_count", 0.0))
    return out


def save_prediction_with_classified_label_plus(
    total_qid_to_classification_pred: dict,
    dataset_name: str,
    stepNum_result_file: str,
    dataName_to_multi_one_zero_file: dict,
    d_predictions_file_by_dataset: dict,
    output_path: str,
):
    """Assemble per-dataset adaptive predictions for the 4-way router.

    For each qid:
      * A -> nor pipeline answer (existing pred file)
      * B -> oner pipeline answer (existing pred file), step=1
      * C -> ircot pipeline answer (existing pred file), step from stepNum.json
              (or from traces.jsonl when the dataset is a D dataset)
      * D -> global pipeline answer (predictions.jsonl), step=1

    For a class-D dataset (gov_report, qmsum) the entries in
    ``dataName_to_multi_one_zero_file[dataset_name]`` point at
    ``predictions/test/<pipeline>_<model>_<dataset>/predictions.jsonl``,
    so an A/B/C misclassification still resolves to a concrete answer
    from the corresponding D baseline pipeline (same policy as the
    original 3-way `adaptive-rag` row on D).
    """
    qid_to_classification_pred = {}
    qid_to_classification_pred_option = {}
    total_stepNum = 0

    d_pred_lookup_cache: Dict[str, Dict[str, str]] = {}
    abc_lookup_cache: Dict[str, Dict[str, str]] = {}
    d_step_cache: Dict[str, Dict[str, float]] = {}
    is_d_dataset = dataset_name in d_predictions_file_by_dataset

    def _load_pred_file(path: str) -> Dict[str, str]:
        if path in abc_lookup_cache:
            return abc_lookup_cache[path]
        if path.endswith(".jsonl"):
            data = load_jsonl_qid_to_prediction(path)
        else:
            data = load_json(path)
        abc_lookup_cache[path] = data
        return data

    for qid, info in total_qid_to_classification_pred.items():
        if dataset_name != info.get("dataset_name"):
            continue

        predicted_option = info.get("prediction")

        if predicted_option == "C":
            if is_d_dataset:
                # Read step count from the D ircot traces file (sibling
                # of the predictions.jsonl referenced for "C").
                ircot_pred_path = dataName_to_multi_one_zero_file[dataset_name].get("C")
                if ircot_pred_path:
                    traces_path = os.path.join(os.path.dirname(ircot_pred_path), "traces.jsonl")
                    if traces_path not in d_step_cache:
                        d_step_cache[traces_path] = load_jsonl_qid_to_step(traces_path)
                    stepNum = int(round(d_step_cache[traces_path].get(qid, 0.0)))
                else:
                    stepNum = 0
            else:
                stepNum = load_json(stepNum_result_file)[qid]
        elif predicted_option == "B":
            stepNum = 1
        elif predicted_option == "A":
            stepNum = 0
        elif predicted_option == "D":
            stepNum = 1
        else:
            continue

        if predicted_option in ("A", "B", "C"):
            abc_path = dataName_to_multi_one_zero_file[dataset_name].get(predicted_option)
            if not abc_path:
                # No mapping (e.g. legacy None entry). Skip rather than crash.
                continue
            pred_lookup = _load_pred_file(abc_path)
            pred = pred_lookup.get(qid, "")
        else:  # D
            d_path = d_predictions_file_by_dataset.get(dataset_name)
            if not d_path:
                # Classifier predicted D for a non-D QA dataset. Remap to the
                # ircot (C) branch as the safest fallback (most capable baseline)
                # so we never lose a prediction for this qid.
                import warnings
                warnings.warn(
                    f"Classifier predicted D for non-D dataset {dataset_name!r} "
                    f"(qid={qid}); remapping to C (ircot)."
                )
                fallback_path = dataName_to_multi_one_zero_file[dataset_name].get("C")
                if not fallback_path:
                    continue
                pred_lookup = _load_pred_file(fallback_path)
                pred = pred_lookup.get(qid, "")
                if predicted_option == "D":
                    stepNum = load_json(stepNum_result_file).get(qid, 0)
                qid_to_classification_pred[qid] = pred
                qid_to_classification_pred_option[qid] = {
                    "prediction": pred,
                    "option": "C",  # remapped
                    "stepNum": stepNum,
                }
                total_stepNum += stepNum
                continue
            if d_path not in d_pred_lookup_cache:
                d_pred_lookup_cache[d_path] = load_jsonl_qid_to_prediction(d_path)
            pred = d_pred_lookup_cache[d_path].get(qid, "")

        qid_to_classification_pred[qid] = pred
        qid_to_classification_pred_option[qid] = {
            "prediction": pred,
            "option": predicted_option,
            "stepNum": stepNum,
        }
        total_stepNum += stepNum

    print("==============")
    save_json(os.path.join(output_path, dataset_name, dataset_name + ".json"), qid_to_classification_pred)
    save_json(os.path.join(output_path, dataset_name, dataset_name + "_option.json"), qid_to_classification_pred_option)
    print("StepNum")
    print(f"{dataset_name}: {total_stepNum}")
