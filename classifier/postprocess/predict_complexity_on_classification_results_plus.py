"""4-way (A/B/C/D) variant of predict_complexity_on_classification_results.

Given a 4-way classifier output and the per-pipeline prediction artifacts
for A (nor_qa), B (oner_qa), C (ircot_qa) and D (global_qa), this script
emits the per-dataset adaptive prediction file consumed by downstream
evaluators.

D pipeline outputs are expected at:
    predictions/test/global_qa_<model>_<dataset>/predictions.jsonl

with rows ``{qid, prediction, ...}`` produced by
``adaptive_rag_plus.global_inference.run_global_pipeline_for_dataset``.
"""

from __future__ import annotations

import argparse
import os

from postprocess_utils_plus import (
    load_json,
    save_prediction_with_classified_label_plus,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    type=str,
    help="model name (matches base_configs naming)",
    choices=("flan_t5_xl", "flan_t5_xxl", "gpt"),
)
parser.add_argument(
    "--classification_result_file",
    type=str,
    required=True,
    help="path to classifier dict_id_pred_results.json",
)
parser.add_argument(
    "--d_datasets",
    type=str,
    nargs="+",
    default=["gov_report", "qmsum"],
    help="LongBench D datasets that have global_qa pipeline outputs",
)
args = parser.parse_args()


if args.model_name == "gpt":
    oner_bm25 = "6"
    ircot_bm25 = "3"
else:
    oner_bm25 = "15"
    ircot_bm25 = "6"


classification_result_file = args.classification_result_file

stepNum_result_file = os.path.join(
    "predictions", "test", f"ircot_qa_{args.model_name}", "total", "stepNum.json"
)

# Adaptive output root mirrors the 3-way layout but lives under
# `predictions/classifier_plus/` to keep the original 3-way artifacts
# untouched and for clean provenance.
parts = classification_result_file.split("/")
after_model = "/".join(parts[parts.index("model") + 1 : -2])
output_path = os.path.join("predictions", "classifier_plus", after_model)


def _abc_files(model_name: str, ds: str) -> dict:
    multi = os.path.join(
        "predictions", "test",
        f"ircot_qa_{model_name}_{ds}____prompt_set_1___bm25_retrieval_count__{ircot_bm25}___distractor_count__1",
        f"prediction__{ds}_to_{ds}__test_subsampled.json",
    )
    one = os.path.join(
        "predictions", "test",
        f"oner_qa_{model_name}_{ds}____prompt_set_1___bm25_retrieval_count__{oner_bm25}___distractor_count__1",
        f"prediction__{ds}_to_{ds}__test_subsampled.json",
    )
    zero = os.path.join(
        "predictions", "test",
        f"nor_qa_{model_name}_{ds}____prompt_set_1",
        f"prediction__{ds}_to_{ds}__test_subsampled.json",
    )
    return {"C": multi, "B": one, "A": zero}


# Existing 3-way QA datasets (unchanged routing)
QA_DATASETS = ("musique", "hotpotqa", "2wikimultihopqa", "nq", "trivia", "squad")
dataName_to_multi_one_zero_file = {ds: _abc_files(args.model_name, ds) for ds in QA_DATASETS}

# D datasets: even though the deterministic class-D label is `D`, the
# 4-way classifier may emit A/B/C for some D queries (especially in short
# smoke runs). Mirror the 3-way row's policy: route the misclassification
# to the corresponding D baseline pipeline output so we can still score a
# concrete answer for that qid (no missing rows). All four pipelines'
# JSONL outputs live next to each other under
# `predictions/test/<pipeline>_gpt_<dataset>/predictions.jsonl`.
def _d_pipeline_jsonl(model_name: str, ds: str, pipeline: str) -> str:
    return os.path.join(
        "predictions", "test", f"{pipeline}_{model_name}_{ds}", "predictions.jsonl"
    )


for ds in args.d_datasets:
    dataName_to_multi_one_zero_file[ds] = {
        "A": _d_pipeline_jsonl(args.model_name, ds, "nor_qa"),
        "B": _d_pipeline_jsonl(args.model_name, ds, "oner_qa"),
        "C": _d_pipeline_jsonl(args.model_name, ds, "ircot_qa"),
    }


def _d_predictions_path(model_name: str, ds: str) -> str:
    return os.path.join(
        "predictions", "test", f"global_qa_{model_name}_{ds}", "predictions.jsonl"
    )


d_predictions_file_by_dataset = {
    ds: _d_predictions_path(args.model_name, ds) for ds in args.d_datasets
}


total_qid_to_classification_pred = load_json(classification_result_file)

for data_name in dataName_to_multi_one_zero_file.keys():
    save_prediction_with_classified_label_plus(
        total_qid_to_classification_pred,
        data_name,
        stepNum_result_file,
        dataName_to_multi_one_zero_file,
        d_predictions_file_by_dataset,
        output_path,
    )
