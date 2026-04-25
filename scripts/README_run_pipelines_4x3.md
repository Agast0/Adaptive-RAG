# 4 Pipelines x 3 Datasets (GPT) Orchestration

This script orchestrates the full Adaptive-RAG comparison reported as a 4x3 grouped table:

- Rows (pipelines): `nor_qa`, `oner_qa`, `ircot_qa`, `adaptive-rag`
- Columns (datasets): `trivia`, `nq`, `hotpotqa`
- Subcolumns per cell: `EM | F1 | Acc | Step | Time`

Backend: GPT (`gpt-3.5-turbo-instruct`, hard-coded in `base_configs/*_gpt_*.jsonnet`).

## Prerequisites

1. Activate the project virtualenv: `source .venv/bin/activate` (Python 3.10).
2. `OPENAI_API_KEY` must be set, either as an env var or in `.env` at the repo root.
3. If you intend to run any missing base pipelines (`--full-run`), Elasticsearch and the FastAPI retriever server must be reachable on the port you pass via `--llm-port` (the default is `8010` to match the existing experiments). Existing 500-sample base predictions are reused by default; this is the only reason ES is otherwise unnecessary.
4. `raw_data/hotpotqa/hotpot_dev_distractor_v1.json` must exist (downloaded once via `download/raw_data.sh` or `curl -L http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -o raw_data/hotpotqa/hotpot_dev_distractor_v1.json`).
5. Classifier training data must be present under `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{predict.json, gpt/binary_silver/train.json, gpt/silver/valid.json}` and the labels file `flan_t5_xl/binary_silver/train.json` (used by the classifier base; the `gpt`-flavored data is consumed at adaptive postprocess time, not at training time).

## Modes

| flag             | classifier sweep epochs | base pipeline runs |
|------------------|-------------------------|--------------------|
| `--smoke-test`   | `1, 2`                  | reuse existing only |
| `--full-run`     | `10, 15, 20, 25`        | runs missing combos |

Override the sweep with `--epochs E1 E2 ...` (e.g. `--epochs 5 8`).

You may skip the sweep entirely (e.g. when iterating on output formatting) with:

```bash
python scripts/run_pipelines_4x3.py --smoke-test \
  --skip-classifier-sweep \
  --classifier-predict-file classifier/outputs/.../predict/dict_id_pred_results.json
```

## Smoke test (recommended first run)

```bash
source .venv/bin/activate
python scripts/run_pipelines_4x3.py --smoke-test
```

This will:

1. Audit existing base prediction artifacts for all 9 base cells and HotpotQA raw data.
2. Reuse the existing 500-sample base predictions instead of re-running them.
3. Train `t5-small` for epochs `1, 2` on the binary+silver classifier data, picking the best by validation `final_acc_score`.
4. Run `classifier/postprocess/predict_complexity_on_classification_results.py gpt` against the best classifier predict file.
5. Run `evaluate_final_acc.py` with `ADAPTIVE_BASE_PRED_PATH` pointing at the new adaptive output root.
6. Emit:
   - `reports/<ts>_smoke/pipeline_results_grouped.csv`
   - `reports/<ts>_smoke/pipeline_results_grouped.md`
   - `reports/<ts>_smoke/pipeline_results_provenance.json`
   - `reports/<ts>_smoke/pipeline_results_notes.md`
   - `reports/<ts>_smoke/classifier_sweep_summary.json`
   - `reports/<ts>_smoke/coverage.json`
   - `reports/<ts>_smoke/smoke_test_report.md`

The smoke run validates that **all 12 cells are populated** end-to-end, and that the adaptive row is produced from a sweep-selected checkpoint.

## Full run (paper-aligned)

```bash
source .venv/bin/activate
python scripts/run_pipelines_4x3.py --full-run
```

What changes vs smoke:

- Classifier sweep covers `10, 15, 20, 25` epochs (~few minutes per epoch on a Mac CPU; expect ~20-30 minutes total for `t5-small`).
- If any of the 9 base prediction directories is missing, `run_retrieval_test.sh` will be invoked for that combination. **This requires a running ES + retriever + LLM stack**, will spend OpenAI tokens, and can take ~5-25 min per pipeline-dataset cell.

You can resume after a failure by simply re-running with `--full-run`; existing artifacts are not re-computed (the script always reuses the per-cell base prediction directory if both `evaluation_metrics_*.json` and `prediction_*.json` exist).

## Output schema

`pipeline_results_grouped.csv` shape:

| pipeline    | trivia (EM,F1,Acc,Step,Time) | nq (EM,F1,Acc,Step,Time) | hotpotqa (EM,F1,Acc,Step,Time) |
|-------------|------------------------------|--------------------------|--------------------------------|
| nor_qa      | ...                          | ...                      | ...                            |
| oner_qa     | ...                          | ...                      | ...                            |
| ircot_qa    | ...                          | ...                      | ...                            |
| adaptive-rag| ...                          | ...                      | ...                            |

Metric definitions:

- `EM` / `F1`: from `predictions/test/<pipe>_<dataset>_*/evaluation_metrics_*.json` for base rows; from official hotpot evaluator / SQuAD-style eval for adaptive row (matches `evaluate_final_acc.py`).
- `Acc`:
  - `nq`, `trivia` (base + adaptive): substring-after-normalization (`gt in normalize(prediction)`), as in `evaluate_final_acc.evaluate_by_dicts`.
  - `hotpotqa` (base + adaptive): official `acc` field from `official_evaluation/hotpotqa/hotpot_evaluate_v1.py`.
- `Step`:
  - `nor_qa`: 0
  - `oner_qa`: 1
  - `ircot_qa`: average over `stepNum.json` for that combo
  - `adaptive-rag`: average routed step (0 for A-routed, 1 for B-routed, ircot stepNum for C-routed)
- `Time`: average per-question wall-clock seconds. Base: `prediction_time_taken.txt / count`. Adaptive: weighted average of base per-q time using the routed option per qid.

## Troubleshooting

- **Missing OpenAI key** during base runs: ensure `.env` is present (`OPENAI_API_KEY=sk-...`).
- **ES not running** during base runs: `docker ps` and `bash setup.sh` (or your existing ES + retriever workflow) before invoking with `--full-run` on a missing combo.
- **`Acc=NA` for hotpotqa**: confirm `raw_data/hotpotqa/hotpot_dev_distractor_v1.json` exists (~44 MB).
- **`final_acc_score` missing in sweep**: usually means classifier validation failed; inspect `classifier/outputs/.../valid/logs.log`. The script will hard-fail if no run produces a metric.
- **`accelerate` device errors on Mac**: this script does not pass `CUDA_VISIBLE_DEVICES` and the classifier uses HF `accelerate` for placement; CPU is the default on Apple silicon.

## Handoff

Once the smoke test passes, run:

```bash
source .venv/bin/activate
python scripts/run_pipelines_4x3.py --full-run
```

The final 4x3 table for hand-in is in `reports/<full-run-ts>/pipeline_results_grouped.md` and `pipeline_results_grouped.csv`.
