# `run_pipelines_4x4_plus.py` — AdaptiveRAG+ orchestrator

Runs the full AdaptiveRAG+ experiment end-to-end and emits a single
5-pipeline x 4-dataset-group results table with task-specific subcolumns.

| pipeline (rows) | dataset groups (cols) | per-cell subcolumns |
| --- | --- | --- |
| `nor_qa`, `oner_qa`, `ircot_qa`, `adaptive-rag`, `adaptive-rag+` | `trivia`, `nq`, `hotpotqa`, `longbench` | QA: `EM`/`F1`/`Acc`/`Step`/`Time` ; LongBench: `ROUGE-L`/`Step`/`Time` |

`longbench` is the macro-average of LongBench `gov_report` and `qmsum`
(class D). Each baseline pipeline (`nor_qa`, `oner_qa`, `ircot_qa`) is
evaluated on both D datasets independently before aggregation, so the
final table is fully comparable across all 5 pipelines.

The two adaptive rows differ only in their classifier:

* **`adaptive-rag`** — original 3-way classifier (A/B/C). For QA columns
  it reuses the existing `predictions/classifier/.../<sweep>/<best_epoch>/`
  artifacts. For the `longbench` column the orchestrator runs the 3-way
  classifier on the LongBench D test slice; since it has no D label,
  every D query is routed to A/B/C and answered by the corresponding D
  baseline output.
* **`adaptive-rag+`** — 4-way classifier (A/B/C/D). The D class is
  routed to the global retrieval pipeline (`global_qa`).

## Prerequisites

1. The 3-way (Adaptive-RAG) artifacts must exist for `trivia`, `nq`,
   and `hotpotqa`. If they do not, run
   `scripts/run_pipelines_4x3.py --full-run` first (see
   `scripts/README_run_pipelines_4x3.md`).
2. `OPENAI_API_KEY` is exported (or available in `.env`) for the GPT
   backend (`gpt-3.5-turbo-instruct`) and embeddings
   (`text-embedding-3-small`). All OpenAI calls go through a bounded
   exponential-backoff retry that respects `Retry-After` headers.
3. Python deps installed in `.venv` (the existing repo venv) plus
   `rouge-score` and `scikit-learn` (auto-installed during AdaptiveRAG+
   bring-up; see `requirements.txt`).
4. Free disk for embedding caches under `cache/longbench_index/` and
   raw LongBench JSONL under `raw_data/longbench/`. Combined < 50 MB.
5. **The LongBench class-D global indexes must be built first.**
   Indexing is now a separate, fully resumable step (see below). The
   orchestrator's Phase 0 aborts with a clear message if either
   `gov_report` or `qmsum` index is missing.

## Two-step workflow

### Step 1 — Build the LongBench global indexes (resumable)

```bash
# Default: index BOTH gov_report and qmsum (200 records each), resume
# from last checkpoint if a previous run was interrupted.
python scripts/build_longbench_index.py

# Inspect progress without doing any work.
python scripts/build_longbench_index.py --status

# Polite throttle on a low-rpm OpenAI tier.
python scripts/build_longbench_index.py --inter-request-pause 0.5

# Just one dataset.
python scripts/build_longbench_index.py --dataset gov_report
```

The builder checkpoints after every embedding batch, every cluster
summary, and every summary-embedding batch, persisting state to
`cache/longbench_index/<dataset>/state.json`. A SIGKILL, rate-limit,
or connection failure costs at most the in-flight unit of work; just
re-run the same command and it continues exactly where it left off.

Recovery / surgical rebuilds:

```bash
# Wipe and rebuild a dataset from scratch (e.g. after switching embed
# models or chunk geometry).
python scripts/build_longbench_index.py --dataset qmsum --force

# Drop a stage and everything downstream of it; useful for re-running
# only the cheap parts after fixing a stage-specific bug.
python scripts/build_longbench_index.py --rebuild-stage summaries
```

Valid `--rebuild-stage` values: `chunks`, `chunk_embeddings`,
`clusters`, `summaries`, `summary_embeddings`.

#### Hyper-parameter compatibility (split fingerprints)

The indexer tracks two fingerprints over the build parameters and
applies different policies depending on which one changes:

| Parameter family | Fingerprint | Behavior on change |
| --- | --- | --- |
| `--chunk-chars`, `--chunk-overlap`, `--embed-model`, record set, `--max-chunks-per-record`, `--max-total-chunks`, `--embed-batch` | **embedding** | Hard fail with a "embedding-affecting parameters changed" error. Use `--force` to rebuild from scratch. |
| `--num-clusters`, `--summary-model`, `--summary-max-tokens` | **downstream** | **Auto-reuse**: cached `chunks.jsonl` and `chunk_embeddings.npy` are kept; only clusters, summaries, and summary embeddings are recomputed. |

This means k-sweeps over `--num-clusters` are cheap: chunk embedding
cost (the dominant API cost for large LongBench builds) is paid once.

#### k-sweep workflow

```bash
# 1. First full build at k=64 -- this pays the chunk-embedding cost once.
python scripts/build_longbench_index.py --num-clusters 64

# 2. Sweep k=96 -- preflight will report:
#       embedding-compatible parameter change ... -> REUSING chunk
#       embeddings, rebuilding from clusters stage
#    No chunk-embedding API calls; only cluster / summarize / summary-embed run.
python scripts/build_longbench_index.py --num-clusters 96

# 3. Sweep k=128 -- same fast path.
python scripts/build_longbench_index.py --num-clusters 128
```

When you change something embedding-affecting (e.g. `--chunk-chars` or
`--embed-model`), the preflight surfaces:

```
preflight[gov_report]: embedding-affecting parameters changed
  (chunks/embed model/records); build will FAIL without --force
```

Pass `--force` to wipe and rebuild from scratch in that case.

### Step 2 — Run the orchestrator

## Modes

```bash
# Quick end-to-end smoke (small slices, 1-2 epochs sweep, ~1-3 minutes
# of OpenAI use depending on quota).
python scripts/run_pipelines_4x4_plus.py --smoke-test

# Reporting full run (default 40 D records per dataset, sweep
# {10,15,20,25} epochs, larger cluster k).
python scripts/run_pipelines_4x4_plus.py --full-run
```

Useful overrides:

| flag | purpose |
| --- | --- |
| `--out-dir DIR` | place artifacts elsewhere (default `reports/<ts>_plus_<mode>`) |
| `--skip-existing` (default on) | reuse existing per-dataset prediction files |
| `--no-skip-existing` | force re-run of all D pipelines |
| `--epochs N [N ...]` | override classifier epoch sweep list |
| `--d-test-per-dataset N` | override D test slice size |
| `--d-train-per-dataset N` / `--d-valid-per-dataset N` | resize D classifier splits |
| `--num-clusters K` | **(deprecated here)** configure on `scripts/build_longbench_index.py`; orchestrator no longer indexes inline |
| `--chunk-chars` / `--chunk-overlap` | chunker geometry for D baselines (must match index when relevant) |
| `--summary-max-tokens` | **(deprecated here)** configure on `scripts/build_longbench_index.py` |
| `--gen-max-tokens` | answer length cap (D pipelines) |
| `--n-ircot-steps` | retrieval rounds for D ircot baseline |
| `--embed-model` / `--gen-model` | swap OpenAI models |
| `--inter-request-pause SECS` | **(deprecated for indexing)** configure on `scripts/build_longbench_index.py` |
| `--skip-classifier-sweep --classifier-predict-file PATH` | reuse a prior 4-way predict file |
| `--skip-adaptive-rag-3way` | omit the 3-way `adaptive-rag` row (cells become NA) |
| `--adaptive-rag-3way-ckpt PATH` | explicit 3-way classifier checkpoint dir for D inference (default: auto-discover under `classifier/outputs/.../sweep`) |
| `--adaptive-rag-3way-adaptive-root PATH` | explicit 3-way adaptive root for QA cells (default: auto-discover under `predictions/classifier/.../sweep`) |

## Outputs

Under `--out-dir`:

* `pipeline_results_grouped.csv` — final 4x4 grouped table (CSV).
* `pipeline_results_grouped.md`  — same table as Markdown.
* `pipeline_results_provenance.json` — every cell mapped to its source
  artifact paths (per-dataset D breakdowns included).
* `pipeline_results_notes.md` — caveats and metric definitions.
* `coverage.json` — presence flags for every required artifact.
* `classifier_sweep_summary.json` — per-epoch validation accuracy and
  best-epoch selection for the 4-way classifier.
* `classifier_corpus_info.json` — record-level provenance of the 4-way
  training/validation/test splits (including LongBench test qids per
  dataset, used to keep splits disjoint).
* `smoke_test_report.md` — only in `--smoke-test` mode; summarizes
  completeness of every (pipeline, dataset-group) cell.

D pipeline artifacts (per dataset):

```
predictions/test/<pipeline>_gpt_<dataset>/
  predictions.jsonl    one PredictionRecord per query
  traces.jsonl         one EfficiencyTrace per query (Step/Time/tokens)
  summary.json         aggregate counts and timings
```

Adaptive (4-way) artifacts:

```
predictions/classifier_plus/<...checkpoint path...>/
  <ds>/<ds>.json          adaptive predictions per qid
  <ds>/<ds>_option.json   per-qid {prediction, option, stepNum}
```

3-way adaptive D-row artifacts (per-run, under `--out-dir`):

```
<out-dir>/adaptive_rag_3way/
  classifier_d_inference/
    d_predict.json                 D test slice repackaged for the 3-way
                                   classifier (per-record question/dataset)
    dict_id_pred_results.json      per-qid A/B/C predictions
    final_eval_results.json        accuracy field is meaningless here
                                   (3-way has no D label and we don't use it)
  adaptive_d/
    <ds>/<ds>.json                 routed predictions per qid
    <ds>/<ds>_option.json          per-qid {prediction, option, stepNum}
```

(Original 3-way `predictions/classifier/` artifacts are untouched; only
read for the QA cells of the `adaptive-rag` row.)

Global retrieval index cache (per D dataset, written by
`scripts/build_longbench_index.py`):

```
cache/longbench_index/<dataset>/
  state.json                resumable build state machine: stage,
                            counters, hyper-parameter fingerprint,
                            last error. Always-up-to-date source of
                            truth for "how far along is the build".
  chunks.jsonl              one chunk per line (atomically rewritten)
  chunk_embeddings.npy      grows in batches; rows always == state.n_chunk_embedded
  clusters.json             {labels, k}
  summaries.jsonl           append-only, one cluster summary per line
  summary_embeddings.npy    grows in batches; rows == state.n_summary_embedded
  manifest.json             present ONLY when the build is fully done
```

`index_exists(dataset)` requires `manifest.json`; partial builds are
ignored by the orchestrator, which means the only way to "see" an
index from the orchestrator is for the dedicated builder to have
finished it. This rules out silently consuming a half-built index.

## Metric definitions

* QA cells (`trivia`, `nq`, `hotpotqa`):
  * `EM` / `F1`: from existing 3-way `evaluation_metrics__*.json`.
  * `Acc`:
    * `nq` / `trivia`: substring-after-normalization (matches
      `evaluate_final_acc.evaluate_by_dicts`).
    * `hotpotqa`: official `hotpot_evaluate_v1.py` `acc` field.
  * `Step`: `0` for `nor_qa`, `1` for `oner_qa`, mean of
    `stepNum.json` for `ircot_qa`. For both adaptive rows, mean over
    routed step counts (A=0, B=1, C=ircot stepNum; D=1 for `adaptive-rag+`).
  * `Time`: per-question average; for adaptive rows, weighted from
    each routed pipeline's average per-question time.
* `longbench` (class-D):
  * `ROUGE-L`: max F1 across reference answers per query, then mean
    across queries (rouge-score with stemmer).
  * `Step` / `Time`: mean across the D test slice from the per-query
    `traces.jsonl` (or per-routed-pipeline weighted equivalent for the
    adaptive rows). For `adaptive-rag` (3-way), Step/Time are read
    per-qid from the routed A/B/C baseline trace (no D path).
  * Cell value is the macro-average of `gov_report` and `qmsum`; the
    per-dataset breakdown is preserved in
    `pipeline_results_provenance.json`.

## Class-D labelling policy

Any LongBench `gov_report` or `qmsum` query is deterministically
labelled `D` based on dataset identity (no manual rubric). The 4-way
classifier is trained on a disjoint train/valid/test split by `_id`;
the test slice is the only set ever scored.

## Resuming / partial runs

* **Indexing**: rerun `python scripts/build_longbench_index.py` — it
  resumes from the last successful checkpoint (per-batch for
  embeddings, per-cluster for summaries). Use `--status` first to
  inspect progress, `--rebuild-stage <stage>` to surgically redo one
  stage, or `--force` to wipe and rebuild a dataset. See the
  "Two-step workflow" section above for full recovery commands.
* **D pipelines**: the orchestrator is idempotent on D pipeline
  outputs when `--skip-existing` (default) is on. Delete a specific
  `predictions/test/<pipeline>_gpt_<dataset>/predictions.jsonl` to
  re-run only that cell.
* Pass `--skip-classifier-sweep --classifier-predict-file PATH` to
  reuse a previously trained 4-way classifier without retraining.
* The 3-way `adaptive-rag` row auto-discovers the latest sweep at
  `classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-small/<llm>/sweep/`
  and the matching adaptive root under
  `predictions/classifier/t5-small/<llm>/sweep/`. Override with
  `--adaptive-rag-3way-ckpt` and `--adaptive-rag-3way-adaptive-root`,
  or omit the row entirely with `--skip-adaptive-rag-3way`.

## Cost guidance (GPT-3.5 + text-embedding-3-small)

* Smoke (default `--smoke-test`):
  * ~30-50 OpenAI calls total
  * ~10-30k tokens combined
  * Pennies of OpenAI spend
* Full (`--full-run` defaults):
  * 40 D records per dataset, 4 pipelines x 2 D datasets = ~320 LLM
    completions plus embeddings
  * Roughly $0.50-$2 of OpenAI spend (varies with cluster count, IR-CoT
    steps, and answer length).

## Troubleshooting

* `Missing LongBench global index(es)`: Phase 0 of the orchestrator
  refuses to start without a complete index per D dataset. The error
  message includes the exact `python scripts/build_longbench_index.py
  ...` invocation needed.
* `RateLimitError` with `insufficient_quota`: this is a billing-side
  failure that retries cannot fix; top up OpenAI credit and rerun
  (the indexer will resume from its last checkpoint, no work lost).
* Other `RateLimitError` / connection / timeout errors: retried
  automatically with bounded exponential backoff (cap 60 s, up to 8
  attempts) inside the indexer and inference paths. If you hit
  per-minute caps, run the indexer with
  `--inter-request-pause 0.5` (or higher).
* `embedding-affecting parameters changed` / `Cached index ... was
  built with different embedding-affecting parameters`: you changed
  `--chunk-chars`, `--chunk-overlap`, `--embed-model`, or the record
  set since the last build. Either restore the original values or
  rebuild with `python scripts/build_longbench_index.py --dataset <ds>
  --force` (this is a full rebuild — chunk embeddings will be
  recomputed). Pure `--num-clusters` / summary-param changes do **not**
  raise this error; they reuse chunk embeddings automatically (see
  *k-sweep workflow* above).
* Partial-state recovery (e.g. one stage looks corrupt): use
  `--rebuild-stage chunk_embeddings|summaries|summary_embeddings`
  to drop only that stage and resume.
* Index build seems stuck: `python scripts/build_longbench_index.py
  --status` shows per-stage counters without doing any work; if
  counters are advancing across reruns, progress is real.
* Missing 3-way QA artifacts: run `scripts/run_pipelines_4x3.py
  --full-run` first.
* Stale 4-way classifier corpus: delete
  `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd_plus/` and rerun.
