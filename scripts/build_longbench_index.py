#!/usr/bin/env python
"""Build (or resume) the LongBench class-D global retrieval indexes.

This is the dedicated indexing entrypoint for AdaptiveRAG+. It is decoupled
from the main orchestrator (``scripts/run_pipelines_4x4_plus.py``) so that
indexing can survive rate limits and other transient OpenAI failures and
continue exactly where it left off across process restarts.

Defaults
========
- Indexes BOTH ``gov_report`` and ``qmsum``.
- Uses ALL records in each LongBench dataset (200 per dataset).
- Resumes from the last successful checkpoint if a partial build exists.
- Caches everything under ``cache/longbench_index/<dataset>/``.

Resumability
============
Each unit of work (one embedding batch, one cluster summary, one
summary-embedding batch) writes its result and updates ``state.json``
atomically before the next OpenAI call. A crash, kill, or rate-limit
abort therefore costs at most the in-flight unit. Re-running this script
is always safe: it picks up exactly where it left off.

Examples
========
::

    # Default: index both gov_report and qmsum (all 200 records each), resume on rerun.
    python scripts/build_longbench_index.py

    # Just one dataset.
    python scripts/build_longbench_index.py --dataset gov_report

    # Inspect what's done without doing any work.
    python scripts/build_longbench_index.py --status

    # Polite throttle on a low-rpm tier.
    python scripts/build_longbench_index.py --inter-request-pause 0.5

    # Force complete rebuild (e.g. after switching embed models).
    python scripts/build_longbench_index.py --force

    # Rebuild only the summaries (and onwards) without re-embedding chunks.
    python scripts/build_longbench_index.py --rebuild-stage summaries

    # Cap records per dataset (smoke-style cheap run).
    python scripts/build_longbench_index.py --records-per-dataset 5 --num-clusters 3

The script exits 0 only when every requested dataset is fully indexed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=_REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass

from adaptive_rag_plus import contracts, global_index, longbench  # noqa: E402


D_DATASETS = list(contracts.LONGBENCH_D_DATASETS)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def section(msg: str) -> None:
    bar = "=" * (len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)


def _load_records(
    dataset: str,
    *,
    records_per_dataset: Optional[int],
    seed: int,
) -> list:
    longbench.download_longbench_jsonl(dataset)
    recs = longbench.load_normalized(
        dataset,
        max_samples=records_per_dataset,
        seed=seed,
    )
    return recs


def _emit_status(dataset: str, root: Path) -> dict:
    status = global_index.index_build_status(dataset, root=root)
    log(
        f"  status[{dataset}]: stage={status.get('stage', '<none>')} "
        f"complete={status.get('complete')} "
        f"chunks={status.get('n_chunk_embedded')}/{status.get('n_chunks')} "
        f"summaries={status.get('n_summaries_done')}/{status.get('n_clusters_actual')} "
        f"summary_embeds={status.get('n_summary_embedded')}/{status.get('n_clusters_actual')}"
    )
    return status


def _build_one(
    dataset: str,
    *,
    args: argparse.Namespace,
    root: Path,
) -> dict:
    section(f"Indexing {dataset!r}")
    if args.status:
        return _emit_status(dataset, root)

    recs = _load_records(
        dataset,
        records_per_dataset=args.records_per_dataset,
        seed=args.seed,
    )
    log(f"  loaded {len(recs)} records for {dataset}")

    _emit_status(dataset, root)

    global_index.build_global_index(
        recs, dataset,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        num_clusters=args.num_clusters,
        embed_model=args.embed_model,
        summary_model=args.summary_model,
        summary_max_tokens=args.summary_max_tokens,
        embed_batch=args.embed_batch,
        inter_request_pause=args.inter_request_pause,
        max_chunks_per_record=args.max_chunks_per_record,
        max_total_chunks=args.max_total_chunks,
        root=root,
        force=args.force,
        rebuild_stage=args.rebuild_stage,
    )
    return _emit_status(dataset, root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or resume the LongBench class-D global retrieval indexes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", action="append", choices=D_DATASETS, default=None,
        help="restrict to a single LongBench dataset (repeatable). "
             "If omitted, both gov_report and qmsum are indexed.",
    )
    parser.add_argument(
        "--records-per-dataset", type=int, default=None,
        help="cap records per dataset (default: all 200).",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for record sub-sampling when capped.")

    parser.add_argument("--chunk-chars", type=int, default=global_index.DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=global_index.DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--num-clusters", type=int, default=global_index.DEFAULT_NUM_CLUSTERS)
    parser.add_argument("--summary-max-tokens", type=int, default=global_index.DEFAULT_SUMMARY_MAX_TOKENS)
    parser.add_argument("--embed-batch", type=int, default=global_index.DEFAULT_EMBED_BATCH)
    parser.add_argument("--max-chunks-per-record", type=int, default=None)
    parser.add_argument("--max-total-chunks", type=int, default=None)

    parser.add_argument("--embed-model", type=str, default=global_index.DEFAULT_EMBED_MODEL)
    parser.add_argument("--summary-model", type=str, default=global_index.DEFAULT_SUMMARY_MODEL)
    parser.add_argument("--inter-request-pause", type=float,
                        default=global_index.DEFAULT_INTER_REQUEST_PAUSE,
                        help="seconds to pause between OpenAI requests (rate-limit polite mode).")

    parser.add_argument("--root", type=str, default=str(_REPO_ROOT / "cache" / "longbench_index"),
                        help="index cache root.")

    rebuild = parser.add_mutually_exclusive_group()
    rebuild.add_argument("--force", action="store_true",
                         help="wipe the dataset index and rebuild from scratch.")
    rebuild.add_argument("--rebuild-stage", choices=list(global_index.REBUILDABLE_STAGES),
                         default=None,
                         help="drop this stage (and all later stages) and resume from there.")

    parser.add_argument("--status", action="store_true",
                        help="print the current build state for each dataset and exit.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="enable DEBUG logging.")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if not os.environ.get("OPENAI_API_KEY") and not args.status:
        log("WARNING: OPENAI_API_KEY is not set; OpenAI calls will fail. "
            "Run --status to inspect existing artifacts without calling the API.")

    targets: List[str] = list(args.dataset) if args.dataset else list(D_DATASETS)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    log(f"target datasets: {targets}")
    log(f"index root:      {root}")
    if args.status:
        log("mode: status (no work performed)")
    elif args.force:
        log("mode: FORCE (full rebuild)")
    elif args.rebuild_stage:
        log(f"mode: rebuild stage='{args.rebuild_stage}' (and onwards)")
    else:
        log("mode: resume (default)")

    final_statuses: dict = {}
    failures: List[str] = []
    for ds in targets:
        try:
            final_statuses[ds] = _build_one(ds, args=args, root=root)
        except Exception as exc:  # noqa: BLE001
            log(f"  ERROR indexing {ds}: {exc}")
            failures.append(ds)
            final_statuses[ds] = global_index.index_build_status(ds, root=root)

    section("Summary")
    for ds in targets:
        s = final_statuses.get(ds) or {}
        if s.get("complete"):
            log(f"  {ds}: COMPLETE "
                f"(chunks={s.get('n_chunks')} clusters={s.get('n_clusters_actual')})")
        else:
            log(f"  {ds}: INCOMPLETE stage={s.get('stage')} "
                f"chunks_embedded={s.get('n_chunk_embedded')}/{s.get('n_chunks')} "
                f"summaries={s.get('n_summaries_done')}/{s.get('n_clusters_actual')} "
                f"summary_embeds={s.get('n_summary_embedded')}/{s.get('n_clusters_actual')}")

    print(json.dumps({
        "targets": targets,
        "failures": failures,
        "statuses": final_statuses,
    }, indent=2, default=str))

    if args.status:
        return 0
    if failures:
        log(f"non-fatal: {len(failures)} dataset(s) did not finish; rerun this command to resume.")
        return 1
    if all(final_statuses.get(ds, {}).get("complete") for ds in targets):
        log("all requested datasets are fully indexed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
