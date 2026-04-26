#!/usr/bin/env python
"""Run global_qa-only LongBench sweeps for multiple k values.

This script is intentionally lightweight: it only builds/loads the
class-D global index and runs the global retrieval pipeline over the D
datasets (gov_report, qmsum). It does not run the full 4x4/5x4
orchestrator, classifier sweep, or other baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=_REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass

from adaptive_rag_plus import contracts, evaluation, global_index, global_inference, longbench  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _compute_metrics(
    *,
    k: int,
    records_per_dataset: Optional[int],
    seed: int,
    out_root: Path,
    force: bool,
    rebuild_stage: Optional[str],
    chunk_chars: int,
    chunk_overlap: int,
    embed_model: str,
    summary_model: str,
    summary_max_tokens: int,
    embed_batch: int,
    gen_model: str,
    gen_max_tokens: int,
    inter_request_pause: float,
    index_root: Path,
) -> Dict[str, object]:
    per_dataset: Dict[str, Dict[str, Optional[float]]] = {}
    run_root = out_root / f"k_{k}"
    run_root.mkdir(parents=True, exist_ok=True)

    for ds in contracts.LONGBENCH_D_DATASETS:
        longbench.download_longbench_jsonl(ds)
        records = longbench.load_normalized(
            ds,
            max_samples=records_per_dataset,
            seed=seed,
        )
        logging.info("k=%d dataset=%s loaded records=%d", k, ds, len(records))

        idx = global_index.build_global_index(
            records,
            ds,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            num_clusters=k,
            embed_model=embed_model,
            summary_model=summary_model,
            summary_max_tokens=summary_max_tokens,
            embed_batch=embed_batch,
            inter_request_pause=inter_request_pause,
            root=index_root,
            force=force,
            rebuild_stage=rebuild_stage,
        )

        ds_out = run_root / ds / "global_qa"
        global_inference.run_global_pipeline_for_dataset(
            records,
            idx,
            ds_out,
            embed_model=embed_model,
            gen_model=gen_model,
            gen_max_tokens=gen_max_tokens,
            pipeline_label="global_qa",
            routed_label="D",
        )

        metrics = evaluation.evaluate_d_predictions_jsonl(ds_out / "predictions.jsonl")
        per_dataset[ds] = metrics
        logging.info(
            "k=%d dataset=%s ROUGE-L=%.4f Step=%.3f Time=%.3f",
            k,
            ds,
            float(metrics.get("ROUGE-L") or 0.0),
            float(metrics.get("Step") or 0.0),
            float(metrics.get("Time") or 0.0),
        )

    longbench_agg = evaluation.aggregate_longbench(per_dataset)
    payload: Dict[str, object] = {
        "k": k,
        "per_dataset": per_dataset,
        "longbench": longbench_agg,
    }
    (run_root / "global_longbench_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run global_qa-only LongBench sweeps across k values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        required=True,
        help="cluster counts to evaluate (e.g. --k-values 64 96 128 256).",
    )
    parser.add_argument(
        "--records-per-dataset",
        type=int,
        default=40,
        help="cap records per D dataset; pass 200 for full LongBench D split.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(_REPO_ROOT / "reports" / "global_k_sweep_only"),
    )
    parser.add_argument(
        "--index-root",
        type=str,
        default=str(_REPO_ROOT / "cache" / "longbench_index"),
    )

    parser.add_argument("--chunk-chars", type=int, default=global_index.DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=global_index.DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--summary-max-tokens", type=int, default=global_index.DEFAULT_SUMMARY_MAX_TOKENS)
    parser.add_argument("--embed-batch", type=int, default=global_index.DEFAULT_EMBED_BATCH)
    parser.add_argument("--embed-model", type=str, default=global_index.DEFAULT_EMBED_MODEL)
    parser.add_argument("--summary-model", type=str, default=global_index.DEFAULT_SUMMARY_MODEL)
    parser.add_argument("--gen-model", type=str, default=global_index.DEFAULT_COMPLETION_MODEL)
    parser.add_argument("--gen-max-tokens", type=int, default=global_inference.DEFAULT_GEN_MAX_TOKENS)
    parser.add_argument("--inter-request-pause", type=float, default=global_index.DEFAULT_INTER_REQUEST_PAUSE)

    rebuild = parser.add_mutually_exclusive_group()
    rebuild.add_argument("--force", action="store_true", help="force full index rebuild for each dataset.")
    rebuild.add_argument(
        "--rebuild-stage",
        choices=list(global_index.REBUILDABLE_STAGES),
        default=None,
        help="drop this stage (and downstream) before each k run.",
    )

    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    index_root = Path(args.index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    if not args.k_values:
        raise SystemExit("--k-values must not be empty")

    summary: Dict[str, object] = {
        "k_values": args.k_values,
        "records_per_dataset": args.records_per_dataset,
        "seed": args.seed,
        "results": {},
    }

    for k in args.k_values:
        payload = _compute_metrics(
            k=k,
            records_per_dataset=args.records_per_dataset,
            seed=args.seed,
            out_root=out_root,
            force=args.force,
            rebuild_stage=args.rebuild_stage,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
            embed_model=args.embed_model,
            summary_model=args.summary_model,
            summary_max_tokens=args.summary_max_tokens,
            embed_batch=args.embed_batch,
            gen_model=args.gen_model,
            gen_max_tokens=args.gen_max_tokens,
            inter_request_pause=args.inter_request_pause,
            index_root=index_root,
        )
        summary["results"][str(k)] = payload

    summary_path = out_root / "k_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logging.info("wrote summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
