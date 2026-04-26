"""Class-D online retrieval + generation.

Given a built :class:`~.global_index.GlobalIndex` and an input query, this
module:

  1. Embeds the query with the same embedding model used for the index.
  2. Retrieves the top-N most relevant cluster summaries (cosine similarity
     on the L2-normalised summary vectors).
  3. Optionally pulls top-M member chunks from those clusters for finer
     grounding (off by default to keep this faithful to the spec, which
     says the summaries themselves are the global context).
  4. Builds a generation prompt and calls ``gpt-3.5-turbo-instruct``.
  5. Emits an :class:`~.contracts.EfficiencyTrace` (latency, retrieval
     calls, token usage) and a :class:`~.contracts.PredictionRecord`.

All OpenAI calls go through the shared retry helper in :mod:`global_index`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .contracts import EfficiencyTrace, NormalizedRecord, PredictionRecord
from .global_index import (
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBED_MODEL,
    GlobalIndex,
    _call_with_retry,
    _embed,
)


logger = logging.getLogger(__name__)


DEFAULT_TOP_N_SUMMARIES = 3
DEFAULT_TOP_M_CHUNKS = 0  # 0 disables chunk grounding; spec says summaries-only is fine
DEFAULT_GEN_MAX_TOKENS = 400
DEFAULT_GEN_MODEL = DEFAULT_COMPLETION_MODEL
DEFAULT_PROMPT_CONTEXT_BUDGET_CHARS = 9000


# ----------------- Retrieval ----------------- #

def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return x / n


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


@dataclass
class RetrievalResult:
    """Top retrieval payload returned by :func:`retrieve_global`."""
    summary_idx: List[int]
    summary_scores: List[float]
    chunk_idx: List[int]
    chunk_scores: List[float]
    summary_calls: int
    chunk_calls: int


def retrieve_global(
    index: GlobalIndex,
    query_vec: np.ndarray,
    *,
    top_n_summaries: int = DEFAULT_TOP_N_SUMMARIES,
    top_m_chunks: int = DEFAULT_TOP_M_CHUNKS,
) -> RetrievalResult:
    """Retrieve top summaries (and optional member chunks) for a query."""
    qv = _normalize(query_vec.reshape(1, -1))[0]

    sum_norm = _normalize(index.summary_embeddings)
    sum_scores = sum_norm @ qv
    n_sum = min(top_n_summaries, sum_norm.shape[0])
    sum_top = _topk_indices(sum_scores, n_sum) if n_sum > 0 else np.zeros((0,), dtype=np.int64)
    summary_calls = 1

    chunk_top: List[int] = []
    chunk_scores: List[float] = []
    chunk_calls = 0
    if top_m_chunks > 0:
        member_chunk_global_idx: List[int] = []
        for s_idx in sum_top.tolist():
            for cid in index.summaries[s_idx]["member_chunk_ids"]:
                gi_idx = next((i for i, c in enumerate(index.chunks) if c["id"] == cid), None)
                if gi_idx is not None:
                    member_chunk_global_idx.append(gi_idx)
        if member_chunk_global_idx:
            sub = index.chunk_embeddings[member_chunk_global_idx]
            sub_norm = _normalize(sub)
            sub_scores = sub_norm @ qv
            m = min(top_m_chunks, sub_norm.shape[0])
            top = _topk_indices(sub_scores, m)
            chunk_top = [int(member_chunk_global_idx[i]) for i in top.tolist()]
            chunk_scores = [float(sub_scores[i]) for i in top.tolist()]
            chunk_calls = 1

    return RetrievalResult(
        summary_idx=[int(i) for i in sum_top.tolist()],
        summary_scores=[float(sum_scores[i]) for i in sum_top.tolist()],
        chunk_idx=chunk_top,
        chunk_scores=chunk_scores,
        summary_calls=summary_calls,
        chunk_calls=chunk_calls,
    )


# ----------------- Generation ----------------- #

def _build_prompt(query: str, summaries: List[str], chunks: List[str], budget: int) -> str:
    parts: List[str] = []
    parts.append(
        "You are an expert assistant. Use the global summaries (and any "
        "supporting passages) below to answer the user query. The summaries "
        "describe themes from a knowledge base; treat them as authoritative "
        "context. If the answer requires a multi-paragraph summary, produce "
        "one. Be faithful to the provided context.\n"
    )
    if summaries:
        parts.append("Global summaries:")
        for i, s in enumerate(summaries):
            parts.append(f"[Summary {i + 1}] {s}")
    if chunks:
        parts.append("\nSupporting passages:")
        for i, c in enumerate(chunks):
            parts.append(f"[Passage {i + 1}] {c}")
    parts.append("\nUser query:")
    parts.append(query.strip() or "(no explicit query; produce the standard task output for this dataset)")
    parts.append("\nAnswer:")
    prompt = "\n".join(parts)
    if len(prompt) > budget:
        # Trim from the middle of supporting context (keep query + leading instruction).
        head = parts[0] + "\n"
        tail_lines = parts[-3:]  # blank, query block, "Answer:"
        body = "\n".join(parts[1:-3])
        keep = max(0, budget - len(head) - sum(len(x) + 1 for x in tail_lines))
        body = body[:keep]
        prompt = head + body + "\n" + "\n".join(tail_lines)
    return prompt


def _generate(prompt: str, model: str, max_tokens: int) -> Tuple[str, dict]:
    import openai

    resp = _call_with_retry(
        openai.Completion.create,
        model=model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        n=1,
        op_label=f"completions.create[{model}]",
    )
    text = (resp["choices"][0]["text"] or "").strip()
    usage = dict(resp.get("usage") or {})
    return text, usage


# ----------------- Per-record D inference ----------------- #

def answer_with_global_index(
    record: NormalizedRecord,
    index: GlobalIndex,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    gen_model: str = DEFAULT_GEN_MODEL,
    top_n_summaries: int = DEFAULT_TOP_N_SUMMARIES,
    top_m_chunks: int = DEFAULT_TOP_M_CHUNKS,
    gen_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
    prompt_budget_chars: int = DEFAULT_PROMPT_CONTEXT_BUDGET_CHARS,
    pipeline_label: str = "global_qa",
    routed_label: Optional[str] = "D",
) -> Tuple[PredictionRecord, EfficiencyTrace]:
    """Run a single class-D query end-to-end and return (prediction, trace)."""
    t_start = time.time()

    query_vec = _embed([record.query], model=embed_model, batch=1)
    embed_calls = 1

    retrieval = retrieve_global(
        index, query_vec[0],
        top_n_summaries=top_n_summaries,
        top_m_chunks=top_m_chunks,
    )

    summaries = [index.summaries[i]["text"] for i in retrieval.summary_idx]
    chunks = [index.chunks[i]["text"] for i in retrieval.chunk_idx]

    prompt = _build_prompt(record.query, summaries, chunks, prompt_budget_chars)
    text, usage = _generate(prompt, model=gen_model, max_tokens=gen_max_tokens)

    latency = time.time() - t_start

    retrieval_calls = embed_calls + retrieval.summary_calls + retrieval.chunk_calls

    trace = EfficiencyTrace(
        qid=record.qid,
        pipeline=pipeline_label,
        routed_label=routed_label,
        retrieval_calls=retrieval_calls,
        step_count=1.0,
        latency_seconds=latency,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )

    pred = PredictionRecord(
        qid=record.qid,
        dataset=record.dataset,
        pipeline=pipeline_label,
        routed_label=routed_label,
        prediction=text,
        references=list(record.references),
    )
    return pred, trace


# ----------------- Dataset-level driver ----------------- #

def run_global_pipeline_for_dataset(
    records: List[NormalizedRecord],
    index: GlobalIndex,
    out_dir: Path,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    gen_model: str = DEFAULT_GEN_MODEL,
    top_n_summaries: int = DEFAULT_TOP_N_SUMMARIES,
    top_m_chunks: int = DEFAULT_TOP_M_CHUNKS,
    gen_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
    prompt_budget_chars: int = DEFAULT_PROMPT_CONTEXT_BUDGET_CHARS,
    pipeline_label: str = "global_qa",
    routed_label: Optional[str] = "D",
) -> Path:
    """Run the D pipeline over a record list and persist predictions+traces.

    Output layout under ``out_dir``:
        predictions.jsonl  -- one PredictionRecord per line
        traces.jsonl       -- one EfficiencyTrace per line
        summary.json       -- aggregate counts and timing
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.jsonl"
    trace_path = out_dir / "traces.jsonl"
    summary_path = out_dir / "summary.json"

    n = 0
    total_latency = 0.0
    total_retrieval_calls = 0
    total_tokens = 0
    with pred_path.open("w") as pf, trace_path.open("w") as tf:
        for rec in records:
            pred, trace = answer_with_global_index(
                rec, index,
                embed_model=embed_model, gen_model=gen_model,
                top_n_summaries=top_n_summaries, top_m_chunks=top_m_chunks,
                gen_max_tokens=gen_max_tokens, prompt_budget_chars=prompt_budget_chars,
                pipeline_label=pipeline_label, routed_label=routed_label,
            )
            pf.write(json.dumps(pred.to_dict()) + "\n")
            tf.write(json.dumps(trace.to_dict()) + "\n")
            n += 1
            total_latency += trace.latency_seconds
            total_retrieval_calls += trace.retrieval_calls
            total_tokens += trace.total_tokens
            logger.info("D-infer %s qid=%s lat=%.2fs tok=%d",
                        rec.dataset, rec.qid, trace.latency_seconds, trace.total_tokens)

    summary = {
        "dataset": records[0].dataset if records else None,
        "pipeline": pipeline_label,
        "n": n,
        "avg_latency_seconds": (total_latency / n) if n else None,
        "avg_retrieval_calls": (total_retrieval_calls / n) if n else None,
        "avg_total_tokens": (total_tokens / n) if n else None,
        "predictions_path": str(pred_path),
        "traces_path": str(trace_path),
        "index_dataset": index.dataset,
        "index_manifest": index.manifest.__dict__,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return out_dir
