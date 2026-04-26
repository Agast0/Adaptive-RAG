"""Class-D baselines: run the existing nor/oner/ircot retrieval styles
against LongBench gov_report/qmsum records.

The plan requires that *all four* pipelines (nor_qa, oner_qa, ircot_qa,
adaptive-rag+) be evaluated on LongBench D data for fair comparison. The
existing nor/oner/ircot pipelines target Wikipedia-style QA via
Elasticsearch BM25; LongBench items are self-contained long documents
shipped with each record, so for class-D we adapt the same *style* of
retrieval (no / single-step / multi-step) to the per-record ``context``:

  - ``nor_qa``  : no retrieval; LLM is asked to answer / summarize using
                  only the query (and the dataset task hint).
  - ``oner_qa`` : single-step retrieval; chunk the record's own context,
                  embed once, retrieve top-1 chunk, condition the LLM
                  on that single chunk.
  - ``ircot_qa``: multi-step retrieval; perform K=3 rounds of "retrieve
                  -> short reasoning step -> re-retrieve" against the
                  record's own context, accumulating selected chunks as
                  the working context; final answer uses all selected
                  chunks.

Each pipeline emits the same ``predictions.jsonl`` / ``traces.jsonl`` /
``summary.json`` triple as the global-D pipeline so downstream evaluation
is uniform.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import EfficiencyTrace, NormalizedRecord, PredictionRecord
from .global_index import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBED_MODEL,
    _call_with_retry,
    _chunk_text,
    _embed,
)


logger = logging.getLogger(__name__)


DEFAULT_GEN_MAX_TOKENS = 400
DEFAULT_PROMPT_BUDGET_CHARS = 9000
DEFAULT_IRCOT_STEPS = 3


# ----------------- Common helpers ----------------- #

def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return x / n


def _topk(scores: np.ndarray, k: int) -> List[int]:
    if k >= scores.shape[0]:
        return list(np.argsort(-scores))
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part])]
    return [int(i) for i in order]


def _gen_completion(prompt: str, model: str, max_tokens: int) -> Tuple[str, dict]:
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


def _trim_to_budget(prompt: str, budget: int) -> str:
    if len(prompt) <= budget:
        return prompt
    return prompt[:budget]


def _task_hint(dataset: str) -> str:
    if dataset == "gov_report":
        return "Write a one-page summary of the report."
    if dataset == "qmsum":
        return "Provide a faithful, focused answer to the meeting summarization query."
    return ""


# ----------------- nor_qa (no retrieval) on D ----------------- #

def run_nor_d(
    record: NormalizedRecord,
    *,
    gen_model: str = DEFAULT_COMPLETION_MODEL,
    gen_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
    pipeline_label: str = "nor_qa",
    routed_label: Optional[str] = "A",
) -> Tuple[PredictionRecord, EfficiencyTrace]:
    t0 = time.time()
    hint = _task_hint(record.dataset)
    prompt = (
        f"Task: {hint}\n"
        f"Query: {record.query}\n"
        "You are not given any context. Produce the best answer you can.\n"
        "Answer:"
    )
    text, usage = _gen_completion(prompt, model=gen_model, max_tokens=gen_max_tokens)
    latency = time.time() - t0
    trace = EfficiencyTrace(
        qid=record.qid, pipeline=pipeline_label, routed_label=routed_label,
        retrieval_calls=0, step_count=0.0, latency_seconds=latency,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )
    pred = PredictionRecord(
        qid=record.qid, dataset=record.dataset, pipeline=pipeline_label,
        routed_label=routed_label, prediction=text, references=list(record.references),
    )
    return pred, trace


# ----------------- oner_qa (single-step retrieval) on D ----------------- #

@dataclass
class _LocalIndex:
    chunks: List[str]
    embeddings: np.ndarray  # normalized


def _build_local_index(
    text: str,
    *,
    embed_model: str,
    chunk_chars: int,
    chunk_overlap: int,
) -> Tuple[_LocalIndex, int]:
    """Chunk + embed a single record's context. Returns (index, embed_calls)."""
    chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap=chunk_overlap)
    if not chunks:
        return _LocalIndex(chunks=[], embeddings=np.zeros((0, 1), dtype=np.float32)), 0
    embs = _embed(chunks, model=embed_model, batch=64)
    return _LocalIndex(chunks=chunks, embeddings=_normalize(embs)), 1


def run_oner_d(
    record: NormalizedRecord,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    gen_model: str = DEFAULT_COMPLETION_MODEL,
    gen_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    prompt_budget: int = DEFAULT_PROMPT_BUDGET_CHARS,
    pipeline_label: str = "oner_qa",
    routed_label: Optional[str] = "B",
) -> Tuple[PredictionRecord, EfficiencyTrace]:
    t0 = time.time()
    idx, idx_calls = _build_local_index(
        record.context, embed_model=embed_model,
        chunk_chars=chunk_chars, chunk_overlap=chunk_overlap,
    )
    qvec = _embed([record.query or _task_hint(record.dataset)], model=embed_model, batch=1)
    embed_calls = idx_calls + 1
    if idx.embeddings.shape[0] == 0:
        top_chunk = ""
    else:
        scores = _normalize(qvec)[0] @ idx.embeddings.T
        top_idx = _topk(scores, 1)
        top_chunk = idx.chunks[top_idx[0]] if top_idx else ""
    hint = _task_hint(record.dataset)
    prompt = (
        f"Task: {hint}\n"
        f"Query: {record.query}\n"
        f"Retrieved passage:\n{top_chunk}\n\n"
        "Use the retrieved passage to produce the answer.\n"
        "Answer:"
    )
    prompt = _trim_to_budget(prompt, prompt_budget)
    text, usage = _gen_completion(prompt, model=gen_model, max_tokens=gen_max_tokens)
    latency = time.time() - t0
    trace = EfficiencyTrace(
        qid=record.qid, pipeline=pipeline_label, routed_label=routed_label,
        retrieval_calls=embed_calls + 1, step_count=1.0, latency_seconds=latency,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )
    pred = PredictionRecord(
        qid=record.qid, dataset=record.dataset, pipeline=pipeline_label,
        routed_label=routed_label, prediction=text, references=list(record.references),
    )
    return pred, trace


# ----------------- ircot_qa (multi-step retrieval) on D ----------------- #

def run_ircot_d(
    record: NormalizedRecord,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    gen_model: str = DEFAULT_COMPLETION_MODEL,
    gen_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    prompt_budget: int = DEFAULT_PROMPT_BUDGET_CHARS,
    n_steps: int = DEFAULT_IRCOT_STEPS,
    pipeline_label: str = "ircot_qa",
    routed_label: Optional[str] = "C",
) -> Tuple[PredictionRecord, EfficiencyTrace]:
    t0 = time.time()
    idx, idx_calls = _build_local_index(
        record.context, embed_model=embed_model,
        chunk_chars=chunk_chars, chunk_overlap=chunk_overlap,
    )
    embed_calls = idx_calls
    selected_idx: List[int] = []
    selected_text: List[str] = []
    last_thought = record.query or _task_hint(record.dataset)
    completion_tokens_total = 0
    prompt_tokens_total = 0

    for step in range(n_steps):
        qvec = _embed([last_thought], model=embed_model, batch=1)
        embed_calls += 1
        if idx.embeddings.shape[0] == 0:
            break
        scores = _normalize(qvec)[0] @ idx.embeddings.T
        top_idx = _topk(scores, 1)
        if not top_idx or top_idx[0] in selected_idx:
            break
        selected_idx.append(top_idx[0])
        selected_text.append(idx.chunks[top_idx[0]])

        if step == n_steps - 1:
            break
        thought_prompt = (
            f"Query: {record.query}\n"
            "Selected passages so far:\n" + "\n---\n".join(selected_text) +
            "\n\nWrite a single short reasoning sentence about what additional information is still missing. "
            "Be brief.\nThought:"
        )
        thought_prompt = _trim_to_budget(thought_prompt, prompt_budget)
        thought, usage = _gen_completion(thought_prompt, model=gen_model, max_tokens=80)
        prompt_tokens_total += int(usage.get("prompt_tokens") or 0)
        completion_tokens_total += int(usage.get("completion_tokens") or 0)
        last_thought = thought.strip() or last_thought

    hint = _task_hint(record.dataset)
    final_prompt = (
        f"Task: {hint}\n"
        f"Query: {record.query}\n"
        "Retrieved passages (in order of selection):\n" + "\n---\n".join(selected_text) +
        "\n\nProduce the final answer.\nAnswer:"
    )
    final_prompt = _trim_to_budget(final_prompt, prompt_budget)
    text, usage = _gen_completion(final_prompt, model=gen_model, max_tokens=gen_max_tokens)
    prompt_tokens_total += int(usage.get("prompt_tokens") or 0)
    completion_tokens_total += int(usage.get("completion_tokens") or 0)

    latency = time.time() - t0
    trace = EfficiencyTrace(
        qid=record.qid, pipeline=pipeline_label, routed_label=routed_label,
        retrieval_calls=embed_calls + len(selected_idx),
        step_count=float(len(selected_idx)),
        latency_seconds=latency,
        prompt_tokens=prompt_tokens_total,
        completion_tokens=completion_tokens_total,
        total_tokens=prompt_tokens_total + completion_tokens_total,
    )
    pred = PredictionRecord(
        qid=record.qid, dataset=record.dataset, pipeline=pipeline_label,
        routed_label=routed_label, prediction=text, references=list(record.references),
    )
    return pred, trace


# ----------------- Bulk driver ----------------- #

def run_baseline_for_dataset(
    pipeline: str,
    records: List[NormalizedRecord],
    out_dir: Path,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    gen_model: str = DEFAULT_COMPLETION_MODEL,
    gen_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    prompt_budget: int = DEFAULT_PROMPT_BUDGET_CHARS,
    n_steps: int = DEFAULT_IRCOT_STEPS,
) -> Path:
    """Run a single baseline (nor/oner/ircot) over a list of D records."""
    runners = {
        "nor_qa": lambda r: run_nor_d(r, gen_model=gen_model, gen_max_tokens=gen_max_tokens),
        "oner_qa": lambda r: run_oner_d(
            r, embed_model=embed_model, gen_model=gen_model, gen_max_tokens=gen_max_tokens,
            chunk_chars=chunk_chars, chunk_overlap=chunk_overlap, prompt_budget=prompt_budget,
        ),
        "ircot_qa": lambda r: run_ircot_d(
            r, embed_model=embed_model, gen_model=gen_model, gen_max_tokens=gen_max_tokens,
            chunk_chars=chunk_chars, chunk_overlap=chunk_overlap, prompt_budget=prompt_budget,
            n_steps=n_steps,
        ),
    }
    if pipeline not in runners:
        raise ValueError(f"Unsupported D baseline pipeline: {pipeline}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    trace_path = out_dir / "traces.jsonl"
    summary_path = out_dir / "summary.json"

    n = 0
    total_lat = 0.0
    total_steps = 0.0
    total_calls = 0
    total_tokens = 0
    runner = runners[pipeline]
    with pred_path.open("w") as pf, trace_path.open("w") as tf:
        for rec in records:
            pred, trace = runner(rec)
            pf.write(json.dumps(pred.to_dict()) + "\n")
            tf.write(json.dumps(trace.to_dict()) + "\n")
            n += 1
            total_lat += trace.latency_seconds
            total_steps += trace.step_count
            total_calls += trace.retrieval_calls
            total_tokens += trace.total_tokens
            logger.info("D-baseline %s qid=%s lat=%.2fs steps=%.1f tok=%d",
                        pipeline, rec.qid, trace.latency_seconds, trace.step_count, trace.total_tokens)

    summary_path.write_text(json.dumps({
        "pipeline": pipeline,
        "dataset": records[0].dataset if records else None,
        "n": n,
        "avg_latency_seconds": (total_lat / n) if n else None,
        "avg_step_count": (total_steps / n) if n else None,
        "avg_retrieval_calls": (total_calls / n) if n else None,
        "avg_total_tokens": (total_tokens / n) if n else None,
    }, indent=2))
    return out_dir
