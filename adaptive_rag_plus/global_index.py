"""Class-D offline global index: chunk -> embed -> cluster -> summarize -> store.

The implementation follows the AdaptiveRAG+ specification:

  1. Process the knowledge base by dividing documents into smaller chunks.
  2. Generate vector embeddings for each chunk.
  3. Group embeddings semantically using agglomerative clustering until k
     clusters remain.
  4. Use an LLM to summarize each cluster (the "high-level themes" view).
  5. Embed those summaries and store them as the searchable index.

All artifacts are persisted to disk so the index is built once per dataset
and reused across pipelines (D-only baseline, AdaptiveRAG+ system, etc.).

Index layout (per dataset under ``cache/longbench_index/<dataset>/``):
    state.json                -- resumable state machine: stage, counters,
                                 hyper-parameter fingerprint, last error.
    chunks.jsonl              -- one chunk per line: {id, text, source_qid}.
                                 Written atomically after the chunks stage.
    chunk_embeddings.npy      -- float32 [n_chunk_embedded, dim]. Grows in
                                 batches; ``state.n_chunk_embedded`` is the
                                 source of truth for how many rows are valid.
    clusters.json             -- {labels: [...], k: int}. Written atomically.
    summaries.jsonl           -- one cluster summary per line; appended after
                                 every successful cluster summary.
    summary_embeddings.npy    -- float32 [n_summary_embedded, dim]. Grows
                                 incrementally like chunk_embeddings.
    manifest.json             -- written ONLY when the build is fully done;
                                 :func:`index_exists` requires it.

Resume semantics:
  - Every successful unit of work (one embedding batch, one cluster
    summary, one summary-embedding batch) is followed by an atomic
    persistence + ``state.json`` update before the next call. A crash,
    kill, or rate-limit abort therefore costs at most the in-flight unit.
  - The hyper-parameter fingerprint protects against silent reuse when
    chunking geometry, models, or input records change. A mismatch on
    resume aborts unless the caller passes ``force=True`` (full rebuild).

Backends:
  - Embeddings: OpenAI ``text-embedding-3-small`` via the existing
    ``openai`` (==0.28) SDK already wired into the rest of the repo.
  - LLM summaries: ``gpt-3.5-turbo-instruct`` (matches base pipelines).
  - Clustering: ``sklearn.cluster.AgglomerativeClustering`` with cosine
    affinity + average linkage. Falls back to a single cluster if
    ``n_chunks <= 1`` or ``k`` exceeds available chunk count.

Reliability:
  - All OpenAI calls go through :func:`_call_with_retry`, which honors
    ``Retry-After`` when present and otherwise applies bounded exponential
    backoff with jitter on 429 / connection / timeout / transient 5xx.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import NormalizedRecord


logger = logging.getLogger(__name__)


# ----------------- Configuration ----------------- #

DEFAULT_INDEX_ROOT = Path("cache") / "longbench_index"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
# Model used only for offline cluster summarization during index build.
DEFAULT_SUMMARY_MODEL = "gpt-4o-mini"
# Model used for completion-style generation in inference pipelines.
DEFAULT_COMPLETION_MODEL = "gpt-3.5-turbo-instruct"
DEFAULT_CHUNK_CHARS = 2000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_NUM_CLUSTERS = 8
DEFAULT_SUMMARY_MAX_TOKENS = 220
DEFAULT_EMBED_BATCH = 64

# Retry policy for OpenAI calls. Tuned to be polite without hanging the
# orchestrator forever; the orchestrator surfaces final failures rather than
# silently swallowing them.
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_INITIAL_BACKOFF = 2.0   # seconds
DEFAULT_MAX_BACKOFF = 60.0      # seconds
DEFAULT_INTER_REQUEST_PAUSE = 0.0  # seconds, optional throttle between calls


# ----------------- Stage enum ----------------- #

STAGE_INIT = "init"
STAGE_CHUNKS_DONE = "chunks_done"
STAGE_CHUNK_EMBED = "chunk_embed"
STAGE_CLUSTERS_DONE = "clusters_done"
STAGE_SUMMARIES = "summaries"
STAGE_SUMMARIES_DONE = "summaries_done"
STAGE_SUMMARY_EMBED = "summary_embed"
STAGE_COMPLETE = "complete"

ALL_STAGES = (
    STAGE_INIT,
    STAGE_CHUNKS_DONE,
    STAGE_CHUNK_EMBED,
    STAGE_CLUSTERS_DONE,
    STAGE_SUMMARIES,
    STAGE_SUMMARIES_DONE,
    STAGE_SUMMARY_EMBED,
    STAGE_COMPLETE,
)

REBUILDABLE_STAGES = (
    "chunks",
    "chunk_embeddings",
    "clusters",
    "summaries",
    "summary_embeddings",
)


# ----------------- Reliability helpers ----------------- #

def _classify_openai_error(exc: BaseException) -> "tuple[bool, Optional[float]]":
    """Return (is_retryable, suggested_wait_seconds)."""
    try:
        import openai
    except Exception:  # pragma: no cover
        return False, None

    # Authentication / invalid request -> not retryable.
    if isinstance(exc, getattr(openai.error, "AuthenticationError", ())):
        return False, None
    if isinstance(exc, getattr(openai.error, "InvalidRequestError", ())):
        return False, None
    if isinstance(exc, getattr(openai.error, "PermissionError", ())):
        return False, None

    # Quota exhausted is technically a 429 but with a billing message; the
    # SDK surfaces it as RateLimitError but with code=insufficient_quota.
    if isinstance(exc, getattr(openai.error, "RateLimitError", ())):
        code = getattr(exc, "code", None) or ""
        msg = str(exc).lower()
        if code == "insufficient_quota" or "insufficient_quota" in msg or "exceeded your current quota" in msg:
            return False, None
        wait = _retry_after_from_exc(exc)
        return True, wait

    if isinstance(exc, getattr(openai.error, "Timeout", ())):
        return True, None
    if isinstance(exc, getattr(openai.error, "APIConnectionError", ())):
        return True, None
    if isinstance(exc, getattr(openai.error, "ServiceUnavailableError", ())):
        return True, None
    if isinstance(exc, getattr(openai.error, "APIError", ())):
        status = getattr(exc, "http_status", None)
        if status and 500 <= int(status) < 600:
            return True, None
        return False, None
    return False, None


def _retry_after_from_exc(exc: BaseException) -> Optional[float]:
    headers = getattr(exc, "headers", None) or {}
    try:
        ra = headers.get("Retry-After") if hasattr(headers, "get") else None
    except Exception:
        ra = None
    if ra is None:
        return None
    try:
        return max(0.0, float(ra))
    except (TypeError, ValueError):
        return None


def _call_with_retry(
    fn: Callable,
    *args,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    op_label: str = "openai_call",
    **kwargs,
):
    """Call ``fn(*args, **kwargs)`` with bounded exponential backoff.

    Honors ``Retry-After`` headers when the SDK exposes them; otherwise uses
    ``initial_backoff * 2**(attempt-1)`` capped at ``max_backoff`` with a
    small jitter. Re-raises non-retryable errors immediately.
    """
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt < max_attempts:
        attempt += 1
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            retryable, hinted_wait = _classify_openai_error(exc)
            last_exc = exc
            if not retryable or attempt >= max_attempts:
                raise
            if hinted_wait is not None:
                wait = hinted_wait
            else:
                wait = min(max_backoff, initial_backoff * (2 ** (attempt - 1)))
                wait += random.uniform(0, min(1.0, wait * 0.25))
            logger.warning(
                "%s failed (attempt %d/%d): %s -> sleeping %.2fs",
                op_label, attempt, max_attempts, exc, wait,
            )
            time.sleep(wait)
    if last_exc is not None:  # pragma: no cover (logically unreachable)
        raise last_exc
    raise RuntimeError(f"{op_label}: exhausted retries with no exception")


# ----------------- Embedding / summarization primitives ----------------- #

def _embed_openai_batch(texts: Sequence[str], model: str) -> np.ndarray:
    """Single embedding call. The caller is responsible for batching."""
    import openai

    resp = _call_with_retry(
        openai.Embedding.create,
        model=model,
        input=list(texts),
        op_label=f"embeddings.create[{model}]",
    )
    vecs = [np.asarray(d["embedding"], dtype=np.float32) for d in resp["data"]]
    return np.vstack(vecs)


def _embed(
    texts: Sequence[str],
    *,
    model: str,
    batch: int = DEFAULT_EMBED_BATCH,
    inter_request_pause: float = DEFAULT_INTER_REQUEST_PAUSE,
) -> np.ndarray:
    """Embed a list of texts in batches. Used by inference paths that
    don't need the resumable on-disk checkpointing of the index builder.
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    out: List[np.ndarray] = []
    n_batches = (len(texts) + batch - 1) // batch
    for bi, i in enumerate(range(0, len(texts), batch), start=1):
        chunk = list(texts[i:i + batch])
        logger.debug("embedding batch %d/%d (size=%d)", bi, n_batches, len(chunk))
        out.append(_embed_openai_batch(chunk, model=model))
        if inter_request_pause > 0 and bi < n_batches:
            time.sleep(inter_request_pause)
    return np.vstack(out)


def _summarize_cluster(
    member_chunks: Sequence[str],
    *,
    model: str,
    max_tokens: int,
) -> str:
    """LLM summary of cluster contents.

    Same OpenAI completion API the base pipelines use so credentials and
    quota are shared.
    """
    import openai

    if not member_chunks:
        return ""
    joined = "\n\n---\n\n".join(member_chunks)
    if len(joined) > 12000:
        joined = joined[:12000]
    prompt = (
        "You are summarizing a cluster of related text passages drawn from a "
        "larger knowledge base. Produce a concise, faithful summary that "
        "captures the key themes, entities, and findings. Avoid speculation "
        "and do not introduce information beyond the passages.\n\n"
        f"Passages:\n{joined}\n\nSummary:"
    )
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
    return (resp["choices"][0]["text"] or "").strip()


# ----------------- Chunking ----------------- #

def _chunk_text(text: str, *, chunk_chars: int, overlap: int) -> List[str]:
    """Char-based sliding window chunker.

    Char windows keep the implementation tokenizer-free and predictable. The
    LongBench contexts are plain English, so a 2k-char window approximates
    400-500 tokens which is comfortable for the embedding model.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks: List[str] = []
    step = max(1, chunk_chars - overlap)
    for start in range(0, len(text), step):
        end = start + chunk_chars
        chunks.append(text[start:end])
        if end >= len(text):
            break
    return chunks


def _build_chunks(
    records: List[NormalizedRecord],
    *,
    chunk_chars: int,
    chunk_overlap: int,
    max_chunks_per_record: Optional[int],
    max_total_chunks: Optional[int],
) -> Tuple[List[dict], bool]:
    chunks: List[dict] = []
    sampled = False
    for rec in records:
        rec_chunks = _chunk_text(rec.context, chunk_chars=chunk_chars, overlap=chunk_overlap)
        if max_chunks_per_record is not None:
            rec_chunks = rec_chunks[:max_chunks_per_record]
        for i, ch in enumerate(rec_chunks):
            chunks.append({
                "id": f"{rec.qid}_{i}",
                "text": ch,
                "source_qid": rec.qid,
                "source_dataset": rec.dataset,
            })
    if max_total_chunks is not None and len(chunks) > max_total_chunks:
        chunks = chunks[:max_total_chunks]
        sampled = True
    return chunks, sampled


# ----------------- Clustering ----------------- #

def _cluster(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return cluster labels. Falls back to single-cluster when k>=n."""
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n == 1 or k <= 1:
        return np.zeros((n,), dtype=np.int32)
    k = min(k, n)
    from sklearn.cluster import AgglomerativeClustering

    try:
        clusterer = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=k,
            affinity="cosine",
            linkage="average",
        )
    labels = clusterer.fit_predict(embeddings)
    return labels.astype(np.int32)


# ----------------- Atomic IO helpers ----------------- #

def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # ``np.save`` would append ``.npy`` to a string/Path that doesn't end in
    # ``.npy``; pass an open file handle so the on-disk name is exact.
    with tmp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


# ----------------- Resumable state machine ----------------- #

@dataclass
class IndexState:
    """Persistent build state for one dataset's index.

    The state file is the single source of truth for "how far along is the
    build". All on-disk artifacts are sized/truncated according to the
    counters here on resume.
    """

    dataset: str
    stage: str = STAGE_INIT
    params_fingerprint: str = ""
    params: dict = field(default_factory=dict)
    n_records: int = 0
    n_chunks: int = 0
    n_chunk_embedded: int = 0
    embedding_dim: Optional[int] = None
    n_clusters_target: int = 0
    n_clusters_actual: Optional[int] = None
    n_summaries_done: int = 0
    n_summary_embedded: int = 0
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    last_error: Optional[str] = None
    completed: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_path(cls, path: Path) -> Optional["IndexState"]:
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(**data)

    def save(self, path: Path) -> None:
        self.updated_at = time.time()
        _atomic_write_text(path, self.to_json())


def _params_fingerprint(
    *,
    records: List[NormalizedRecord],
    embed_model: str,
    summary_model: str,
    chunk_chars: int,
    chunk_overlap: int,
    num_clusters: int,
    summary_max_tokens: int,
    embed_batch: int,
    max_chunks_per_record: Optional[int],
    max_total_chunks: Optional[int],
) -> Tuple[str, dict]:
    params = {
        "embed_model": embed_model,
        "summary_model": summary_model,
        "chunk_chars": int(chunk_chars),
        "chunk_overlap": int(chunk_overlap),
        "num_clusters": int(num_clusters),
        "summary_max_tokens": int(summary_max_tokens),
        "embed_batch": int(embed_batch),
        "max_chunks_per_record": max_chunks_per_record,
        "max_total_chunks": max_total_chunks,
        "n_records": len(records),
        "records_signature": [
            {"qid": r.qid, "context_len": len(r.context or "")} for r in records
        ],
    }
    h = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
    return h, params


def _index_paths(out_dir: Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        "state": out_dir / "state.json",
        "chunks": out_dir / "chunks.jsonl",
        "chunk_embeddings": out_dir / "chunk_embeddings.npy",
        "clusters": out_dir / "clusters.json",
        "summaries": out_dir / "summaries.jsonl",
        "summary_embeddings": out_dir / "summary_embeddings.npy",
        "manifest": out_dir / "manifest.json",
    }


# ----------------- Data classes ----------------- #

@dataclass
class IndexManifest:
    dataset: str
    embed_model: str
    summary_model: str
    chunk_chars: int
    chunk_overlap: int
    num_clusters: int
    n_records: int
    n_chunks: int
    n_clusters: int
    embedding_dim: int
    build_seconds: float
    paths: dict = field(default_factory=dict)
    sampled_chunks: bool = False


@dataclass
class GlobalIndex:
    """In-memory handle around a built index. Use :meth:`load` to rehydrate."""
    dataset: str
    chunks: List[dict]
    chunk_embeddings: np.ndarray
    cluster_labels: np.ndarray
    summaries: List[dict]
    summary_embeddings: np.ndarray
    manifest: IndexManifest

    def index_dir(self, root: Path = DEFAULT_INDEX_ROOT) -> Path:
        return Path(root) / self.dataset

    @classmethod
    def load(cls, dataset: str, root: Path = DEFAULT_INDEX_ROOT) -> "GlobalIndex":
        d = Path(root) / dataset
        manifest = IndexManifest(**json.loads((d / "manifest.json").read_text()))
        chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines() if l]
        summaries = [json.loads(l) for l in (d / "summaries.jsonl").read_text().splitlines() if l]
        chunk_emb = np.load(d / "chunk_embeddings.npy")
        summary_emb = np.load(d / "summary_embeddings.npy")
        labels = np.asarray(json.loads((d / "clusters.json").read_text())["labels"], dtype=np.int32)
        return cls(
            dataset=dataset,
            chunks=chunks,
            chunk_embeddings=chunk_emb,
            cluster_labels=labels,
            summaries=summaries,
            summary_embeddings=summary_emb,
            manifest=manifest,
        )


# ----------------- Build entrypoint ----------------- #

def index_exists(dataset: str, root: Path = DEFAULT_INDEX_ROOT) -> bool:
    """A *complete* index is one whose ``manifest.json`` exists.

    Partial / mid-build artifacts are intentionally not enough; the
    resumable builder writes ``manifest.json`` only as the final step.
    """
    d = Path(root) / dataset
    needed = ["chunks.jsonl", "chunk_embeddings.npy", "clusters.json",
              "summaries.jsonl", "summary_embeddings.npy", "manifest.json"]
    return all((d / x).exists() for x in needed)


def index_build_status(dataset: str, root: Path = DEFAULT_INDEX_ROOT) -> dict:
    """Inspect on-disk progress without doing any work.

    Returns a dict with state, completion booleans, and remaining counts.
    Useful for ``--status`` flags and orchestrator preflight.
    """
    out_dir = Path(root) / dataset
    paths = _index_paths(out_dir)
    state = IndexState.from_path(paths["state"])
    info: dict = {
        "dataset": dataset,
        "out_dir": str(out_dir),
        "state_exists": state is not None,
        "manifest_exists": paths["manifest"].exists(),
    }
    if state is not None:
        info.update({
            "stage": state.stage,
            "params_fingerprint": state.params_fingerprint,
            "n_records": state.n_records,
            "n_chunks": state.n_chunks,
            "n_chunk_embedded": state.n_chunk_embedded,
            "n_clusters_target": state.n_clusters_target,
            "n_clusters_actual": state.n_clusters_actual,
            "n_summaries_done": state.n_summaries_done,
            "n_summary_embedded": state.n_summary_embedded,
            "completed": state.completed,
            "last_error": state.last_error,
            "remaining_chunk_embeddings": max(0, state.n_chunks - state.n_chunk_embedded),
            "remaining_summaries": (
                max(0, (state.n_clusters_actual or 0) - state.n_summaries_done)
                if state.n_clusters_actual is not None
                else None
            ),
            "remaining_summary_embeddings": (
                max(0, (state.n_clusters_actual or 0) - state.n_summary_embedded)
                if state.n_clusters_actual is not None
                else None
            ),
        })
    info["complete"] = info["manifest_exists"] and (
        state.completed if state is not None else False
    )
    return info


def _load_existing_chunks(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_existing_summaries(path: Path) -> List[dict]:
    return _load_existing_chunks(path)  # same JSONL parser


def _truncate_npy(path: Path, n_rows: int) -> None:
    """Truncate an existing .npy to its first ``n_rows`` rows in place.

    Used to reconcile an array against the canonical row count recorded in
    state.json (e.g. after a crash mid-batch when state was updated before
    the partial array write completed -- defensively safer to trim).
    """
    if not path.exists() or n_rows <= 0:
        return
    arr = np.load(path)
    if arr.shape[0] == n_rows:
        return
    if arr.shape[0] < n_rows:
        return
    _atomic_save_npy(path, arr[:n_rows])


def _truncate_jsonl(path: Path, n_lines: int) -> None:
    """Truncate a JSONL file to its first ``n_lines`` lines."""
    if not path.exists() or n_lines <= 0:
        return
    with path.open() as f:
        kept = [next(f) for _ in range(n_lines)]
    _atomic_write_text(path, "".join(kept))


def _reconcile_artifacts_with_state(state: IndexState, paths: Dict[str, Path]) -> None:
    """Make on-disk artifacts match the counters in ``state``.

    Conservative principle: state.json is the source of truth (it is the
    last thing written after each successful unit, so any artifact rows
    beyond its counters were written by a partial run that did not finish
    its checkpoint cycle and must be discarded).
    """
    if state.n_chunk_embedded > 0:
        _truncate_npy(paths["chunk_embeddings"], state.n_chunk_embedded)
    elif paths["chunk_embeddings"].exists():
        paths["chunk_embeddings"].unlink()

    if state.n_summaries_done > 0:
        _truncate_jsonl(paths["summaries"], state.n_summaries_done)
    elif paths["summaries"].exists():
        paths["summaries"].unlink()

    if state.n_summary_embedded > 0:
        _truncate_npy(paths["summary_embeddings"], state.n_summary_embedded)
    elif paths["summary_embeddings"].exists():
        paths["summary_embeddings"].unlink()


def _wipe_stage(state: IndexState, paths: Dict[str, Path], stage: str) -> None:
    """Drop on-disk artifacts and reset counters for one stage and beyond."""
    if stage == "chunks":
        for k in ("chunks", "chunk_embeddings", "clusters", "summaries", "summary_embeddings"):
            if paths[k].exists():
                paths[k].unlink()
        state.stage = STAGE_INIT
        state.n_chunks = 0
        state.n_chunk_embedded = 0
        state.embedding_dim = None
        state.n_clusters_actual = None
        state.n_summaries_done = 0
        state.n_summary_embedded = 0
    elif stage == "chunk_embeddings":
        for k in ("chunk_embeddings", "clusters", "summaries", "summary_embeddings"):
            if paths[k].exists():
                paths[k].unlink()
        state.stage = STAGE_CHUNKS_DONE
        state.n_chunk_embedded = 0
        state.embedding_dim = None
        state.n_clusters_actual = None
        state.n_summaries_done = 0
        state.n_summary_embedded = 0
    elif stage == "clusters":
        for k in ("clusters", "summaries", "summary_embeddings"):
            if paths[k].exists():
                paths[k].unlink()
        state.stage = STAGE_CHUNK_EMBED  # will resume->complete chunk_embed then go on
        if state.n_chunk_embedded == state.n_chunks and state.n_chunks > 0:
            state.stage = STAGE_CHUNKS_DONE  # ready for re-cluster
            # actually we want to allow re-cluster: bump to a state that triggers cluster
        state.n_clusters_actual = None
        state.n_summaries_done = 0
        state.n_summary_embedded = 0
    elif stage == "summaries":
        for k in ("summaries", "summary_embeddings"):
            if paths[k].exists():
                paths[k].unlink()
        state.stage = STAGE_CLUSTERS_DONE
        state.n_summaries_done = 0
        state.n_summary_embedded = 0
    elif stage == "summary_embeddings":
        if paths["summary_embeddings"].exists():
            paths["summary_embeddings"].unlink()
        state.stage = STAGE_SUMMARIES_DONE
        state.n_summary_embedded = 0
    else:
        raise ValueError(f"Unknown rebuild stage: {stage!r} (must be one of {REBUILDABLE_STAGES})")
    state.completed = False


def build_global_index(
    records: List[NormalizedRecord],
    dataset: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    num_clusters: int = DEFAULT_NUM_CLUSTERS,
    embed_model: str = DEFAULT_EMBED_MODEL,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
    embed_batch: int = DEFAULT_EMBED_BATCH,
    inter_request_pause: float = DEFAULT_INTER_REQUEST_PAUSE,
    max_chunks_per_record: Optional[int] = None,
    max_total_chunks: Optional[int] = None,
    root: Path = DEFAULT_INDEX_ROOT,
    force: bool = False,
    rebuild_stage: Optional[str] = None,
) -> GlobalIndex:
    """Build (or resume / load if cached) a class-D global index.

    Caching:
      - If a complete index exists (``manifest.json`` present) and
        ``force=False`` and ``rebuild_stage is None``, it is loaded and
        returned immediately.
      - Otherwise the resumable state machine in
        ``cache/longbench_index/<dataset>/state.json`` is consulted and
        the build continues from the last successful checkpoint.

    Behavior:
      - ``force=True``  : wipe the directory and rebuild from scratch.
      - ``rebuild_stage`` in :data:`REBUILDABLE_STAGES` : drop that stage
        (and all later stages) and resume from there. Validates the
        existing fingerprint matches the requested params.
      - Hyper-parameter changes (chunk geometry, models, record set) cause
        a fingerprint mismatch and the builder will refuse to resume
        unless ``force=True``.

    Cost / rate-limit control:
      - ``max_chunks_per_record`` caps per-record chunking depth.
      - ``max_total_chunks`` caps overall chunk count (truncates with
        ``sampled_chunks=True`` recorded in the manifest).
      - ``inter_request_pause`` adds a polite delay between API batches
        when an account has restrictive RPM quotas.
    """
    if rebuild_stage is not None and rebuild_stage not in REBUILDABLE_STAGES:
        raise ValueError(
            f"rebuild_stage must be one of {REBUILDABLE_STAGES}, got {rebuild_stage!r}"
        )

    out_dir = Path(root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _index_paths(out_dir)

    fingerprint, params = _params_fingerprint(
        records=records,
        embed_model=embed_model,
        summary_model=summary_model,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        num_clusters=num_clusters,
        summary_max_tokens=summary_max_tokens,
        embed_batch=embed_batch,
        max_chunks_per_record=max_chunks_per_record,
        max_total_chunks=max_total_chunks,
    )

    existing_state = IndexState.from_path(paths["state"])

    # Quick-load path: complete index, no rebuild requested, fingerprint matches.
    if not force and rebuild_stage is None and index_exists(dataset, root):
        if existing_state is None or existing_state.params_fingerprint == fingerprint:
            logger.info("Reusing cached global index at %s", out_dir)
            return GlobalIndex.load(dataset, root=root)
        raise RuntimeError(
            f"Cached index for {dataset!r} was built with different parameters "
            f"(fingerprint mismatch). Pass force=True to rebuild or use matching "
            f"parameters. Existing params: {existing_state.params}; requested: {params}"
        )

    state = existing_state
    if force:
        logger.info("force=True -> wiping %s", out_dir)
        for k in ("chunks", "chunk_embeddings", "clusters", "summaries",
                  "summary_embeddings", "manifest", "state"):
            if paths[k].exists():
                paths[k].unlink()
        state = None

    if state is not None and state.params_fingerprint and state.params_fingerprint != fingerprint:
        raise RuntimeError(
            f"Cached index for {dataset!r} was built with different parameters "
            f"(fingerprint mismatch). Pass force=True to rebuild or use "
            f"matching parameters. Existing params: {state.params}; "
            f"requested: {params}"
        )

    if state is None:
        state = IndexState(
            dataset=dataset,
            stage=STAGE_INIT,
            params_fingerprint=fingerprint,
            params=params,
            n_records=len(records),
            n_clusters_target=int(num_clusters),
            started_at=time.time(),
        )
        state.save(paths["state"])

    if rebuild_stage is not None:
        logger.info("rebuild_stage=%s -> wiping and resuming from there", rebuild_stage)
        _wipe_stage(state, paths, rebuild_stage)
        if paths["manifest"].exists():
            paths["manifest"].unlink()
        state.save(paths["state"])

    _reconcile_artifacts_with_state(state, paths)

    t0 = time.time()

    # ---------- Stage: chunks ----------
    if state.stage == STAGE_INIT:
        logger.info("[chunks] building from %d records ...", len(records))
        chunks, sampled = _build_chunks(
            records,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            max_chunks_per_record=max_chunks_per_record,
            max_total_chunks=max_total_chunks,
        )
        if not chunks:
            raise ValueError(f"No chunks produced for dataset {dataset!r}; check input records")
        _atomic_write_text(
            paths["chunks"],
            "".join(json.dumps(c) + "\n" for c in chunks),
        )
        state.n_chunks = len(chunks)
        state.params["sampled_chunks"] = sampled
        state.stage = STAGE_CHUNKS_DONE
        state.save(paths["state"])
        logger.info("[chunks] done: %d chunks (sampled=%s)", len(chunks), sampled)
    else:
        chunks = _load_existing_chunks(paths["chunks"])
        if len(chunks) != state.n_chunks:
            raise RuntimeError(
                f"chunks.jsonl row count ({len(chunks)}) disagrees with state "
                f"({state.n_chunks}); rerun with force=True or rebuild_stage='chunks'."
            )

    # ---------- Stage: chunk embeddings ----------
    if state.stage in (STAGE_CHUNKS_DONE, STAGE_CHUNK_EMBED):
        if state.stage == STAGE_CHUNKS_DONE:
            state.stage = STAGE_CHUNK_EMBED
            state.save(paths["state"])
        n_done = state.n_chunk_embedded
        if n_done > 0 and paths["chunk_embeddings"].exists():
            existing = np.load(paths["chunk_embeddings"])
            if existing.shape[0] != n_done:
                _truncate_npy(paths["chunk_embeddings"], n_done)
                existing = np.load(paths["chunk_embeddings"])
        else:
            existing = None
        total = state.n_chunks
        logger.info("[chunk_embed] %d/%d already done; embedding the rest in batches of %d",
                    n_done, total, embed_batch)
        i = n_done
        while i < total:
            batch_texts = [c["text"] for c in chunks[i:i + embed_batch]]
            new_emb = _embed_openai_batch(batch_texts, model=embed_model)
            if state.embedding_dim is None:
                state.embedding_dim = int(new_emb.shape[1])
            if existing is None:
                merged = new_emb
            else:
                merged = np.vstack([existing, new_emb])
            _atomic_save_npy(paths["chunk_embeddings"], merged)
            existing = merged
            i += new_emb.shape[0]
            state.n_chunk_embedded = i
            state.save(paths["state"])
            logger.info("[chunk_embed] progress: %d/%d", i, total)
            if inter_request_pause > 0 and i < total:
                time.sleep(inter_request_pause)
        state.stage = STAGE_CLUSTERS_DONE  # we'll do clustering below; reuses term loosely
        state.save(paths["state"])

    # Reload chunk embeddings for the next stages.
    chunk_embeddings = np.load(paths["chunk_embeddings"])
    if chunk_embeddings.shape[0] != state.n_chunks:
        raise RuntimeError(
            f"chunk_embeddings shape[0]={chunk_embeddings.shape[0]} disagrees "
            f"with state.n_chunks={state.n_chunks}"
        )

    # ---------- Stage: clusters ----------
    # If clusters.json missing or n_clusters_actual unknown, recompute clusters.
    if not paths["clusters"].exists() or state.n_clusters_actual is None:
        k = min(num_clusters, state.n_chunks)
        logger.info("[cluster] k=%d (n_chunks=%d) ...", k, state.n_chunks)
        labels = _cluster(chunk_embeddings, k)
        _atomic_write_text(
            paths["clusters"],
            json.dumps({"labels": [int(x) for x in labels.tolist()], "k": int(k)}),
        )
        state.n_clusters_actual = int(k)
        state.save(paths["state"])
        logger.info("[cluster] done: k=%d", k)
    else:
        labels = np.asarray(json.loads(paths["clusters"].read_text())["labels"], dtype=np.int32)

    cluster_to_idx: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        cluster_to_idx.setdefault(int(lab), []).append(idx)
    sorted_cluster_ids = sorted(cluster_to_idx)

    # ---------- Stage: summaries ----------
    if state.stage in (STAGE_CLUSTERS_DONE, STAGE_SUMMARIES):
        if state.stage == STAGE_CLUSTERS_DONE:
            state.stage = STAGE_SUMMARIES
            state.save(paths["state"])
        n_done = state.n_summaries_done
        logger.info("[summarize] %d/%d clusters already done; resuming",
                    n_done, len(sorted_cluster_ids))
        for i in range(n_done, len(sorted_cluster_ids)):
            lab = sorted_cluster_ids[i]
            member_idx = cluster_to_idx[lab]
            member_texts = [chunks[j]["text"] for j in member_idx]
            member_qids = sorted({chunks[j]["source_qid"] for j in member_idx})
            text = _summarize_cluster(
                member_texts, model=summary_model, max_tokens=summary_max_tokens
            )
            _append_jsonl(paths["summaries"], {
                "cluster_id": int(lab),
                "text": text,
                "member_chunk_ids": [chunks[j]["id"] for j in member_idx],
                "member_qids": member_qids,
                "size": len(member_idx),
            })
            state.n_summaries_done = i + 1
            state.save(paths["state"])
            logger.info("[summarize] progress: %d/%d", i + 1, len(sorted_cluster_ids))
            if inter_request_pause > 0 and (i + 1) < len(sorted_cluster_ids):
                time.sleep(inter_request_pause)
        state.stage = STAGE_SUMMARIES_DONE
        state.save(paths["state"])

    summaries = _load_existing_summaries(paths["summaries"])
    if len(summaries) != state.n_clusters_actual:
        raise RuntimeError(
            f"summaries.jsonl row count ({len(summaries)}) disagrees with "
            f"n_clusters_actual ({state.n_clusters_actual})"
        )

    # ---------- Stage: summary embeddings ----------
    if state.stage in (STAGE_SUMMARIES_DONE, STAGE_SUMMARY_EMBED):
        if state.stage == STAGE_SUMMARIES_DONE:
            state.stage = STAGE_SUMMARY_EMBED
            state.save(paths["state"])
        n_done = state.n_summary_embedded
        if n_done > 0 and paths["summary_embeddings"].exists():
            existing_se = np.load(paths["summary_embeddings"])
            if existing_se.shape[0] != n_done:
                _truncate_npy(paths["summary_embeddings"], n_done)
                existing_se = np.load(paths["summary_embeddings"])
        else:
            existing_se = None
        total = len(summaries)
        logger.info("[summary_embed] %d/%d already done; embedding the rest", n_done, total)
        i = n_done
        while i < total:
            batch_texts = [s["text"] for s in summaries[i:i + embed_batch]]
            new_emb = _embed_openai_batch(batch_texts, model=embed_model)
            if state.embedding_dim is None:
                state.embedding_dim = int(new_emb.shape[1])
            if existing_se is None:
                merged = new_emb
            else:
                merged = np.vstack([existing_se, new_emb])
            _atomic_save_npy(paths["summary_embeddings"], merged)
            existing_se = merged
            i += new_emb.shape[0]
            state.n_summary_embedded = i
            state.save(paths["state"])
            logger.info("[summary_embed] progress: %d/%d", i, total)
            if inter_request_pause > 0 and i < total:
                time.sleep(inter_request_pause)
        state.stage = STAGE_COMPLETE
        state.completed = True
        state.save(paths["state"])

    summary_embeddings = np.load(paths["summary_embeddings"])

    # ---------- Finalize manifest ----------
    sampled = bool(state.params.get("sampled_chunks"))
    manifest = IndexManifest(
        dataset=dataset,
        embed_model=embed_model,
        summary_model=summary_model,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        num_clusters=int(state.n_clusters_actual or 0),
        n_records=state.n_records,
        n_chunks=state.n_chunks,
        n_clusters=int(state.n_clusters_actual or 0),
        embedding_dim=int(state.embedding_dim or chunk_embeddings.shape[1]),
        build_seconds=time.time() - t0,
        paths={k: str(v) for k, v in paths.items() if k != "state"},
        sampled_chunks=sampled,
    )
    _atomic_write_text(paths["manifest"], json.dumps(manifest.__dict__, indent=2))
    logger.info(
        "Built global index for %s: chunks=%d clusters=%d in %.1fs",
        dataset, state.n_chunks, len(summaries), manifest.build_seconds,
    )

    return GlobalIndex(
        dataset=dataset,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        cluster_labels=labels,
        summaries=summaries,
        summary_embeddings=summary_embeddings,
        manifest=manifest,
    )
