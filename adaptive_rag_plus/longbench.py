"""LongBench ingestion for AdaptiveRAG+ class-D datasets.

Scope: ONLY ``gov_report`` and ``qmsum`` (the two summarization-style tasks
from LongBench used as class-D). Loaded once via ``datasets.load_dataset``
and cached locally as JSONL so subsequent runs are fully offline-capable.

Each cached record preserves the original LongBench schema:
    {input, context, answers, length, dataset, language, all_classes, _id}

The ``label_for_dataset`` policy in :mod:`contracts` deterministically tags
every cached record as routing label ``D`` and metric type ``rouge``.

Why JSONL caching?
- LongBench `gov_report` contexts are long; we want to ingest once and reuse.
- Smoke / full splits are derived from the cache via deterministic seeded
  sampling so reports remain reproducible.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional

from .contracts import (
    LONGBENCH_D_DATASETS,
    NormalizedRecord,
    label_for_dataset,
    metric_type_for_dataset,
)


logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "THUDM/LongBench"
DEFAULT_CACHE_DIR = Path("raw_data") / "longbench"
DEFAULT_PROCESSED_DIR = Path("processed_data") / "longbench"


# LongBench `gov_report` ships with an empty `input` field; the task
# prompt is implied by dataset identity. We seed the query slot with the
# canonical LongBench instruction so the global retriever has something
# meaningful to embed.
DEFAULT_QUERY_BY_DATASET: dict = {
    "gov_report": "Write a one-page summary of the report.",
    "qmsum": "",
}


def _ensure_dataset_name(dataset: str) -> None:
    if dataset not in LONGBENCH_D_DATASETS:
        raise ValueError(
            f"Unsupported LongBench dataset for AdaptiveRAG+: {dataset!r}. "
            f"Allowed: {LONGBENCH_D_DATASETS}"
        )


def cache_path_for(dataset: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    _ensure_dataset_name(dataset)
    return cache_dir / f"{dataset}.jsonl"


LONGBENCH_DATA_ZIP_URL = "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"


def download_longbench_jsonl(
    dataset: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> Path:
    """Ensure ``<cache_dir>/<dataset>.jsonl`` exists.

    The official LongBench HF repo ships a single ``data.zip`` archive (the
    legacy script loader is no longer supported by ``datasets>=2.21``). We
    fetch the archive once, extract only the two D-class subtasks, and cache
    them as plain JSONL so subsequent runs are offline.
    """
    _ensure_dataset_name(dataset)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_path_for(dataset, cache_dir)
    if out_path.exists() and not force:
        logger.info("LongBench %s already cached at %s", dataset, out_path)
        return out_path

    import io
    import urllib.request
    import zipfile

    logger.info("Downloading LongBench archive from %s ...", LONGBENCH_DATA_ZIP_URL)
    with urllib.request.urlopen(LONGBENCH_DATA_ZIP_URL) as resp:
        archive_bytes = resp.read()
    logger.info("Fetched %.1f MiB; extracting %s ...", len(archive_bytes) / (1024 * 1024), dataset)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        member = f"data/{dataset}.jsonl"
        if member not in zf.namelist():
            raise FileNotFoundError(f"{member} not present in LongBench archive")
        with zf.open(member) as src, out_path.open("wb") as dst:
            dst.write(src.read())
    logger.info("Cached LongBench %s -> %s", dataset, out_path)
    return out_path


def iter_cached_records(path: Path) -> Iterable[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_normalized(
    dataset: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_samples: Optional[int] = None,
    seed: int = 42,
    download_if_missing: bool = True,
) -> List[NormalizedRecord]:
    """Return NormalizedRecord list for a LongBench D dataset.

    Sampling is deterministic for given (dataset, seed, max_samples).
    """
    _ensure_dataset_name(dataset)
    path = cache_path_for(dataset, cache_dir)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Missing local LongBench cache: {path}")
        download_longbench_jsonl(dataset, cache_dir=cache_dir)

    rows = list(iter_cached_records(path))
    if max_samples is not None and 0 < max_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)
        rows.sort(key=lambda r: r.get("_id", ""))

    records: List[NormalizedRecord] = []
    for row in rows:
        ds_name = row.get("dataset") or dataset
        query = (row.get("input") or "").strip()
        if not query:
            query = DEFAULT_QUERY_BY_DATASET.get(ds_name, "")
        records.append(NormalizedRecord(
            qid=str(row["_id"]),
            query=query,
            context=row.get("context", ""),
            references=list(row.get("answers", []) or []),
            dataset=ds_name,
            routing_label=label_for_dataset(ds_name) or "D",
            target_metric_type=metric_type_for_dataset(ds_name),
            length=row.get("length"),
            language=row.get("language"),
            all_classes=row.get("all_classes"),
            raw=row,
        ))
    return records


def write_processed(
    records: List[NormalizedRecord],
    out_path: Path,
) -> Path:
    """Write normalized records as JSONL (1 per line) for downstream consumers."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    return out_path


def materialize_processed_split(
    dataset: str,
    split_name: str,
    max_samples: Optional[int],
    seed: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Convenience: download->normalize->write a named split (e.g. smoke/full)."""
    records = load_normalized(
        dataset, cache_dir=cache_dir, max_samples=max_samples, seed=seed
    )
    out = Path(processed_dir) / dataset / f"{split_name}.jsonl"
    write_processed(records, out)
    logger.info("Materialized %d records to %s", len(records), out)
    return out
