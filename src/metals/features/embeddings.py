"""Embedding generation with a chunked-Parquet on-disk cache.

Phase 3 step 3.6.

Design choices
==============

**Default model.** ``all-MiniLM-L6-v2`` (384-dim). At 48M+ headlines the
storage and throughput win over mpnet is substantial; cluster quality on
news-style text is essentially unchanged for our scenario-discovery goal.

**Storage.** One Parquet file per shard, where the shard is determined by the
first three hex characters of ``sha256(text)`` - 4,096 shards in total. At a
typical Phase 3 corpus size that's ~10K embeddings per shard. Total disk
usage scales linearly: ~37 GB for 48.5M MiniLM embeddings in fp16.

**Cache location.** Defaults to ``%LOCALAPPDATA%\\metals\\embeddings`` on
Windows, ``~/.cache/metals/embeddings`` elsewhere - explicitly OUTSIDE the
project folder so OneDrive (or any other sync engine) doesn't try to upload
tens of GB of cache files. ``METALS_EMBEDDING_CACHE_DIR`` env var lets you
point somewhere else. The resolver warns if the resolved path contains
sync-folder tokens (OneDrive, Dropbox, GoogleDrive, iCloud, Box).

Public API
==========

Backwards-compatible with the previous npy-per-text implementation:

- ``embed_texts(texts, ...) -> np.ndarray``
- ``embed_dataframe(df, text_column, ...) -> np.ndarray``
- ``EmbedConfig`` dataclass (model name + normalize + dtype)
- ``DEFAULT_MODEL``, ``CACHE_ROOT`` (module-level constants)

Returned arrays are always fp32 even when the cache stores fp16, because most
downstream consumers (UMAP, scikit-learn, BERTopic) prefer fp32 input.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DTYPE = "fp16"
SHARD_PREFIX_LEN = 3
SHARD_LRU_SIZE = 32
DANGEROUS_PATH_TOKENS = frozenset(
    {
        "onedrive",
        "dropbox",
        "googledrive",
        "google drive",
        "icloud",
        "box sync",
    }
)

_model_cache: dict[str, object] = {}


def _warn_if_synced(path: Path) -> Path:
    parts_lower = {p.lower() for p in path.parts}
    overlap = parts_lower & DANGEROUS_PATH_TOKENS
    if overlap:
        warnings.warn(
            f"Embedding cache at {path} appears to be inside a sync-managed "
            f"directory ({sorted(overlap)}). This will trigger massive sync "
            f"uploads of tens of GB. Set METALS_EMBEDDING_CACHE_DIR to a path "
            f"outside any sync folder.",
            RuntimeWarning,
            stacklevel=3,
        )
    return path


def resolve_cache_dir(env: dict | None = None) -> Path:
    """Resolve the embeddings cache directory.

    Priority:
      1. METALS_EMBEDDING_CACHE_DIR env var (warned if sync-folder).
      2. Platform default outside any sync folder:
         Windows: %LOCALAPPDATA%\\metals\\embeddings
         Other:    ~/.cache/metals/embeddings
    """
    env = env if env is not None else os.environ
    override = env.get("METALS_EMBEDDING_CACHE_DIR")
    if override:
        return _warn_if_synced(Path(override).expanduser())
    if sys.platform == "win32":
        base_str = env.get("LOCALAPPDATA")
        base = Path(base_str) if base_str else (Path.home() / "AppData" / "Local")
    else:
        base = Path.home() / ".cache"
    return base / "metals" / "embeddings"


CACHE_ROOT = resolve_cache_dir()


def _hash_bytes(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _hash_hex(text: str) -> str:
    return _hash_bytes(text).hex()


def _shard_prefix(hex_hash: str) -> str:
    return hex_hash[:SHARD_PREFIX_LEN]


@dataclass(frozen=True)
class EmbedConfig:
    model_name: str = DEFAULT_MODEL
    normalize: bool = True
    dtype: str = DEFAULT_DTYPE

    def fingerprint(self) -> str:
        blob = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _arrow_dtype(dtype: str):
    if dtype == "fp16":
        return pa.float16()
    if dtype == "fp32":
        return pa.float32()
    raise ValueError(f"Unsupported dtype {dtype!r}; use 'fp16' or 'fp32'.")


def _shard_schema(dtype: str) -> pa.Schema:
    return pa.schema(
        [
            ("text_hash", pa.binary(32)),
            ("embedding", pa.list_(_arrow_dtype(dtype))),
        ]
    )


class ParquetEmbeddingCache:
    """Sharded Parquet cache. One file per 12-bit prefix of the text hash."""

    def __init__(self, cache_root: Path, config: EmbedConfig):
        self.root = Path(cache_root) / config.fingerprint()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._lru: OrderedDict[str, dict[bytes, np.ndarray]] = OrderedDict()

    def shard_path(self, prefix: str) -> Path:
        return self.root / f"shard_{prefix}.parquet"

    def _load_shard(self, prefix: str) -> dict[bytes, np.ndarray]:
        if prefix in self._lru:
            self._lru.move_to_end(prefix)
            return self._lru[prefix]
        path = self.shard_path(prefix)
        if path.exists():
            tbl = pq.read_table(path, columns=["text_hash", "embedding"])
            hashes = tbl.column("text_hash").to_pylist()
            embeddings = tbl.column("embedding").to_pylist()
            shard = {
                h: np.asarray(e, dtype=np.float32) for h, e in zip(hashes, embeddings, strict=False)
            }
        else:
            shard = {}
        self._lru[prefix] = shard
        while len(self._lru) > SHARD_LRU_SIZE:
            self._lru.popitem(last=False)
        return shard

    def read_many(self, hex_hashes: Sequence[str]) -> dict[str, np.ndarray]:
        if not hex_hashes:
            return {}
        by_shard: dict[str, list[str]] = {}
        for h in hex_hashes:
            by_shard.setdefault(_shard_prefix(h), []).append(h)
        out: dict[str, np.ndarray] = {}
        for prefix, group in by_shard.items():
            shard = self._load_shard(prefix)
            for hex_h in group:
                raw = bytes.fromhex(hex_h)
                if raw in shard:
                    out[hex_h] = shard[raw]
        return out

    def write_many(self, items: dict[str, np.ndarray]) -> int:
        if not items:
            return 0
        by_shard: dict[str, dict[bytes, np.ndarray]] = {}
        for hex_h, vec in items.items():
            by_shard.setdefault(_shard_prefix(hex_h), {})[bytes.fromhex(hex_h)] = vec
        n_written = 0
        for prefix, new_items in by_shard.items():
            existing = self._load_shard(prefix)
            merged = {**existing, **new_items}
            self._write_shard(prefix, merged)
            self._lru[prefix] = merged
            self._lru.move_to_end(prefix)
            n_written += len(new_items)
        while len(self._lru) > SHARD_LRU_SIZE:
            self._lru.popitem(last=False)
        return n_written

    def _write_shard(self, prefix: str, items: dict[bytes, np.ndarray]) -> None:
        if not items:
            return
        hashes = list(items.keys())
        embeddings = [np.asarray(v, dtype=np.float32).tolist() for v in items.values()]
        tbl = pa.Table.from_pydict(
            {"text_hash": hashes, "embedding": embeddings},
            schema=_shard_schema(self.config.dtype),
        )
        path = self.shard_path(prefix)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(tbl, tmp, compression="zstd")
        os.replace(tmp, path)


def _get_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


def _resolve_model_name(model_name: str | None) -> str:
    if model_name:
        return model_name
    return os.getenv("METALS_EMBEDDING_MODEL", DEFAULT_MODEL)


def embed_texts(
    texts: Sequence[str],
    model_name: str | None = None,
    *,
    batch_size: int = 64,
    normalize: bool = True,
    use_cache: bool = True,
    dtype: str | None = None,
) -> np.ndarray:
    """Embed ``texts`` and return an ``(n, d)`` float32 array."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    config = EmbedConfig(
        model_name=_resolve_model_name(model_name),
        normalize=normalize,
        dtype=dtype or DEFAULT_DTYPE,
    )
    hexes = [_hash_hex(t) if t else "" for t in texts]
    cache = ParquetEmbeddingCache(resolve_cache_dir(), config) if use_cache else None
    cached = cache.read_many([h for h in hexes if h]) if cache is not None else {}
    missing_idx: list[int] = []
    for i, (t, h) in enumerate(zip(texts, hexes, strict=False)):
        if not t:
            continue
        if h not in cached:
            missing_idx.append(i)
    if missing_idx:
        model = _get_model(config.model_name)
        new_vecs = model.encode(
            [texts[i] for i in missing_idx],
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(missing_idx) > 1_000,
            convert_to_numpy=True,
        )
        new_vecs = np.asarray(new_vecs, dtype=np.float32)
        new_items = {hexes[i]: new_vecs[j] for j, i in enumerate(missing_idx)}
        if cache is not None:
            cache.write_many(new_items)
        cached.update(new_items)
    dim = next((v.shape[0] for v in cached.values()), 0)
    ordered: list[np.ndarray] = []
    for t, h in zip(texts, hexes, strict=False):
        if not t:
            ordered.append(np.zeros(dim, dtype=np.float32))
        else:
            ordered.append(cached[h])
    return np.vstack(ordered).astype(np.float32, copy=False)


def cache_embeddings(
    texts: Sequence[str],
    model_name: str | None = None,
    *,
    batch_size: int = 256,
    normalize: bool = True,
    dtype: str | None = None,
    sub_chunk: int = 50_000,
) -> int:
    """Encode ``texts`` and persist to the on-disk cache *without* building a
    full in-RAM result array.

    ``embed_texts`` returns one fp32 vector per input row and ends in a single
    ``np.vstack`` — fine for bounded inputs, but ~97 GB (a guaranteed OOM) for
    the full 63 M-row corpus. ``cache_embeddings`` streams in ``sub_chunk``-sized
    blocks: each block reads the cache, encodes only the misses (de-duplicated
    within the block), writes them, and is then discarded. Peak memory is one
    block of vectors, not the whole corpus. Returns the count of newly-encoded
    (cache-miss) vectors.
    """
    texts = list(texts)
    if not texts:
        return 0
    config = EmbedConfig(
        model_name=_resolve_model_name(model_name),
        normalize=normalize,
        dtype=dtype or DEFAULT_DTYPE,
    )
    cache = ParquetEmbeddingCache(resolve_cache_dir(), config)
    model = None
    n_new = 0
    for start in range(0, len(texts), sub_chunk):
        block = texts[start : start + sub_chunk]
        hexes = [_hash_hex(t) if t else "" for t in block]
        present = cache.read_many([h for h in hexes if h])
        # De-duplicate misses within the block so identical texts encode once;
        # cross-block duplicates are caught by the cache read above.
        to_encode: dict[str, str] = {}
        for h, t in zip(hexes, block, strict=True):
            if h and h not in present:
                to_encode.setdefault(h, t)
        if not to_encode:
            continue
        if model is None:
            model = _get_model(config.model_name)
        keys = list(to_encode.keys())
        vecs = model.encode(
            [to_encode[h] for h in keys],
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        cache.write_many({h: vecs[j] for j, h in enumerate(keys)})
        n_new += len(keys)
    return n_new


def embed_dataframe(
    df: pd.DataFrame,
    text_column: str,
    model_name: str | None = None,
    **kwargs,
) -> np.ndarray:
    return embed_texts(
        df[text_column].astype(str).tolist(),
        model_name=model_name,
        **kwargs,
    )


def cache_inventory(config: EmbedConfig | None = None) -> dict:
    config = config or EmbedConfig()
    root = resolve_cache_dir() / config.fingerprint()
    if not root.exists():
        return {"root": str(root), "shards": 0, "rows": 0, "bytes": 0}
    shards = list(root.glob("shard_*.parquet"))
    rows = 0
    nbytes = 0
    for s in shards:
        nbytes += s.stat().st_size
        try:
            rows += pq.read_metadata(s).num_rows
        except Exception:
            pass
    return {
        "root": str(root),
        "shards": len(shards),
        "rows": rows,
        "bytes": nbytes,
    }
