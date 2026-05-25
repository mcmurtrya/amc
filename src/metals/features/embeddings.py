"""Sentence-Transformers wrapper with on-disk caching.

Phase 3 step 3.6.

Design notes
------------

* Model loading is slow (~5–10 s for ``all-mpnet-base-v2``) so the wrapper
  caches the loaded model in-process and the embeddings on disk.
* On-disk cache is keyed by ``sha256(text)`` so identical strings across
  re-runs hit the cache. A separate ``model_config`` hash prefixes the cache
  directory so changing the model invalidates the cache without manual
  cleanup.
* GDELT GKG records contain a URL but no headline text. For Phase 3 we
  embed the URL slug as a (lossy) headline proxy; daily aggregation
  remains the dominant signal channel. The wrapper itself is text-agnostic
  — callers decide what to embed.

Usage
-----

>>> from metals.features.embeddings import embed_texts
>>> vecs = embed_texts(["hawkish FOMC drives gold lower",
...                     "central banks signal patience"])
>>> vecs.shape
(2, 768)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
CACHE_ROOT = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "embeddings"
)

_model_cache: dict[str, object] = {}


@dataclass(frozen=True)
class EmbedConfig:
    """Identifies the embedding model for cache-invalidation purposes."""

    model_name: str = DEFAULT_MODEL
    normalize: bool = True

    def fingerprint(self) -> str:
        """Stable short identifier mixing the model name and config."""
        blob = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _get_model(model_name: str):
    """Lazy-load and cache the sentence-transformers model in-process."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    from sentence_transformers import SentenceTransformer   # noqa: WPS433

    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_dir(config: EmbedConfig) -> Path:
    return CACHE_ROOT / config.fingerprint()


def _read_cache(hashes: Sequence[str], config: EmbedConfig) -> dict[str, np.ndarray]:
    """Return any cached vectors for the given text hashes."""
    cdir = _cache_dir(config)
    if not cdir.exists():
        return {}
    cached: dict[str, np.ndarray] = {}
    for h in hashes:
        p = cdir / f"{h}.npy"
        if p.exists():
            cached[h] = np.load(p)
    return cached


def _write_cache(
    pairs: Iterable[tuple[str, np.ndarray]],
    config: EmbedConfig,
) -> None:
    cdir = _cache_dir(config)
    cdir.mkdir(parents=True, exist_ok=True)
    for h, vec in pairs:
        np.save(cdir / f"{h}.npy", vec)


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    *,
    batch_size: int = 64,
    normalize: bool = True,
    use_cache: bool = True,
) -> np.ndarray:
    """Embed ``texts`` and return an ``(n, d)`` float32 array.

    Parameters
    ----------
    texts
        Sequence of strings. Empty strings are allowed but produce a zero
        vector (sentence-transformers refuses to embed empty input).
    model_name
        HuggingFace model id understood by sentence-transformers.
    batch_size
        Encoder batch size (only matters when there's a cache miss).
    normalize
        Whether to L2-normalise the output. Default True so cosine
        similarity reduces to a dot product downstream.
    use_cache
        Read and write the on-disk per-text cache. Set False for tests
        or one-shot work that doesn't merit cache pollution.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    config = EmbedConfig(model_name=model_name, normalize=normalize)
    hashes = [_hash_text(t) for t in texts]

    cached: dict[str, np.ndarray] = {}
    if use_cache:
        cached = _read_cache(hashes, config)

    missing_idx = [i for i, h in enumerate(hashes) if h not in cached]
    missing_texts = [texts[i] for i in missing_idx]
    missing_hashes = [hashes[i] for i in missing_idx]

    new_vecs: list[np.ndarray] = []
    if missing_texts:
        model = _get_model(model_name)
        non_empty_mask = [bool(t and t.strip()) for t in missing_texts]
        non_empty = [t for t, m in zip(missing_texts, non_empty_mask) if m]
        if non_empty:
            arr = model.encode(
                non_empty,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32, copy=False)
        else:
            arr = np.zeros((0, 1), dtype=np.float32)

        dim = arr.shape[1] if arr.size else 768
        zero = np.zeros((dim,), dtype=np.float32)
        cursor = 0
        for is_text in non_empty_mask:
            if is_text:
                new_vecs.append(arr[cursor])
                cursor += 1
            else:
                new_vecs.append(zero)
        if use_cache:
            _write_cache(zip(missing_hashes, new_vecs), config)

    # Re-assemble in original order.
    new_lookup = dict(zip(missing_hashes, new_vecs))
    ordered = [cached.get(h, new_lookup.get(h)) for h in hashes]
    if any(v is None for v in ordered):
        raise RuntimeError("internal: failed to fill all embedding slots")
    return np.vstack(ordered).astype(np.float32, copy=False)


def embed_dataframe(
    df: pd.DataFrame,
    text_column: str,
    model_name: str = DEFAULT_MODEL,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper: embed ``df[text_column]`` preserving index order."""
    return embed_texts(df[text_column].astype(str).tolist(),
                       model_name=model_name, **kwargs)
