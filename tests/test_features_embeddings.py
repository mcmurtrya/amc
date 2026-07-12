"""Tests for the chunked-Parquet embedding cache.

The sentence-transformers model itself is mocked - we never load the real
network model in the test suite. The cache, sharding, hash math, and
cache-directory resolution are all exercised against synthetic embeddings.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from metals.features.embeddings import (
    DEFAULT_DTYPE,
    DEFAULT_MODEL,
    SHARD_PREFIX_LEN,
    EmbedConfig,
    ParquetEmbeddingCache,
    _hash_hex,
    _shard_prefix,
    cache_embeddings,
    cache_inventory,
    embed_dataframe,
    embed_texts,
    resolve_cache_dir,
)

# ---------------------------------------------------------------------------
# Default model + config
# ---------------------------------------------------------------------------


def test_default_model_is_minilm():
    """MiniLM is the new default after the 2026-06-23 storage-driven swap."""
    assert "MiniLM" in DEFAULT_MODEL


def test_default_dtype_is_fp16():
    assert DEFAULT_DTYPE == "fp16"


def test_embed_config_fingerprint_changes_with_model():
    a = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2").fingerprint()
    b = EmbedConfig(model_name="sentence-transformers/all-mpnet-base-v2").fingerprint()
    assert a != b


def test_embed_config_fingerprint_changes_with_dtype():
    a = EmbedConfig(dtype="fp16").fingerprint()
    b = EmbedConfig(dtype="fp32").fingerprint()
    assert a != b


# ---------------------------------------------------------------------------
# Cache directory resolution + OneDrive avoidance
# ---------------------------------------------------------------------------


def test_resolve_cache_dir_uses_env_override(tmp_path):
    env = {"METALS_EMBEDDING_CACHE_DIR": str(tmp_path / "custom" / "cache")}
    out = resolve_cache_dir(env=env)
    assert out == Path(env["METALS_EMBEDDING_CACHE_DIR"])


def test_resolve_cache_dir_warns_when_path_inside_onedrive(tmp_path):
    bad = tmp_path / "Users" / "alice" / "OneDrive" / "metals_cache"
    env = {"METALS_EMBEDDING_CACHE_DIR": str(bad)}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        resolve_cache_dir(env=env)
    msgs = [str(w.message) for w in captured]
    assert any("sync-managed" in m and "onedrive" in m.lower() for m in msgs)


def test_resolve_cache_dir_warns_for_other_sync_engines(tmp_path):
    for token in ("Dropbox", "iCloud", "GoogleDrive"):
        bad = tmp_path / "Users" / "alice" / token / "cache"
        env = {"METALS_EMBEDDING_CACHE_DIR": str(bad)}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            resolve_cache_dir(env=env)
        assert any("sync-managed" in str(w.message) for w in captured), token


def test_resolve_cache_dir_no_warning_for_clean_path(tmp_path):
    env = {"METALS_EMBEDDING_CACHE_DIR": str(tmp_path / "plain" / "cache")}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        resolve_cache_dir(env=env)
    assert not any("sync-managed" in str(w.message) for w in captured)


def test_resolve_cache_dir_default_uses_localappdata_on_windows():
    env = {"LOCALAPPDATA": r"C:\Users\test\AppData\Local"}
    with patch("metals.features.embeddings.sys") as fake_sys:
        fake_sys.platform = "win32"
        out = resolve_cache_dir(env=env)
    parts_lower = [p.lower() for p in out.parts]
    assert "metals" in parts_lower
    assert "embeddings" in parts_lower


def test_resolve_cache_dir_default_uses_xdg_cache_on_unix():
    env = {}  # nothing set
    with patch("metals.features.embeddings.sys") as fake_sys:
        fake_sys.platform = "linux"
        out = resolve_cache_dir(env=env)
    parts_lower = [p.lower() for p in out.parts]
    assert ".cache" in parts_lower
    assert "metals" in parts_lower


# ---------------------------------------------------------------------------
# Sharding and hash math
# ---------------------------------------------------------------------------


def test_hash_hex_is_64_chars():
    assert len(_hash_hex("hello world")) == 64


def test_hash_hex_is_deterministic():
    assert _hash_hex("xyz") == _hash_hex("xyz")


def test_hash_hex_differs_for_different_texts():
    assert _hash_hex("a") != _hash_hex("b")


def test_shard_prefix_length_matches_constant():
    h = _hash_hex("some text")
    assert len(_shard_prefix(h)) == SHARD_PREFIX_LEN


def test_shard_prefix_distribution_is_roughly_uniform():
    """Sanity check that the first-3-hex distribution doesn't pile into one bucket."""
    counts: dict[str, int] = {}
    for i in range(2000):
        h = _hash_hex(f"text-{i}")
        counts[_shard_prefix(h)] = counts.get(_shard_prefix(h), 0) + 1
    # Even with 2000 samples across 4096 buckets, no single bucket should
    # collect more than ~10 items by luck (binomial mean ~0.5).
    assert max(counts.values()) < 15


# ---------------------------------------------------------------------------
# ParquetEmbeddingCache round-trip
# ---------------------------------------------------------------------------


def test_parquet_cache_write_then_read_round_trip(tmp_path):
    cfg = EmbedConfig(dtype="fp32")  # easier to compare exactly
    cache = ParquetEmbeddingCache(tmp_path, cfg)
    items = {
        _hash_hex("alpha"): np.array([0.1, 0.2, 0.3], dtype=np.float32),
        _hash_hex("beta"): np.array([0.4, 0.5, 0.6], dtype=np.float32),
        _hash_hex("gamma"): np.array([0.7, 0.8, 0.9], dtype=np.float32),
    }
    n = cache.write_many(items)
    assert n == 3
    out = cache.read_many(list(items.keys()))
    assert set(out.keys()) == set(items.keys())
    for k in items:
        np.testing.assert_allclose(out[k], items[k], atol=1e-6)


def test_parquet_cache_read_returns_empty_for_missing(tmp_path):
    cache = ParquetEmbeddingCache(tmp_path, EmbedConfig())
    out = cache.read_many([_hash_hex("not-there")])
    assert out == {}


def test_parquet_cache_handles_fp16_round_trip_with_acceptable_loss(tmp_path):
    cfg = EmbedConfig(dtype="fp16")
    cache = ParquetEmbeddingCache(tmp_path, cfg)
    vec = np.array([0.123, -0.456, 0.789], dtype=np.float32)
    h = _hash_hex("text")
    cache.write_many({h: vec})
    out = cache.read_many([h])
    np.testing.assert_allclose(out[h], vec, atol=5e-3)  # fp16 precision


def test_parquet_cache_creates_shard_file_with_expected_name(tmp_path):
    cache = ParquetEmbeddingCache(tmp_path, EmbedConfig())
    h = _hash_hex("alpha")
    cache.write_many({h: np.zeros(4, dtype=np.float32)})
    expected = cache.shard_path(_shard_prefix(h))
    assert expected.exists()
    assert expected.name.startswith("shard_") and expected.suffix == ".parquet"


def test_parquet_cache_merges_writes_to_same_shard(tmp_path):
    """Two write_many() calls touching the same shard should preserve both."""
    cache = ParquetEmbeddingCache(tmp_path, EmbedConfig(dtype="fp32"))
    # We need two hashes that fall in the same shard. Construct by trial.
    h_a = _hash_hex("alpha")
    target_prefix = _shard_prefix(h_a)
    h_b = None
    for i in range(20000):
        cand = _hash_hex(f"candidate-{i}")
        if _shard_prefix(cand) == target_prefix and cand != h_a:
            h_b = cand
            break
    assert h_b is not None, "could not synthesise a same-shard pair in 20000 tries"
    cache.write_many({h_a: np.ones(3, dtype=np.float32)})
    # Fresh cache to ensure the second write loads from disk
    cache2 = ParquetEmbeddingCache(tmp_path, EmbedConfig(dtype="fp32"))
    cache2.write_many({h_b: 2 * np.ones(3, dtype=np.float32)})
    cache3 = ParquetEmbeddingCache(tmp_path, EmbedConfig(dtype="fp32"))
    out = cache3.read_many([h_a, h_b])
    assert set(out.keys()) == {h_a, h_b}


def test_parquet_cache_atomic_write_no_stray_tmp(tmp_path):
    cache = ParquetEmbeddingCache(tmp_path, EmbedConfig())
    h = _hash_hex("anything")
    cache.write_many({h: np.ones(4, dtype=np.float32)})
    leftovers = list(cache.root.glob("*.parquet.tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# embed_texts: public API with mocked encoder
# ---------------------------------------------------------------------------


class _FakeModel:
    """Stand-in for a sentence-transformers model."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.encode_calls = 0
        self._counter = 0

    def encode(
        self,
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        self.encode_calls += 1
        # Produce deterministic-but-distinct vectors per (text, call-counter).
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hash(t) % (2**16)
            out[i, :] = (h + np.arange(self.dim)) / 1000.0
            if normalize_embeddings:
                out[i, :] /= max(np.linalg.norm(out[i, :]), 1e-9)
        return out


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("METALS_EMBEDDING_CACHE_DIR", str(tmp_path))
    yield tmp_path
    # cleanup module-level model cache so tests don't leak state
    from metals.features import embeddings as emb

    emb._model_cache.clear()


def test_embed_texts_empty_input(isolated_cache):
    arr = embed_texts([])
    assert arr.shape == (0, 0)


def test_embed_texts_uses_cache_on_repeat(isolated_cache):
    fake = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake):
        a = embed_texts(["foo", "bar", "baz"])
        b = embed_texts(["foo", "bar", "baz"])  # should hit cache
    assert a.shape == (3, 8)
    # Round-trip through fp16 storage introduces small precision loss.
    np.testing.assert_allclose(a, b, atol=5e-3)
    # second call should not encode anything new
    assert fake.encode_calls == 1


def test_embed_texts_handles_empty_string_as_zero_vec(isolated_cache):
    fake = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake):
        arr = embed_texts(["foo", "", "bar"])
    assert arr.shape == (3, 8)
    assert np.allclose(arr[1], 0.0)


def test_embed_texts_no_cache_does_not_persist(isolated_cache):
    fake = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake):
        embed_texts(["uncached"], use_cache=False)
    inv = cache_inventory(EmbedConfig())
    assert inv["rows"] == 0


def test_embed_texts_writes_to_cache_directory(isolated_cache):
    fake = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake):
        embed_texts(["cached"])
    inv = cache_inventory(EmbedConfig())
    assert inv["rows"] == 1
    assert inv["shards"] == 1


def test_embed_texts_model_swap_invalidates_cache(isolated_cache):
    """Different model -> different fingerprint -> different cache dir."""
    fake_a = _FakeModel(dim=8)
    fake_b = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake_a):
        embed_texts(["text"], model_name="model-a")
    with patch("metals.features.embeddings._get_model", return_value=fake_b):
        embed_texts(["text"], model_name="model-b")  # cache miss expected
    assert fake_a.encode_calls == 1
    assert fake_b.encode_calls == 1


def test_embed_dataframe_preserves_index_order(isolated_cache):
    fake = _FakeModel(dim=8)
    df = pd.DataFrame({"headline": ["a", "b", "c", "a"]}, index=[10, 20, 30, 40])
    with patch("metals.features.embeddings._get_model", return_value=fake):
        arr = embed_dataframe(df, "headline")
    assert arr.shape == (4, 8)
    # rows 0 and 3 (both "a") should be identical
    np.testing.assert_array_equal(arr[0], arr[3])


def test_env_var_chooses_model_name(isolated_cache, monkeypatch):
    monkeypatch.setenv("METALS_EMBEDDING_MODEL", "env-selected-model")
    fake = _FakeModel(dim=4)
    captured: dict = {}

    def fake_get(name):
        captured["name"] = name
        return fake

    with patch("metals.features.embeddings._get_model", side_effect=fake_get):
        embed_texts(["t"])
    assert captured["name"] == "env-selected-model"


# ---------------------------------------------------------------------------
# cache_embeddings — streaming cache warm (no full-corpus vstack)
# ---------------------------------------------------------------------------
def test_cache_embeddings_populates_cache_and_counts(isolated_cache):
    fake = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake):
        n_new = cache_embeddings(["a", "b", "c"], sub_chunk=2)
    assert n_new == 3
    inv = cache_inventory(EmbedConfig())
    assert inv["rows"] == 3


def test_cache_embeddings_is_idempotent_on_repeat(isolated_cache):
    fake = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake):
        first = cache_embeddings(["a", "b", "c"])
        second = cache_embeddings(["a", "b", "c"])  # all cache hits now
    assert first == 3
    assert second == 0


def test_cache_embeddings_dedups_within_block(isolated_cache):
    fake = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake):
        # 5 inputs, 2 distinct -> only 2 newly encoded
        n_new = cache_embeddings(["x", "y", "x", "y", "x"])
    assert n_new == 2


def test_cache_embeddings_matches_embed_texts_values(isolated_cache):
    """A cache warmed by cache_embeddings serves embed_texts without re-encoding."""
    fake = _FakeModel(dim=8)
    with patch("metals.features.embeddings._get_model", return_value=fake):
        cache_embeddings(["foo", "bar"])
        calls_after_warm = fake.encode_calls
        arr = embed_texts(["foo", "bar"])  # should be pure cache hits
    assert arr.shape == (2, 8)
    assert fake.encode_calls == calls_after_warm  # no new encodes


def test_cache_embeddings_empty_input(isolated_cache):
    assert cache_embeddings([]) == 0
