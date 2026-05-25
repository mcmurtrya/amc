"""Tests for the embeddings wrapper (use a tiny mock model — no downloads)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metals.features.embeddings import (
    EmbedConfig,
    _hash_text,
    embed_texts,
)


class _FakeSentenceTransformer:
    """Stand-in for sentence-transformers that returns deterministic vectors.

    Each text gets a 4-D vector derived from its first 4 characters, so
    different texts produce different vectors and identical texts produce
    identical vectors.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t[:4]):
                out[i, j] = ord(ch) / 256.0
        return out


@pytest.fixture
def fake_st(monkeypatch):
    """Replace the lazy SentenceTransformer import with a deterministic stub."""
    import metals.features.embeddings as embeddings_mod

    # Reset module-level cache between tests so model swaps take effect.
    embeddings_mod._model_cache.clear()

    fake_module = MagicMock()
    fake_module.SentenceTransformer = _FakeSentenceTransformer

    # Replace the lazy import by injecting into sys.modules. Embeddings does
    # `from sentence_transformers import SentenceTransformer` inside _get_model.
    import sys
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    yield


def test_embed_texts_shape_and_dtype(fake_st):
    out = embed_texts(["hello world", "goodbye"], use_cache=False)
    assert out.shape == (2, 4)
    assert out.dtype == np.float32


def test_embed_texts_returns_in_input_order(fake_st):
    texts = ["alpha", "beta", "gamma"]
    out = embed_texts(texts, use_cache=False)
    assert out[0, 0] == pytest.approx(ord("a") / 256.0)
    assert out[1, 0] == pytest.approx(ord("b") / 256.0)
    assert out[2, 0] == pytest.approx(ord("g") / 256.0)


def test_embed_texts_empty_input(fake_st):
    out = embed_texts([], use_cache=False)
    assert out.shape == (0, 0)


def test_embed_texts_empty_string_yields_zero_vector(fake_st):
    out = embed_texts(["", "hello"], use_cache=False)
    assert out.shape == (2, 4)
    assert np.allclose(out[0], 0.0)
    assert not np.allclose(out[1], 0.0)


def test_embed_texts_cache_hit_avoids_recompute(fake_st, tmp_path, monkeypatch):
    """Second call with same text must hit cache, not the encoder."""
    import metals.features.embeddings as embeddings_mod
    monkeypatch.setattr(embeddings_mod, "CACHE_ROOT", tmp_path)
    embeddings_mod._model_cache.clear()

    # First call: populates cache
    first = embed_texts(["abc"], use_cache=True)
    assert first.shape == (1, 4)

    # Wrap the fake encoder to count calls
    import sys
    fake_st_module = sys.modules["sentence_transformers"]
    original_encode = _FakeSentenceTransformer.encode
    call_count = {"n": 0}

    def counting_encode(self, texts, **kwargs):
        call_count["n"] += 1
        return original_encode(self, texts, **kwargs)

    monkeypatch.setattr(_FakeSentenceTransformer, "encode", counting_encode)
    embeddings_mod._model_cache.clear()  # force a fresh model load

    # Second call: same input, should hit cache without invoking encode
    second = embed_texts(["abc"], use_cache=True)
    assert call_count["n"] == 0, "encoder was called despite full cache hit"
    assert np.allclose(first, second)


def test_embed_config_fingerprint_changes_with_model(fake_st):
    a = EmbedConfig(model_name="model-a")
    b = EmbedConfig(model_name="model-b")
    c = EmbedConfig(model_name="model-a", normalize=False)
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_hash_text_stable_and_distinct():
    assert _hash_text("abc") == _hash_text("abc")
    assert _hash_text("abc") != _hash_text("abd")
    assert len(_hash_text("anything")) == 64   # sha256 hex
