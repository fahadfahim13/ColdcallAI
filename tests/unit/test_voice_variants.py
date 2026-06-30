"""Tests for core/voice/variants.py"""

import pytest
from unittest.mock import MagicMock

from core.voice.variants import (
    VoiceVariant,
    VoiceRegistry,
    build_registry,
    get_variant,
    available_ids,
)


def _settings(**kwargs) -> MagicMock:
    """Mock settings object. Unset attrs return empty string."""
    m = MagicMock()
    defaults = {
        "inworld_voice_formal": "",
        "inworld_voice_warm": "",
        "inworld_voice_energetic": "",
        "inworld_voice_calm": "",
        "inworld_voice_casual": "",
        "inworld_voice_test": "",
    }
    defaults.update(kwargs)
    for attr, val in defaults.items():
        setattr(m, attr, val)
    return m


# ---------------------------------------------------------------------------
# build_registry
# ---------------------------------------------------------------------------

def test_build_registry_single_variant():
    s = _settings(inworld_voice_warm="iw-voice-abc123")
    registry = build_registry(s)
    assert "warm" in registry
    assert registry["warm"].voice_id == "iw-voice-abc123"


def test_build_registry_excludes_empty_voice_ids():
    s = _settings(inworld_voice_formal="iw-formal", inworld_voice_warm="")
    registry = build_registry(s)
    assert "formal" in registry
    assert "warm" not in registry


def test_build_registry_all_six_variants():
    s = _settings(
        inworld_voice_formal="iw-f",
        inworld_voice_warm="iw-w",
        inworld_voice_energetic="iw-e",
        inworld_voice_calm="iw-c",
        inworld_voice_casual="iw-ca",
        inworld_voice_test="iw-t",
    )
    registry = build_registry(s)
    assert set(registry.keys()) == {"formal", "warm", "energetic", "calm", "casual", "test"}


def test_build_registry_raises_when_no_variants_configured():
    s = _settings()  # all empty
    with pytest.raises(RuntimeError, match="No InWorld voice variants configured"):
        build_registry(s)


def test_build_registry_variant_fields():
    s = _settings(inworld_voice_calm="iw-calm-x7")
    registry = build_registry(s)
    v = registry["calm"]
    assert v.id == "calm"
    assert v.voice_id == "iw-calm-x7"
    assert v.label  # non-empty
    assert v.description  # non-empty


def test_build_registry_returns_frozen_variants():
    s = _settings(inworld_voice_test="iw-t")
    registry = build_registry(s)
    v = registry["test"]
    with pytest.raises(Exception):  # frozen dataclass → FrozenInstanceError
        v.voice_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_variant
# ---------------------------------------------------------------------------

def test_get_variant_returns_correct_variant():
    s = _settings(inworld_voice_energetic="iw-energy")
    registry = build_registry(s)
    v = get_variant(registry, "energetic")
    assert v.voice_id == "iw-energy"


def test_get_variant_raises_keyerror_for_unknown_id():
    s = _settings(inworld_voice_formal="iw-f")
    registry = build_registry(s)
    with pytest.raises(KeyError, match="not found"):
        get_variant(registry, "nonexistent")


def test_get_variant_error_message_lists_available():
    s = _settings(inworld_voice_formal="iw-f", inworld_voice_warm="iw-w")
    registry = build_registry(s)
    with pytest.raises(KeyError) as exc_info:
        get_variant(registry, "missing")
    assert "formal" in str(exc_info.value)
    assert "warm" in str(exc_info.value)


def test_get_variant_raises_for_disabled_variant():
    # calm is empty → not in registry → KeyError
    s = _settings(inworld_voice_formal="iw-f")
    registry = build_registry(s)
    with pytest.raises(KeyError):
        get_variant(registry, "calm")


# ---------------------------------------------------------------------------
# available_ids
# ---------------------------------------------------------------------------

def test_available_ids_returns_configured_slugs():
    s = _settings(inworld_voice_warm="iw-w", inworld_voice_calm="iw-c")
    registry = build_registry(s)
    ids = available_ids(registry)
    assert set(ids) == {"warm", "calm"}


def test_available_ids_respects_definition_order():
    s = _settings(
        inworld_voice_formal="iw-f",
        inworld_voice_warm="iw-w",
        inworld_voice_energetic="iw-e",
    )
    registry = build_registry(s)
    ids = available_ids(registry)
    # definition order: formal, warm, energetic, calm, casual, test
    assert ids.index("formal") < ids.index("warm") < ids.index("energetic")


def test_available_ids_empty_for_empty_registry():
    # build_registry raises before we can test this, but let's test the function
    # directly with a manually constructed empty dict
    ids = available_ids({})
    assert ids == []
