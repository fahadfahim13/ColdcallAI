"""
Voice variant registry.

Each VoiceVariant maps our internal slug (stored in DB, used for A/B attribution)
to an InWorld voice_id (passed to InWorldTTSProvider.synthesize()).

Variants are loaded from settings at startup — any variant whose voice_id is empty
is excluded from the registry so the orchestrator never selects an unconfigured voice.

The "test" variant is the exception: it falls back to INWORLD_VOICE_FALLBACK so
local dev/testing always has at least one working voice without credentials for all 5.

Usage:
    registry = build_registry(settings)
    variant = get_variant(registry, "warm")
    chunks = tts.synthesize(text, voice_id=variant.voice_id)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


@dataclass(frozen=True)
class VoiceVariant:
    id: str          # slug — stored in DB (call_result.voice_variant_id)
    voice_id: str    # InWorld voice_id string passed to TTS
    label: str       # human-readable display name
    description: str # persona / tone notes (shown in dashboard, used by QA)


# Ordered by expected frequency of use in A/B testing
_VARIANT_DEFINITIONS = [
    ("formal",    "label", "Formal",    "Professional, authoritative tone — works well in B2B finance/legal"),
    ("warm",      "label", "Warm",      "Friendly, approachable — performs well in healthcare/SMB"),
    ("energetic", "label", "Energetic", "Upbeat, enthusiastic — suited to fast-moving SaaS / startup pitches"),
    ("calm",      "label", "Calm",      "Measured, trustworthy — good for high-ticket / risk-averse buyers"),
    ("casual",    "label", "Casual",    "Conversational, peer-to-peer — works for founder/owner calls"),
    ("test",      "label", "Test",      "Fallback voice for local dev — always loaded if any InWorld key present"),
]

# Map from variant slug → settings attribute name for its voice_id
_SETTINGS_ATTR: dict[str, str] = {
    "formal":    "inworld_voice_formal",
    "warm":      "inworld_voice_warm",
    "energetic": "inworld_voice_energetic",
    "calm":      "inworld_voice_calm",
    "casual":    "inworld_voice_casual",
    "test":      "inworld_voice_test",
}

VoiceRegistry = dict[str, VoiceVariant]


def build_registry(settings: "Settings") -> VoiceRegistry:
    """
    Build the active voice registry from settings.
    Variants whose voice_id env var is empty are silently excluded.
    Raises RuntimeError if no variants are configured at all.
    """
    registry: VoiceRegistry = {}

    for slug, _, label, description in _VARIANT_DEFINITIONS:
        attr = _SETTINGS_ATTR[slug]
        voice_id: str = getattr(settings, attr, "")
        if voice_id:
            registry[slug] = VoiceVariant(
                id=slug,
                voice_id=voice_id,
                label=label,
                description=description,
            )

    if not registry:
        raise RuntimeError(
            "No InWorld voice variants configured. "
            "Set at least one INWORLD_VOICE_* env var in .env. "
            "See .env.example for the full list."
        )

    return registry


def get_variant(registry: VoiceRegistry, variant_id: str) -> VoiceVariant:
    """Return a variant by slug. Raises KeyError if not in registry."""
    try:
        return registry[variant_id]
    except KeyError:
        available = ", ".join(sorted(registry))
        raise KeyError(
            f"Voice variant {variant_id!r} not found. "
            f"Available: {available}"
        ) from None


def available_ids(registry: VoiceRegistry) -> list[str]:
    """Return slugs of all configured variants, in definition order."""
    order = [slug for slug, *_ in _VARIANT_DEFINITIONS]
    return [slug for slug in order if slug in registry]
