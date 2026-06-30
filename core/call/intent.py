"""
Keyword intent classifier for user turns (Phase 1).

No LLM call — pure substring matching on lowercased text.
Phase 2 replaces this with the Qwen judge (T27).

Returns one of: "decline" | "hostile" | "callback" | "objection" | "interested" | "question"
"""

from __future__ import annotations

_DECLINE = {
    "goodbye", "bye", "good bye", "not interested", "remove me", "stop calling",
    "don't call", "take me off", "hang up", "end call",
    "no thank you", "no thanks", "not for us", "never mind",
    "thanks anyway", "thanks for calling", "not a good time",
    "please don't call", "do not call", "we're good", "we're all good",
}

_HOSTILE = {
    "stop", "leave me alone", "don't bother", "go away",
    "lawsuit", "report you", "stop bothering", "harassing",
}

_CALLBACK = {
    "call me back", "call back", "another time", "not now",
    "busy right now", "try again", "reach out later", "follow up",
}

_OBJECTION = {
    "expensive", "budget", "price", "afford", "competitor",
    "already use", "don't need", "not sure", "think about it",
    "need to discuss", "too costly", "can't justify", "contract",
}

_INTERESTED = {
    "interested", "tell me more", "yes", "sure", "okay",
    "sounds good", "book", "schedule", "let's do it", "sign up",
    "go ahead", "makes sense", "love to", "definitely",
}


def classify_intent(text: str) -> str:
    """Return the intent label for a user transcript."""
    if not text:
        return "question"
    t = text.lower()
    for phrase in _DECLINE:
        if phrase in t:
            return "decline"
    for phrase in _HOSTILE:
        if phrase in t:
            return "hostile"
    for phrase in _CALLBACK:
        if phrase in t:
            return "callback"
    for phrase in _OBJECTION:
        if phrase in t:
            return "objection"
    for phrase in _INTERESTED:
        if phrase in t:
            return "interested"
    return "question"
