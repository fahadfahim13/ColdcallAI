"""Unit tests for classify_intent() — keyword intent classifier."""

import pytest
from core.call.intent import classify_intent


def test_decline_goodbye():
    assert classify_intent("not interested, goodbye") == "decline"


def test_decline_bye():
    assert classify_intent("Okay bye, thanks anyway") == "decline"


def test_objection_competitor():
    assert classify_intent("we already use a competitor for this") == "objection"


def test_interested_book():
    assert classify_intent("yes let's book Tuesday") == "interested"


def test_callback():
    assert classify_intent("call me back next week, I'm busy right now") == "callback"


def test_hostile():
    assert classify_intent("leave me alone, stop bothering me") == "hostile"


def test_question_no_match():
    # No keyword match → falls through to "question"
    assert classify_intent("what are your support hours?") == "question"


def test_empty_string():
    assert classify_intent("") == "question"


def test_case_insensitive():
    assert classify_intent("GOODBYE, NOT INTERESTED") == "decline"


def test_decline_priority_over_interested():
    # "not interested" should win over any embedded "interested"
    assert classify_intent("I'm not interested") == "decline"
