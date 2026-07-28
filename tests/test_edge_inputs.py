"""Edge inputs: the messages real customers send that demo scripts never contain.

Six cases, one per row of the article's Edge Input table: a typo, slang, a
double-barreled request, empty input, an overlong rambling message, and a
non-English input.

    input -> invariant -> assertion

Cases 1-4 share one assertion shape (field == value) so they live in a
parametrized table. Cases 5 and 6 have structurally different assertions, so they
are separate named functions rather than being forced into the table.

Nothing in this file may crash the pytest process. "It raised a TypeError on
empty input" is the most common real-world version of this category's bug, so the
tests are written to prove the calls RETURN, not merely that they return
something particular.
"""

import pytest

from chatbot.client import FakeChatbotClient

# (input_text, expected_field, expected_value)
EDGE_CASES = [
    # Typos in both words. 'cancle' and 'odrer' are one transposition each from
    # 'cancel' and 'order'. Exact keyword matching misses this; the invariant is
    # that routing does not.
    ("cancle my odrer plz", "intent", "CANCEL_ORDER"),
    # Slang. 'axe this order' means cancel it. Nothing in the sentence is a
    # cancellation keyword.
    ("yo just axe this order for me", "intent", "CANCEL_ORDER"),
    # Double-barreled with an ambiguous first clause ('return this' has no noun).
    # The invariant is the full set: answering the return and dropping the address
    # change is exactly the half-answer this case exists to catch.
    (
        "return this, also change my shipping address",
        "intents",
        {"RETURN", "ADDRESS_CHANGE"},
    ),
    # Empty input. Someone hit enter. The invariant is a prompt, not a crash and
    # not a hallucinated intent.
    ("", "response_type", "prompt_for_input"),
]


# --------------------------------------------------------------------------- #
# Case 5 data, built deterministically rather than generated per run so the
# fixture is reviewable in a diff. The routing keywords sit in the MIDDLE of the
# message, past the point where a naive client would have truncated.
# --------------------------------------------------------------------------- #

FILLER_SENTENCE = (
    "I have been shopping with you since college and I usually have a great "
    "experience but this week has been genuinely confusing and I would like to "
    "explain the whole sequence of events before I get to what I actually need "
    "from you today. "
)

OVERLONG_MESSAGE = (
    (FILLER_SENTENCE * 6) + "cancel my order please. " + (FILLER_SENTENCE * 6)
)


@pytest.mark.parametrize("message, field, expected", EDGE_CASES)
def test_edge_input_resolves_or_fails_gracefully(message, field, expected):
    """input -> invariant -> assertion, for the four cases with a uniform shape.

    Add a case by appending a tuple to EDGE_CASES.
    """
    client = FakeChatbotClient()

    response = client.send(message)

    actual = getattr(response, field)
    assert actual == expected, (
        f"input={message!r} expected {field}={expected!r} got {actual!r}"
    )


def test_empty_input_returns_instead_of_raising():
    """Explicitly separate from the table above, because the table only proves the
    RETURN VALUE is right. This proves the call returns at all.

    input:     '' (and a whitespace-only variant)
    invariant: no exception, and response_type == 'prompt_for_input'
    assertion: the call completes and the field matches
    """
    client = FakeChatbotClient()

    for blank in ("", "   ", "\n\t "):
        response = client.send(blank)
        assert response is not None
        assert response.response_type == "prompt_for_input"
        assert response.intent is None


def test_overlong_message_still_routes_correctly():
    """input:     a ~500-word rambling message with 'cancel my order' buried in
               the middle of it
    invariant: intent == 'CANCEL_ORDER', resolved from the FULL string
    assertion: equality on intent, plus guards proving the message really is long
               and really does hide the ask past the front of the text

    The bug this catches is silent truncation. A client that slices the message to
    the first N characters, or to the first sentence, drops the actual request and
    then routes on the preamble, which is pure noise. The two guard assertions
    below exist so this test cannot degenerate into a length check that always
    passes: they pin the message at over 2000 characters and confirm the keywords
    sit well past the start.
    """
    word_count = len(OVERLONG_MESSAGE.split())
    keyword_offset = OVERLONG_MESSAGE.index("cancel my order")

    # Guards: the fixture genuinely is overlong, and the ask genuinely is buried.
    assert word_count > 450, f"fixture is only {word_count} words"
    assert len(OVERLONG_MESSAGE) > 2000, f"fixture is only {len(OVERLONG_MESSAGE)} chars"
    assert keyword_offset > 1000, f"keywords appear at char {keyword_offset}"

    client = FakeChatbotClient()

    response = client.send(OVERLONG_MESSAGE)

    assert response.intent == "CANCEL_ORDER"
    # The client flags the message as overlong for observability, but flagging is
    # not the same as truncating: routing above already proved it read the whole
    # string.
    assert response.entities.get("overlong") is True


def test_non_english_input_does_not_crash_or_misroute():
    """input:     'Puedo cancelar mi pedido?'
    invariant: either the correct route (CANCEL_ORDER) or an honest fallback,
               but never a confident WRONG route
    assertion: an OR over the two acceptable outcomes, plus a negative check

    This repo's fake client makes a best-effort attempt at the Spanish phrase by
    matching 'cancelar' + 'pedido', so in practice it routes to CANCEL_ORDER. The
    test intentionally accepts either a correct route or a graceful, honest
    fallback, because for a real bot without Spanish support "I can't help in this
    language" is a correct answer. What is never acceptable is confidently routing
    to something unrelated, so REFUND_REQUEST and friends are asserted against
    explicitly.
    """
    client = FakeChatbotClient()

    response = client.send("Puedo cancelar mi pedido?")

    assert response.intent == "CANCEL_ORDER" or response.response_type in (
        "fallback_admit_unknown",
        "normal",
    )
    assert response.intent not in {
        "REFUND_REQUEST",
        "TRACK_ORDER",
        "DISCOUNT_REQUEST",
        "ADDRESS_CHANGE",
        "CANCEL_SUBSCRIPTION",
    }
