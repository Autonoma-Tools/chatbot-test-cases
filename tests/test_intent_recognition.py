"""Intent recognition: does the bot route the request to the right handler?

Five cases, one per row of the article's Intent Recognition table. Every row
follows the same idiom used throughout this repo:

    input -> invariant -> assertion

The invariant is always a ROUTED FIELD (``intent`` or ``intents``), never
``response.text``. That distinction is the whole point of the category: the same
correct intent can be phrased a dozen different ways, and a reply-text assertion
turns every harmless rewording into a red build.

To add a case, append a tuple to INTENT_CASES. No new test function needed.
"""

import pytest

from chatbot.client import FakeChatbotClient

# (input_text, expected_field, expected_value)
INTENT_CASES = [
    # Plain, unambiguous phrasing. The baseline.
    ("Cancel my order", "intent", "CANCEL_ORDER"),
    # Mid-sentence reversal. The customer never says "cancel"; they say
    # "scratch that" and "undo". Routing has to catch the intent, not the keyword.
    ("Actually, scratch that, undo the order", "intent", "CANCEL_ORDER"),
    # Lowercase, no punctuation, a typo ("anymroe"), and the ask is at the end.
    ("i dont want this anymroe, refund me", "intent", "REFUND_REQUEST"),
    # Double-barreled: two unrelated asks in one message. The invariant is the
    # full SET, because answering only the first half is a real failure mode.
    (
        "wheres my package, also can i get a discount",
        "intents",
        {"TRACK_ORDER", "DISCOUNT_REQUEST"},
    ),
    # Small talk. The invariant here is as much about what does NOT happen
    # (see the extra guard in the test body) as about GREETING itself.
    ("hey", "intent", "GREETING"),
]


@pytest.mark.parametrize("message, field, expected", INTENT_CASES)
def test_intent_is_recognized(message, field, expected):
    """input -> invariant -> assertion.

    input:     the customer's literal message
    invariant: the routed field the bot must resolve
    assertion: exact equality on that field, never on response.text
    """
    client = FakeChatbotClient()

    response = client.send(message)

    actual = getattr(response, field)
    assert actual == expected, (
        f"input={message!r} expected {field}={expected!r} got {actual!r}"
    )


def test_greeting_does_not_fabricate_a_transactional_intent():
    """Belt and suspenders for the 'hey' case.

    A bot that is over-eager to be helpful will map a bare greeting onto whatever
    intent is most common in its training data, which in a support bot is usually
    a cancellation or a refund. Asserting GREETING is true is not enough; assert
    that the confident wrong answers did not happen either.

    input:     'hey'
    invariant: intent is GREETING and specifically NOT a transactional route
    assertion: equality plus two explicit negative checks
    """
    response = FakeChatbotClient().send("hey")

    assert response.intent == "GREETING"
    assert response.intent != "CANCEL_ORDER"
    assert response.intent != "REFUND_REQUEST"
