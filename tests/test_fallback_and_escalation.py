"""Fallback and escalation: what the bot does when it should stop trying.

Four cases, one per row of the article's Fallback and Escalation table. Every
invariant in this file is a boolean or a state value. None of them is a string,
because "did it apologise nicely" is not the property under test. The property
under test is whether the bot handed off, admitted ignorance, or actually
changed something.

    input -> invariant -> assertion
"""

from chatbot.client import TURN_CAP, FakeChatbotClient


def test_fraud_report_creates_escalation_ticket():
    """input:     'Someone charged my card twice, this is fraud'
    invariant: escalation_ticket_created == True
    assertion: a boolean field, not the apology wording in response.text

    A bot can produce a beautifully empathetic paragraph about how seriously it
    takes fraud and still create no ticket. The empathy is not the deliverable.
    """
    client = FakeChatbotClient()

    response = client.send("Someone charged my card twice, this is fraud")

    assert response.escalation_ticket_created is True


def test_repeated_unresolved_issue_triggers_turn_cap():
    """input:     "This still isn't fixed" sent four times in one session
    invariant: escalation_triggered_by_cap == True once the cap is passed
    assertion: a boolean field, independent of the wording used

    TURN_CAP (imported above, default 3) is the number of times a customer may
    restate an unresolved issue before a handoff to a human is mandatory. Past the
    cap, escalation must be automatic. Note what this test does NOT assert: it
    does not care which words the customer used or which words the bot replied
    with, only that repetition of an unresolved issue forces the handoff. That
    keeps the test alive through every future rewording of both sides.
    """
    client = FakeChatbotClient()
    repeated_message = "This still isn't fixed"

    response = None
    for _ in range(TURN_CAP + 1):
        response = client.send(repeated_message)

    assert response.escalation_triggered_by_cap is True


def test_out_of_scope_question_admits_unknown_instead_of_guessing():
    """input:     'Whats the meaning of life'
    invariant: response_type == 'fallback_admit_unknown', and no transactional
               intent was fabricated
    assertion: equality on response_type plus a negative check on intent

    The dangerous failure here is not silence, it is confidence. A bot that maps
    an unanswerable question onto the nearest familiar intent will happily start
    cancelling something.
    """
    response = FakeChatbotClient().send("Whats the meaning of life")

    assert response.response_type == "fallback_admit_unknown"
    assert response.intent is None or response.intent not in {
        "CANCEL_ORDER",
        "REFUND_REQUEST",
        "CANCEL_SUBSCRIPTION",
    }


def test_subscription_cancellation_actually_changes_backend_state():
    """input:     'Cancel my subscription'
    invariant: backend_state['subscription_status'] == 'cancelled'
    assertion: equality on application state, NOT on the reply text

    ======================================================================
    READ THIS ONE. It is the case a response-level assertion cannot honestly
    check.

    response.text will read as a confident confirmation regardless of whether the
    cancellation actually happened. Print it and see: the fake client returns
    "All set, your subscription has been cancelled and you won't be billed
    again." A bot whose backend call silently failed returns a sentence exactly
    like that one. Any assertion over the reply text passes in both worlds, which
    means it is not a test, it is a mood.

    The only trustworthy assertion is against backend_state: the simulated
    equivalent of querying the real application or database and asking whether the
    row actually changed.

    In this repo backend_state is an in-memory dict on the fake client. In a real
    system this is the seam where a behavioral, running-application check (e.g.
    Autonoma) replaces a purely textual one, because something has to actually
    drive the app and read the resulting state back.
    ======================================================================
    """
    client = FakeChatbotClient()

    response = client.send("Cancel my subscription")

    assert response.backend_state["subscription_status"] == "cancelled"
