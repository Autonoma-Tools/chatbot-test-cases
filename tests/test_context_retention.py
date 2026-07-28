"""Multi-turn context tests.

Context breaks across turns, not within one, so every case here drives 3-4 turns
on a single session and asserts the resolved invariant AFTER the final turn,
never mid-conversation. Four cases, one per row of the article's Context
Retention table.

Parametrize is deliberately not used in this file: the turn sequences have
different lengths and different invariants, so four explicit functions read far
better than a table of ragged tuples. The comment idiom is identical to the rest
of the repo:

    input -> invariant -> assertion

Each function builds its OWN FakeChatbotClient. One client is one conversation,
and sharing one across tests would let turn state leak between cases, which is
the single easiest way to write a context test that passes for the wrong reason.
"""

from chatbot.client import FakeChatbotClient


def test_return_then_offered_fix_then_confirmed_return():
    """Turn 1 opens a return, turn 2 offers an alternative, turn 3 overrides back
    to the original ask. Assert on the FINAL resolved action, not on any single
    turn's reply text.

    input:     'I want to return the blue jacket'
            -> 'actually, just the sleeve is torn, can you fix it instead'
            -> 'no wait, just return it'
    invariant: entities['final_action'] == 'RETURN', still scoped to the jacket
    assertion: equality on the resolved action after the last turn

    The trap: after turn 2 a bot has a repair in flight, and plenty of bots let
    the repair win because it was mentioned most recently. Turn 3 is the customer
    correcting them, and 'return it' has no noun in it at all, so resolving it
    requires the jacket from turn 1 to still be the active referent.
    """
    client = FakeChatbotClient()

    client.send("I want to return the blue jacket")
    client.send("actually, just the sleeve is torn, can you fix it instead")
    response = client.send("no wait, just return it")

    assert response.entities.get("final_action") == "RETURN"
    # And the action is still attached to the right item, not floating free.
    assert response.referent.get("item") == "blue jacket"


def test_order_id_survives_several_turns():
    """The order id is stated once, then two unrelated turns happen, then a status
    question relies on it. If order_id is None or wrong here, context was dropped.

    input:     'My order number is 48213'
            -> "what's your return policy"
            -> 'do you ship internationally'
            -> "What's the status of my order?"
    invariant: order_id == '48213' on the final turn
    assertion: equality on the session slot, not on the reply text

    Note the two intervening turns are genuinely off-topic. A bot that keeps only
    the last turn or two in its window passes this test with one filler turn and
    fails it with two, which is why the filler is here rather than trimmed away.
    """
    client = FakeChatbotClient()

    client.send("My order number is 48213")
    client.send("what's your return policy")
    client.send("do you ship internationally")
    response = client.send("What's the status of my order?")

    assert response.order_id == "48213"


def test_budget_survives_topic_switch_and_return():
    """Budget stated, then a full topic switch, then a return to the original
    topic. budget_filter must equal 50 on the final turn.

    input:     'My budget is under $50'
            -> "what's your shipping time to Canada"
            -> 'show me options in my budget'
    invariant: budget_filter == 50
    assertion: equality on the typed slot value (an int, not the string '$50')

    'show me options in my budget' contains no number. The only way to answer it
    correctly is to still be holding the constraint from turn 1.
    """
    client = FakeChatbotClient()

    client.send("My budget is under $50")
    client.send("what's your shipping time to Canada")
    response = client.send("show me options in my budget")

    assert response.budget_filter == 50


def test_pronoun_resolves_to_current_referent():
    """'the left one' must resolve against the CURRENT referent (red shoes, size
    9), not reset it.

    input:     "I'm looking at the red shoes"
            -> 'show me in a size 9'
            -> 'what about the left one'
    invariant: referent still carries product == 'red shoes' and size == 9
    assertion: equality on both referent keys after the ambiguous third turn

    The failure mode this catches is a bot that treats an ambiguous reference as a
    new topic and clears its slots, so the customer gets asked "which product?"
    two turns after telling it.
    """
    client = FakeChatbotClient()

    client.send("I'm looking at the red shoes")
    client.send("show me in a size 9")
    response = client.send("what about the left one")

    assert response.referent.get("product") == "red shoes"
    assert response.referent.get("size") == 9
