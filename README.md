# Chatbot Test Cases: 23 Examples You Can Steal

A pytest-based chatbot test case library. 23 cases across five categories, a fake
chatbot client that needs no API key, and a side-by-side comparison of the three
assertion styles you can use on a chatbot reply.

> Companion code for the Autonoma blog post: **[Chatbot Test Cases: 23 Examples You Can Steal](https://getautonoma.com/blog/chatbot-test-cases)**

## Requirements

Python 3.10+ (CI pins 3.11) and `pytest`. Nothing else. No API key, no deployed
chatbot, no network access.

## Quickstart

```bash
git clone https://github.com/Autonoma-Tools/chatbot-test-cases.git
cd chatbot-test-cases
pip install pytest
pytest tests/ -v
```

Expected result on a fresh clone: **28 passed, 1 skipped**. The one skip is the
LLM-as-judge demonstration, which needs a model and skips cleanly without one.
There should be zero failures and zero errors.

## The idiom: input -> invariant -> assertion

Every case in this repo is written the same way, and the shape is the point.

1. **Input** is the literal message a customer sends. Not a cleaned-up version of
   it. Real inputs have typos, slang, two requests in one sentence, and sometimes
   nothing at all.
2. **Invariant** is the thing that must be true afterwards, expressed as a
   *structured field* rather than a sentence. `intent == "CANCEL_ORDER"`.
   `order_id == "48213"`. `escalation_ticket_created is True`.
   `backend_state["subscription_status"] == "cancelled"`.
3. **Assertion** checks that field, and only that field.

What you will not find anywhere in `tests/` is an assertion on `response.text`.
That is deliberate, and it is the single most important habit in chatbot testing.
A bot can cancel an order correctly and say so in a hundred different sentences.
An assertion against one hand-written expected sentence fails on ninety-nine of
them, all of which were correct. So the test suite would go red for a rewording,
your team would learn to ignore it, and it would then miss the real bug.

Pick a field that can only be right or wrong, and assert on that.

To see the failure mode side by side, read
[`tests/test_semantic_assertions.py`](tests/test_semantic_assertions.py). It runs
exact-match, semantic similarity, and LLM-as-judge against the same reply, and
documents what each one costs you.

## Project structure

```
chatbot/
  __init__.py
  client.py                        # FakeChatbotClient: deterministic, rule-based, offline
tests/
  test_intent_recognition.py       # 5 cases  - did it route to the right handler
  test_context_retention.py        # 4 cases  - does it remember across turns
  test_fallback_and_escalation.py  # 4 cases  - does it hand off instead of guessing
  test_edge_inputs.py              # 6 cases  - typos, slang, empty, overlong, non-English
  test_safety_and_injection.py     # 4 cases  - adversarial, run as its own suite
  test_semantic_assertions.py      # the three assertion styles, compared
data/
  test_cases.json                  # all 23 cases as data, for non-Python teammates
.github/workflows/
  chatbot-tests.yml                # runs the suite on every pull request
conftest.py                        # puts the repo root on sys.path
requirements.txt                   # pytest required, deepeval optional
```

## The 23 cases

| Category | Cases | What breaks | Assertion style |
|---|---|---|---|
| Intent recognition | 5 | Right words, wrong handler | Deterministic exact-match |
| Context retention | 4 | Turn 4 forgets turn 1 | Deterministic exact-match |
| Fallback and escalation | 4 | Confident guessing instead of handoff | Deterministic exact-match |
| Edge inputs | 6 | Typos, slang, empty, overlong, non-English | Deterministic exact-match |
| Safety and injection | 4 | Jailbreaks, prompt leaks, cross-tenant data | Deterministic exact-match |

All 23 are deterministic. Reach for semantic similarity or an LLM judge only when
the property under test genuinely is prose quality, because both cost you either a
threshold to tune or a model to pay for.

[`data/test_cases.json`](data/test_cases.json) holds the same 23 cases as plain
data: `id`, `category`, `input`, `invariant`, `assertion_style`. Its inputs match
the parametrize tables in the test files exactly, so a support lead or PM can add
a case there and hand it to an engineer without touching pytest.

## Adding a case

Most files use `@pytest.mark.parametrize` over a module-level table, so a new case
is one tuple:

```python
INTENT_CASES = [
    ("Cancel my order", "intent", "CANCEL_ORDER"),
    ("where is my stuff", "intent", "TRACK_ORDER"),   # <- your new case
]
```

`tests/test_context_retention.py` uses four explicit functions instead, because
its turn sequences have different lengths and different invariants. Ragged tuples
would be worse than plain functions there.

## Pointing this at a real chatbot

`FakeChatbotClient` exists so the suite runs green on a fresh clone with zero
setup. It is not a mock; it is a small rule-based class that implements enough
routing to make every case resolvable offline.

The tests only ever touch its public surface: `send(message) -> ChatbotResponse`.
So swap in a class with the same signature and every test runs unmodified against
your real bot:

```python
# chatbot/real_client.py
import os
import requests

from chatbot.client import ChatbotResponse


class ApiChatbotClient:
    """Same interface as FakeChatbotClient, backed by a real deployment.

    One instance is one conversation, exactly like the fake, so the multi-turn
    tests in test_context_retention.py keep working without changes.
    """

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url or os.environ["CHATBOT_BASE_URL"]
        self.api_key = api_key or os.environ["CHATBOT_API_KEY"]
        self.conversation_id = None

    def send(self, message: str) -> ChatbotResponse:
        payload = {"message": message, "conversation_id": self.conversation_id}
        result = requests.post(
            f"{self.base_url}/chat",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        ).json()
        self.conversation_id = result["conversation_id"]

        # Map YOUR bot's structured output onto the same fields the tests read.
        return ChatbotResponse(
            text=result["reply"],
            intent=result.get("intent"),
            intents=set(result.get("intents", [])),
            entities=result.get("entities", {}),
            response_type=result.get("response_type", "normal"),
            order_id=result.get("slots", {}).get("order_id"),
            budget_filter=result.get("slots", {}).get("budget_filter"),
            referent=result.get("slots", {}).get("referent", {}),
            escalation_ticket_created=result.get("escalated", False),
            escalation_triggered_by_cap=result.get("escalated_by_cap", False),
            refund_authorized=result.get("refund_authorized", False),
            system_prompt_leaked=False,
            guardrails_active=result.get("guardrails_active", True),
            other_user_data_returned=False,
            # The important one. Read this from your ACTUAL application, not from
            # the bot's own claim about what it did.
            backend_state=fetch_backend_state(result["customer_id"]),
        )
```

Then point the tests at it, for example with a fixture in `conftest.py` that
returns `ApiChatbotClient` when `CHATBOT_BASE_URL` is set and `FakeChatbotClient`
otherwise.

Two things to get right when you do this:

**Never hardcode credentials.** Read them from the environment. There is no `.env`
in this repo and there should not be one in yours.

**`backend_state` must come from your application, not from the bot.** This is the
whole lesson of
[`test_subscription_cancellation_actually_changes_backend_state`](tests/test_fallback_and_escalation.py).
The bot will tell you it cancelled the subscription in a fluent, confident
sentence whether or not the backend call succeeded. The only assertion that
survives contact with reality is one that queries the real system and asks whether
the row actually changed. If `backend_state` is populated from the bot's own reply,
you have written a test that can never fail.

That gap, between what a chatbot says it did and what your application actually
did, is exactly where a behavioral check on the running app earns its keep.
[Autonoma](https://getautonoma.com) drives the real application and reads the
resulting state back, which is the class of verification a text assertion cannot
give you.

## About

This repository is maintained by [Autonoma](https://getautonoma.com) as reference
material for the linked blog post. Autonoma builds autonomous AI agents that plan,
execute, and maintain end-to-end tests directly from your codebase.

If something here is wrong, out of date, or unclear, please
[open an issue](https://github.com/Autonoma-Tools/chatbot-test-cases/issues/new).

## License

Released under the [MIT License](./LICENSE) © 2026 Autonoma Labs.
