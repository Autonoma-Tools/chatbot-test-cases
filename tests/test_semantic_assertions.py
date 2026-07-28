"""Three assertion styles, side by side, against the same chatbot reply.

The running example is fixed across all three tests so the tradeoff is visible
rather than abstract:

    input:                'cancel my order'
    naive expected string: 'Your order has been cancelled.'
    actual reply:          "Sure, I've cancelled your order for you."

The reply is a correct paraphrase of the expectation. Every assertion style below
is judged on one question: does it pass?

    1. Deterministic exact-match  -> passes on the routed field, fails on the text
    2. Semantic similarity        -> passes on the text, with a threshold
    3. LLM-as-judge               -> passes on a rubric, needs a model

This file stays runnable with `pip install pytest` alone. Test 3 skips cleanly
when deepeval is not installed or no model credentials are configured, so a fresh
clone with zero secrets is green.
"""

import difflib
import os

import pytest

from chatbot.client import FakeChatbotClient

# The article's diagram quotes these three strings. Keep them in sync with it.
INPUT_MESSAGE = "cancel my order"
NAIVE_EXPECTED_TEXT = "Your order has been cancelled."
PARAPHRASED_REPLY = "Sure, I've cancelled your order for you."


def assert_naive_exact_match(reply_text: str, expected: str) -> bool:
    """The assertion almost everybody writes first: plain string equality.

    Returned as a bool rather than raising, so the tests below can DEMONSTRATE its
    failure mode without themselves going red.
    """
    return reply_text == expected


def semantic_similarity(a: str, b: str) -> float:
    """A dependency-free stand-in for semantic similarity.

    This uses difflib.SequenceMatcher, which measures character-sequence overlap,
    not meaning. A production system would swap this one function for a real
    embedding-based cosine similarity (OpenAI or Cohere embeddings, or a local
    sentence-transformers model) and would score paraphrases far higher than a
    character-overlap metric can.

    It is deliberately NOT that here: an embeddings call needs a network round
    trip and an API key, and the premise of this repo is that the suite runs on a
    fresh clone with neither. The shape of the assertion (a float compared against
    a threshold) is identical either way, which is the part worth learning.
    """
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


#: Measured, not guessed. With SequenceMatcher, the two example strings above
#: score 0.371. A real embedding model would put the same pair around 0.90, so a
#: production threshold is typically 0.80-0.85. Recompute this constant if you
#: swap the metric; do not carry 0.35 over to an embeddings implementation.
SIMILARITY_THRESHOLD = 0.35


def test_deterministic_exact_match_on_intent():
    """input     -> 'cancel my order'
    invariant -> intent == 'CANCEL_ORDER'
    assertion -> exact-match on the deterministic field, not the reply text

    Exact-match is the right tool here precisely BECAUSE the target is a routed
    enum rather than prose. There is no paraphrase of 'CANCEL_ORDER'.
    """
    client = FakeChatbotClient()

    response = client.send(INPUT_MESSAGE)

    assert response.intent == "CANCEL_ORDER"


def test_naive_exact_match_on_reply_text_is_the_failure_mode():
    """The same style, aimed at the wrong target.

    input     -> 'cancel my order'
    invariant -> the reply conveys that the order was cancelled
    assertion -> string equality against 'Your order has been cancelled.'

    The reply is CORRECT. The bot cancelled the order and said so. Exact-match
    against a hand-written expected sentence still reports a failure, because the
    bot chose different words. That is the assertion being wrong, not the bot.

    This test asserts the boolean result is False rather than using a raising
    assert, so the demonstration documents the problem without turning the suite
    red.
    """
    client = FakeChatbotClient()

    response = client.send(INPUT_MESSAGE)

    matched = assert_naive_exact_match(response.text, NAIVE_EXPECTED_TEXT)

    assert matched is False, (
        "This test exists to document that exact-match on reply text rejects a "
        "correct paraphrase. If it started passing, the fake client's wording was "
        "changed to coincidentally equal the naive expectation."
    )


def test_semantic_similarity_assertion():
    """input     -> 'cancel my order'
    invariant -> the two strings convey the same fact
    assertion -> semantic similarity above a threshold, not exact equality

    Same pair of strings that exact-match rejected. Scored on a spectrum instead
    of a binary, the paraphrase clears the bar. The cost of that flexibility is
    the threshold itself: it is a tuning parameter with no correct value, and a
    threshold low enough to accept every valid paraphrase is usually also low
    enough to accept some wrong answers.
    """
    score = semantic_similarity(PARAPHRASED_REPLY, NAIVE_EXPECTED_TEXT)

    assert score > SIMILARITY_THRESHOLD, (
        f"similarity {score:.3f} did not clear threshold {SIMILARITY_THRESHOLD}"
    )

    # And the same reply the bot actually produced clears it too.
    response = FakeChatbotClient().send(INPUT_MESSAGE)
    assert semantic_similarity(response.text, NAIVE_EXPECTED_TEXT) > SIMILARITY_THRESHOLD


def _import_deepeval():
    """Gate on deepeval being importable.

    pytest.importorskip raises pytest's own Skipped exception on ImportError,
    which is exactly the behaviour we want. It is factored out so the caller can
    also survive a deepeval that is installed but raises something other than
    ImportError while importing (a misconfigured install, a version conflict in a
    transitive dependency). Neither is a defect in the bot under test.
    """
    metrics = pytest.importorskip(
        "deepeval.metrics",
        reason="LLM-as-judge requires deepeval; skipping in offline mode",
    )
    test_case = pytest.importorskip("deepeval.test_case")
    return metrics, test_case


def test_llm_as_judge_assertion():
    """input     -> 'cancel my order'
    invariant -> the response is faithful to the fact, per a rubric
    assertion -> a second model scores the response against the rubric

    Use this when correctness genuinely is a rubric rather than a string: tone,
    faithfulness to a source document, refusal quality. The cost is that your test
    suite now depends on a model, which means latency, spend, and a judge that can
    itself be wrong.

    This test SKIPS when deepeval is absent or no model credentials are
    configured, which is the default state of a fresh clone. It becomes a real
    gate once a team installs deepeval and wires an API key into CI.
    """
    try:
        deepeval_metrics, deepeval_test_case = _import_deepeval()
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # pragma: no cover - broken deepeval install
        pytest.skip(f"deepeval is installed but unusable: {type(exc).__name__}: {exc}")

    if not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("DEEPEVAL_MODEL")
    ):
        pytest.skip(
            "LLM-as-judge requires model access; skipping in CI-without-keys mode"
        )

    response = FakeChatbotClient().send(INPUT_MESSAGE)

    try:
        metric = deepeval_metrics.GEval(
            name="CancellationConfirmed",
            criteria=(
                "Does the response confirm the order was cancelled, regardless of "
                "exact phrasing?"
            ),
            evaluation_params=[
                deepeval_test_case.LLMTestCaseParams.INPUT,
                deepeval_test_case.LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.7,
        )
        case = deepeval_test_case.LLMTestCase(
            input=INPUT_MESSAGE,
            actual_output=response.text,
        )
        metric.measure(case)
    except Exception as exc:  # pragma: no cover - depends on external model access
        # Missing or misconfigured credentials, rate limits, and offline runners all
        # land here. None of them is a defect in the bot under test, so this must
        # skip rather than fail: a red build for "the judge was unreachable" trains
        # people to ignore red builds.
        pytest.skip(f"LLM-as-judge unavailable: {type(exc).__name__}: {exc}")

    if metric.score is None:
        # A judge that returned no score has told us nothing. Skipping is honest;
        # asserting on None would raise a TypeError and read as a bot defect.
        pytest.skip("LLM-as-judge returned no score")

    # Documented expected outcome: pass. The reply confirms the cancellation.
    assert metric.score >= 0.7, f"judge scored {metric.score}: {metric.reason}"


# --------------------------------------------------------------------------- #
# Assertion-style spine, reproduced from the article so this file is
# self-documenting when read in isolation.
#
#   Category                        Assertion style
#   ------------------------------  ---------------------------------
#   Intent recognition              Deterministic exact-match
#   Context retention               Deterministic exact-match
#   Fallback and escalation         Deterministic exact-match
#   Edge inputs                     Deterministic exact-match
#   Safety and injection            Deterministic exact-match
#   Open-ended response quality     Semantic similarity or LLM-as-judge
#
# All 23 cases in data/test_cases.json are deterministic_exact_match. Reach for
# semantic or judge-based assertions only when the property under test really is
# prose quality, because both of them cost you either a threshold to tune or a
# model to pay for.
# --------------------------------------------------------------------------- #
