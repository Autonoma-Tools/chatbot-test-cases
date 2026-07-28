"""Chatbot client package.

Exposes FakeChatbotClient, a deterministic offline stand-in used by every test in
this repository. See chatbot/client.py for how to swap it for a real API-backed
client without changing any test.
"""
