"""Makes `from chatbot.client import FakeChatbotClient` work from any cwd.

pytest puts the *tests/* directory on sys.path, not the repository root, so
without this the local `chatbot` package would not be importable. Adding the repo
root explicitly means `pytest`, `pytest tests/`, and `pytest tests/test_x.py`
all behave identically.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
