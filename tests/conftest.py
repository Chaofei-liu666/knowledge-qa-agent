import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import Agent  # noqa: E402
from src.index import build_index  # noqa: E402
from src.llm import MockPolicy  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.tools import DocsTools  # noqa: E402


@pytest.fixture(scope="session")
def index():
    return build_index()


@pytest.fixture()
def agent_factory(index):
    def _make(fault: str | None = None, policy=None):
        idx, vec = index
        tools = DocsTools(Retriever(idx, vec), fault_spec=fault)
        return Agent(tools=tools, policy=policy or MockPolicy())
    return _make
