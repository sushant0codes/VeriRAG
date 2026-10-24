import json

import pytest

from generation import (
    INSUFFICIENT_INFORMATION,
    OLLAMA_DEFAULT_MODEL,
    OllamaLLM,
    _financial_table_summary,
    generate_answer,
    load_llm_from_env,
)


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.instructions = ""
        self.input_text = ""

    def generate(self, instructions: str, input_text: str) -> str:
        self.instructions = instructions
        self.input_text = input_text
        return self.answer


def chunks():
    return [
        {
            "chunk_id": "report_c0001",
            "document_id": "report",
            "text": "Apple reported net sales of $100.",
            "pages": [4, 5],
            "metadata": {"source": "report.pdf", "page_start": 4, "page_end": 5},
        }
    ]


def test_generate_answer_uses_only_context_and_appends_citations() -> None:
    llm = FakeLLM("Apple reported net sales of $100.")

    answer = generate_answer("What were net sales?", chunks(), llm)

    assert answer == (
        "Apple reported net sales of $100.\n\n"
        "Sources:\n- [report.pdf, pp. 4, 5]"
    )
    assert "Answer only from the provided context" in llm.instructions
    assert INSUFFICIENT_INFORMATION in llm.instructions
    assert "Question: What were net sales?" in llm.input_text
    assert "[report.pdf, pp. 4, 5]" in llm.input_text
    assert "Apple reported net sales of $100." in llm.input_text


def test_generate_answer_uses_metadata_page_fallback() -> None:
    llm = FakeLLM("Supported answer.")
    context = [
        {
            "document_id": "annual_report",
            "text": "Supported fact.",
            "metadata": {"source": "annual.pdf", "page": 9},
        }
    ]

    answer = generate_answer("Question", context, llm)

    assert answer.endswith("Sources:\n- [annual.pdf, p. 9]")


def test_generate_answer_instructs_models_how_to_read_financial_tables() -> None:
    llm = FakeLLM("Apple's 2025 net income was $112,010 million.")

    generate_answer("What was net income?", chunks(), llm)

    assert "multiple years in sequence" in llm.instructions
    assert "matching each row label to its column headers and dates" in llm.instructions
    assert "If the answer is explicitly present in the context, answer directly" in llm.instructions
    assert "Do not guess" in llm.instructions


def test_financial_table_summary_matches_three_year_values() -> None:
    text = (
        "CONSOLIDATED STATEMENTS OF INCOME (In millions) Years ended "
        "September 27, 2025 September 28, 2024 September 30, 2023 "
        "Net income $ 112,010 $ 93,736 $ 96,995 See accompanying Notes"
    )

    assert _financial_table_summary(text) == (
        "Net income: September 27, 2025 = $112,010; "
        "September 28, 2024 = $93,736; September 30, 2023 = $96,995"
    )


def test_financial_table_summary_returns_none_without_year_headers() -> None:
    assert _financial_table_summary("Net income $ 112,010 $ 93,736 $ 96,995") is None


def test_financial_table_summary_ignores_malformed_rows() -> None:
    text = (
        "Years ended September 27, 2025 September 28, 2024 September 30, 2023 "
        "Net income $ 112,010 $ 93,736 See accompanying Notes"
    )

    assert _financial_table_summary(text) is None


def test_generate_answer_returns_insufficient_information_for_empty_context() -> None:
    llm = FakeLLM("This must not be used.")

    assert generate_answer("Question", [], llm) == INSUFFICIENT_INFORMATION
    assert llm.input_text == ""


def test_ollama_generation_posts_context_to_local_api(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"response": "Apple reported net sales of $100."}'

    def fake_urlopen(request):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr("generation.urlopen", fake_urlopen)

    answer = OllamaLLM("llama3.2:3b").generate("Use context only.", "Context text")

    assert answer == "Apple reported net sales of $100."
    assert captured == {
        "url": "http://localhost:11434/api/generate",
        "payload": {
            "model": "llama3.2:3b",
            "system": "Use context only.",
            "prompt": "Context text",
            "stream": False,
        },
    }


def test_load_llm_requires_a_configured_supported_provider(monkeypatch) -> None:
    monkeypatch.setenv("VERIRAG_LLM_PROVIDER", "other")
    with pytest.raises(ValueError, match="unsupported"):
        load_llm_from_env()

    monkeypatch.setenv("VERIRAG_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_llm_from_env()


def test_load_llm_selects_ollama_without_an_api_key(monkeypatch) -> None:
    monkeypatch.setenv("VERIRAG_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VERIRAG_LLM_MODEL", raising=False)

    llm = load_llm_from_env()

    assert isinstance(llm, OllamaLLM)
    assert llm.model == OLLAMA_DEFAULT_MODEL
