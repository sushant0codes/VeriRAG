from verification import _parse_verdict, verify_claims


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.instructions = []
        self.inputs = []

    def generate(self, instructions, input_text):
        self.instructions.append(instructions)
        self.inputs.append(input_text)
        return next(self.responses)


def chunks():
    return [
        {
            "chunk_id": "sales",
            "document_id": "annual-report",
            "text": "Net sales were $100 million in 2025.",
            "pages": [4, 5],
            "metadata": {"source": "annual.pdf", "page_start": 4, "page_end": 5},
        },
        {
            "chunk_id": "employees",
            "document_id": "annual-report",
            "text": "The company employed 200 people.",
            "pages": [8],
            "metadata": {"source": "annual.pdf", "page": 8},
        },
    ]


def test_supported_claim_uses_actual_source_metadata():
    llm = FakeLLM([
        '{"status":"supported","supporting_chunk_ids":["sales"],"reason":"The amount is stated."}',
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"No contradiction."}',
    ])
    result = verify_claims("Net sales were $100 million in 2025.", chunks(), llm)
    assert result[0]["status"] == "supported"
    assert result[0]["supporting_sources"] == [{"chunk_id": "sales", "document_id": "annual-report", "source": "annual.pdf", "pages": [4, 5], "metadata": {"source": "annual.pdf", "page_start": 4, "page_end": 5}}]
    assert "Net sales were $100 million" in llm.inputs[0]


def test_unsupported_and_ambiguous_claims_are_reported():
    llm = FakeLLM([
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"No direct support."}',
        '{"status":"unsupported","supporting_chunk_ids":["sales"],"reason":"Evidence states $100 million."}',
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"No direct support."}',
        '{"status":"uncertain","supporting_chunk_ids":["employees"],"reason":"The time period is absent."}',
    ])
    result = verify_claims("Net sales were $200 million. The company employed 200 people in 2025.", chunks(), llm)
    assert [item["status"] for item in result] == ["unsupported", "uncertain"]


def test_empty_chunks_are_uncertain_without_calling_llm():
    llm = FakeLLM([])
    result = verify_claims("Revenue increased.", [], llm)
    assert result[0]["status"] == "uncertain"
    assert result[0]["supporting_sources"] == []
    assert llm.inputs == []


def test_malformed_response_and_unsupported_source_cannot_be_supported():
    llm = FakeLLM(["not json", '{"status":"supported","supporting_chunk_ids":["invented"],"reason":"claimed"}'])
    malformed = verify_claims("Revenue increased.", chunks(), llm)
    no_source = verify_claims("Revenue increased.", chunks(), llm)
    assert malformed[0]["status"] == "uncertain"
    assert no_source[0]["status"] == "uncertain"


def test_sources_section_is_not_treated_as_a_claim():
    llm = FakeLLM([
        '{"status":"supported","supporting_chunk_ids":["sales"],"reason":"stated"}',
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"No contradiction."}',
    ])
    result = verify_claims("Net sales were $100 million.\n\nSources:\n- [annual.pdf, p. 4]", chunks(), llm)
    assert len(result) == 1
    assert result[0]["claim"] == "Net sales were $100 million."


def test_parse_verdict_accepts_markdown_fenced_json():
    verdict = _parse_verdict(
        "```json\n"
        '{"status":"supported","supporting_chunk_ids":["sales"],"reason":"stated"}'
        "\n```"
    )

    assert verdict is not None
    assert verdict["status"] == "supported"


def test_parse_verdict_accepts_json_followed_by_explanatory_prose():
    verdict = _parse_verdict(
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"missing evidence"} '
        "The source does not discuss this topic."
    )

    assert verdict is not None
    assert verdict["status"] == "uncertain"


def test_parse_verdict_rejects_malformed_json():
    assert _parse_verdict('{"status":"supported", "supporting_chunk_ids": [}') is None


def test_explicitly_contradicted_claim_keeps_valid_source_attribution():
    llm = FakeLLM([
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"No direct support."}',
        '{"status":"unsupported","supporting_chunk_ids":["sales"],'
        '"reason":"The source states $100 million."}'
    ])

    result = verify_claims("Net sales were $200 million.", chunks(), llm)

    assert result[0]["status"] == "unsupported"
    assert result[0]["supporting_sources"][0]["source"] == "annual.pdf"


def test_uncited_contradiction_verdict_is_not_treated_as_evidence():
    llm = FakeLLM([
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"No direct support."}',
        '{"status":"unsupported","supporting_chunk_ids":[],'
        '"reason":"The claim is not supported."}',
    ])

    result = verify_claims("Revenue was $200 million.", chunks(), llm)

    assert result[0]["status"] == "uncertain"
    assert result[0]["supporting_sources"] == []


def test_claim_absent_from_evidence_is_uncertain():
    llm = FakeLLM([
        '{"status":"uncertain","supporting_chunk_ids":[],'
        '"reason":"The evidence does not mention revenue."}',
        '{"status":"uncertain","supporting_chunk_ids":[],'
        '"reason":"The evidence does not mention revenue."}',
    ])

    result = verify_claims("Revenue was $200 million.", chunks(), llm)

    assert result[0]["status"] == "uncertain"
    assert "lack of mention and related but different" in llm.instructions[1].lower()


def test_unmentioned_product_claim_is_uncertain_not_unsupported():
    product_chunk = {
        "chunk_id": "products",
        "document_id": "company-profile",
        "text": "The company manufactures smartphones.",
        "pages": [3],
        "metadata": {"source": "profile.pdf", "page": 3},
    }
    llm = FakeLLM([
        '{"status":"uncertain","supporting_chunk_ids":[],'
        '"reason":"The evidence does not mention aircraft engines."}',
        '{"status":"uncertain","supporting_chunk_ids":[],'
        '"reason":"The evidence does not mention aircraft engines."}',
    ])

    result = verify_claims(
        "The company manufactures aircraft engines.", [product_chunk], llm
    )

    assert result[0]["status"] == "uncertain"
    assert (
        "related but different entities, products, actions, or attributes"
        in llm.instructions[1].lower()
    )


def test_negated_claim_is_unsupported_when_evidence_states_the_positive_fact():
    product_chunk = {
        "chunk_id": "products",
        "document_id": "company-profile",
        "text": "The company manufactures smartphones.",
        "pages": [3],
        "metadata": {"source": "profile.pdf", "page": 3},
    }
    llm = FakeLLM([
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"No direct support."}',
        '{"status":"unsupported","supporting_chunk_ids":["products"],'
        '"reason":"The evidence directly states the positive fact."}'
    ])

    result = verify_claims(
        "The company does not manufacture smartphones.", [product_chunk], llm
    )

    assert result[0]["status"] == "unsupported"
    assert result[0]["supporting_sources"][0]["pages"] == [3]


def test_directly_supported_smartphone_claim_uses_support_only():
    product_chunk = {
        "chunk_id": "products",
        "document_id": "company-profile",
        "text": "The company manufactures smartphones.",
        "pages": [3],
        "metadata": {"source": "profile.pdf", "page": 3},
    }
    llm = FakeLLM([
        '{"status":"supported","supporting_chunk_ids":["products"],'
        '"reason":"The source directly states this."}',
        '{"status":"uncertain","supporting_chunk_ids":[],"reason":"No contradiction."}',
    ])

    result = verify_claims("The company manufactures smartphones.", [product_chunk], llm)

    assert result[0]["status"] == "supported"
    assert result[0]["supporting_sources"][0]["chunk_id"] == "products"


def test_financial_table_claim_receives_explicit_year_value_relationships():
    financial_chunk = {
        "chunk_id": "income-statement",
        "document_id": "report",
        "text": (
            "The following table shows results (net income in millions): "
            "2025 2024 2023 Numerator: Net income $ 112,010 $ 93,736 $ 96,995 "
            "Denominator: Shares 10 9 8"
        ),
        "pages": [12],
        "metadata": {"source": "report.pdf", "page": 12},
    }
    llm = FakeLLM([])

    result = verify_claims("Net income for 2025 was $112,010 million.", [financial_chunk], llm)

    assert result[0]["status"] == "supported"
    assert result[0]["supporting_sources"][0]["pages"] == [12]
    assert llm.inputs == []


def test_financial_table_different_value_is_unsupported_without_llm():
    financial_chunk = {
        "chunk_id": "income-statement",
        "document_id": "report",
        "text": (
            "The following table shows results (net income in millions): "
            "2025 2024 2023 Numerator: Net income $ 112,010 $ 93,736 $ 96,995 "
            "Denominator: Weighted-average basic shares outstanding 14,948,500 15,343,783 15,744,231"
        ),
        "pages": [12],
        "metadata": {"source": "report.pdf", "page": 12},
    }
    llm = FakeLLM([])

    result = verify_claims("Net income for 2025 was $500,000 million.", [financial_chunk], llm)

    assert result[0]["status"] == "unsupported"
    assert "$112,010 million" in result[0]["reason"]
    assert llm.inputs == []


def test_future_year_without_a_forecast_is_uncertain():
    financial_chunk = {
        "chunk_id": "income-statement",
        "document_id": "report",
        "text": (
            "Financial results (net income in millions): 2025 2024 2023 "
            "Net income $ 112,010 $ 93,736 $ 96,995"
        ),
        "pages": [12],
        "metadata": {"source": "report.pdf", "page": 12},
    }
    llm = FakeLLM([])

    result = verify_claims("Net income in 2030 will be $200,000 million.", [financial_chunk], llm)

    assert result[0]["status"] == "uncertain"
    assert result[0]["supporting_sources"][0]["pages"] == [12]
    assert llm.inputs == []
