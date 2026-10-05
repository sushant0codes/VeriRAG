from ingestion.cleaner import clean_text


def test_cleaner_repairs_line_hyphenation_and_whitespace() -> None:
    assert clean_text("inter-\n national\t report\u00ad\n\n\n2026") == "international report\n\n2026"


def test_cleaner_removes_form_10k_running_footer() -> None:
    text = "Evidence before.\nApple Inc. | 2025 Form 10-K | 17\nEvidence after."

    assert clean_text(text) == "Evidence before.\nEvidence after."
