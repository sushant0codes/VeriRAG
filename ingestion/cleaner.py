"""Conservative cleanup for common PDF extraction artifacts."""

import re

_SOFT_HYPHEN = "\u00ad"
_FORM_10K_RUNNING_FOOTER = re.compile(
    r"(?m)^[ \t]*[^\n|]{1,100}[ \t]*\|[ \t]*\d{4}[ \t]+Form[ \t]+10-K"
    r"[ \t]*\|[ \t]*\d+[ \t]*(?:\n|$)"
)


def clean_text(text: str) -> str:
    """Normalize extracted text and remove an unambiguous 10-K running footer."""
    text = text.replace(_SOFT_HYPHEN, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    # This is a page-decoration pattern, not report content: ``Issuer | YYYY
    # Form 10-K | page``. Removing it avoids making the footer a retrieval hit.
    text = _FORM_10K_RUNNING_FOOTER.sub("", text)
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
