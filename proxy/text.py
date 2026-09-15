"""Deterministic text normalization shared by the request optimizations."""

import re

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    """Remove excess blank lines, trailing whitespace and CRLF endings."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def canonical_text(text: str) -> str:
    """Indentation-insensitive form used to compare two pieces of text."""
    return "\n".join(line.strip() for line in normalize_text(text).split("\n"))


def compact_description(value: str, limit: int) -> str:
    """Trim a description to the whole sentences that fit the limit."""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    kept = ""
    for sentence in _SENTENCE_END.split(text):
        candidate = f"{kept} {sentence}" if kept else sentence
        if kept and len(candidate) > limit:
            break
        kept = candidate
    if len(kept) <= limit:
        return kept
    return f"{kept[:max(limit - 3, 0)].rstrip()}..."


def dedupe_blocks(content: str) -> str:
    """Drop blocks repeated inside a single message; never empty the message."""
    normalized = normalize_text(content)
    seen: set[str] = set()
    unique: list[str] = []
    for block in normalized.split("\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        key = canonical_text(stripped)
        if key in seen:
            continue
        seen.add(key)
        unique.append(stripped)
    return "\n\n".join(unique) or normalized
