from proxy.text import canonical_text, compact_description, dedupe_blocks, normalize_text


def test_normalize_text_collapses_blank_lines_and_crlf():
    assert normalize_text("  a  \r\n\r\n\r\n\r\n  b  ") == "a\n\n  b"


def test_canonical_text_ignores_indentation():
    assert canonical_text("a\n    b") == canonical_text("   a\nb")


def test_compact_description_keeps_whole_sentences_within_the_limit():
    text = "First sentence. Second sentence. Third sentence."

    assert compact_description(text, 100) == text
    assert compact_description(text, 35) == "First sentence. Second sentence."


def test_compact_description_truncates_a_single_long_sentence():
    result = compact_description("word " * 50, 20)

    assert len(result) <= 20
    assert result.endswith("...")


def test_dedupe_blocks_never_empties_a_message():
    assert dedupe_blocks("same\n\nsame") == "same"
    assert dedupe_blocks("   ") == ""
    assert dedupe_blocks("only") == "only"
