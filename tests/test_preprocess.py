"""Tests for text preprocessing and span splitting."""

from __future__ import annotations

import unittest

from pinyinize.preprocess import split_spans
from pinyinize.types import Span
from pinyinize.util import (
    is_ascii_digit,
    is_ascii_letter,
    is_han,
    is_punct_or_symbol,
    is_space,
    is_word_like_protected_kind,
    normalize_word_pinyin,
)


class TestIsHan(unittest.TestCase):
    """Tests for Han character detection."""

    def test_common_chinese_chars(self) -> None:
        """Test common Chinese characters are detected as Han."""
        self.assertTrue(is_han("中"))
        self.assertTrue(is_han("国"))
        self.assertTrue(is_han("银"))
        self.assertTrue(is_han("行"))

    def test_ascii_not_han(self) -> None:
        """Test ASCII characters are not Han."""
        self.assertFalse(is_han("a"))
        self.assertFalse(is_han("A"))
        self.assertFalse(is_han("0"))
        self.assertFalse(is_han(" "))
        self.assertFalse(is_han("!"))

    def test_cjk_extension_chars(self) -> None:
        """Test CJK extension characters are detected as Han."""
        # These are in CJK Unified Ideographs Extension A range (U+3400-U+4DBF)
        self.assertTrue(is_han("㐀"))  # U+3400
        self.assertTrue(is_han("㐁"))  # U+3401

    def test_empty_string(self) -> None:
        """Test empty string is not Han."""
        self.assertFalse(is_han(""))

    def test_punctuation_not_han(self) -> None:
        """Test punctuation is not Han."""
        self.assertFalse(is_han("。"))
        self.assertFalse(is_han("，"))
        self.assertFalse(is_han("！"))


class TestIsAsciiLetter(unittest.TestCase):
    """Tests for ASCII letter detection."""

    def test_lowercase(self) -> None:
        """Test lowercase letters."""
        for c in "abcdefghijklmnopqrstuvwxyz":
            self.assertTrue(is_ascii_letter(c), f"'{c}' should be ASCII letter")

    def test_uppercase(self) -> None:
        """Test uppercase letters."""
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertTrue(is_ascii_letter(c), f"'{c}' should be ASCII letter")

    def test_non_letters(self) -> None:
        """Test non-letters are not ASCII letters."""
        self.assertFalse(is_ascii_letter("0"))
        self.assertFalse(is_ascii_letter(" "))
        self.assertFalse(is_ascii_letter("中"))
        self.assertFalse(is_ascii_letter("!"))


class TestIsAsciiDigit(unittest.TestCase):
    """Tests for ASCII digit detection."""

    def test_digits(self) -> None:
        """Test all ASCII digits."""
        for c in "0123456789":
            self.assertTrue(is_ascii_digit(c), f"'{c}' should be ASCII digit")

    def test_non_digits(self) -> None:
        """Test non-digits are not ASCII digits."""
        self.assertFalse(is_ascii_digit("a"))
        self.assertFalse(is_ascii_digit(" "))
        self.assertFalse(is_ascii_digit("中"))
        self.assertFalse(is_ascii_digit("!"))


class TestIsSpace(unittest.TestCase):
    """Tests for space detection."""

    def test_space(self) -> None:
        """Test regular space."""
        self.assertTrue(is_space(" "))

    def test_tab(self) -> None:
        """Test tab character."""
        self.assertTrue(is_space("\t"))

    def test_newline(self) -> None:
        """Test newline characters."""
        self.assertTrue(is_space("\n"))
        self.assertTrue(is_space("\r"))

    def test_non_space(self) -> None:
        """Test non-space characters."""
        self.assertFalse(is_space("a"))
        self.assertFalse(is_space("中"))
        self.assertFalse(is_space("0"))


class TestIsPunctOrSymbol(unittest.TestCase):
    """Tests for punctuation and symbol detection."""

    def test_common_punctuation(self) -> None:
        """Test common punctuation marks."""
        self.assertTrue(is_punct_or_symbol("."))
        self.assertTrue(is_punct_or_symbol(","))
        self.assertTrue(is_punct_or_symbol("!"))
        self.assertTrue(is_punct_or_symbol("?"))
        self.assertTrue(is_punct_or_symbol(":"))
        self.assertTrue(is_punct_or_symbol(";"))

    def test_chinese_punctuation(self) -> None:
        """Test Chinese punctuation marks."""
        self.assertTrue(is_punct_or_symbol("。"))
        self.assertTrue(is_punct_or_symbol("，"))
        self.assertTrue(is_punct_or_symbol("！"))
        self.assertTrue(is_punct_or_symbol("？"))
        self.assertTrue(is_punct_or_symbol("："))
        self.assertTrue(is_punct_or_symbol("；"))

    def test_symbols(self) -> None:
        """Test symbols."""
        self.assertTrue(is_punct_or_symbol("@"))
        self.assertTrue(is_punct_or_symbol("#"))
        self.assertTrue(is_punct_or_symbol("$"))
        self.assertTrue(is_punct_or_symbol("%"))
        self.assertTrue(is_punct_or_symbol("&"))
        self.assertTrue(is_punct_or_symbol("*"))

    def test_non_punctuation(self) -> None:
        """Test non-punctuation characters."""
        self.assertFalse(is_punct_or_symbol("a"))
        self.assertFalse(is_punct_or_symbol("中"))
        self.assertFalse(is_punct_or_symbol("0"))


class TestSplitSpans(unittest.TestCase):
    """Tests for span splitting."""

    def test_simple_chinese(self) -> None:
        """Test simple Chinese text."""
        text = "中国银行"
        spans = split_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].type, "han")
        self.assertEqual(spans[0].text, "中国银行")
        self.assertEqual(spans[0].start, 0)
        self.assertEqual(spans[0].end, 4)

    def test_chinese_with_punctuation(self) -> None:
        """Test Chinese with punctuation."""
        text = "你好，世界。"
        spans = split_spans(text)

        # Implementation groups consecutive punctuation together
        self.assertEqual(len(spans), 4)
        self.assertEqual(spans[0].text, "你好")
        self.assertEqual(spans[1].text, "，")
        self.assertEqual(spans[2].text, "世界")
        self.assertEqual(spans[3].text, "。")

    def test_mixed_latin_and_chinese(self) -> None:
        """Test mixed Latin and Chinese text."""
        text = "hello世界"
        spans = split_spans(text)

        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].text, "hello")
        self.assertEqual(spans[0].type, "protected")
        self.assertEqual(spans[0].kind, "latin")
        self.assertEqual(spans[1].text, "世界")
        self.assertEqual(spans[1].type, "han")

    def test_numbers(self) -> None:
        """Test number detection."""
        text = "123"
        spans = split_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].type, "protected")
        self.assertEqual(spans[0].kind, "number")
        self.assertEqual(spans[0].text, "123")

    def test_decimal_numbers(self) -> None:
        """Test decimal number detection."""
        text = "3.14159"
        spans = split_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].kind, "number")
        self.assertEqual(spans[0].text, "3.14159")

    def test_percentage(self) -> None:
        """Test percentage detection."""
        text = "50%"
        spans = split_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].kind, "number")
        self.assertEqual(spans[0].text, "50%")

    def test_url_detection(self) -> None:
        """Test URL detection."""
        text = "https://example.com"
        spans = split_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].type, "protected")
        self.assertEqual(spans[0].kind, "url")
        self.assertEqual(spans[0].text, "https://example.com")

    def test_http_url(self) -> None:
        """Test HTTP URL detection."""
        text = "http://example.com/path?query=1"
        spans = split_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].kind, "url")
        self.assertEqual(spans[0].text, "http://example.com/path?query=1")

    def test_url_with_surrounding_text(self) -> None:
        """Test URL with surrounding text."""
        text = "访问https://example.com即可"
        spans = split_spans(text)

        # Implementation splits at the boundary between han and URL
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].type, "han")
        self.assertEqual(spans[1].type, "protected")
        self.assertEqual(spans[1].kind, "url")

    def test_latin_with_underscore(self) -> None:
        """Test Latin word with underscore."""
        text = "hello_world"
        spans = split_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].kind, "latin")
        self.assertEqual(spans[0].text, "hello_world")

    def test_latin_with_hyphen(self) -> None:
        """Test Latin word with hyphen."""
        text = "well-known"
        spans = split_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].kind, "latin")
        self.assertEqual(spans[0].text, "well-known")

    def test_space_handling(self) -> None:
        """Test space handling."""
        text = "hello  world"
        spans = split_spans(text)

        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0].kind, "latin")
        self.assertEqual(spans[1].kind, "space")
        self.assertEqual(spans[2].kind, "latin")

    def test_empty_string(self) -> None:
        """Test empty string."""
        text = ""
        spans = split_spans(text)

        self.assertEqual(len(spans), 0)

    def test_only_punctuation(self) -> None:
        """Test only punctuation."""
        text = "!!!"
        spans = split_spans(text)

        self.assertEqual(len(spans), 3)
        for span in spans:
            self.assertEqual(span.type, "protected")
            self.assertEqual(span.kind, "punct")

    def test_mixed_complex(self) -> None:
        """Test complex mixed content."""
        text = "OpenAI的API v2.0：https://openai.com"
        spans = split_spans(text)

        # Should have: OpenAI, 的, API,  , v2, ., 0, ：, https://openai.com
        # Note: v2.0 is split into v2, ., 0 since decimals are handled specially
        self.assertGreater(len(spans), 0)

        # Verify specific spans
        texts = [s.text for s in spans]
        self.assertIn("OpenAI", texts)
        self.assertIn("v2", texts)
        self.assertIn("0", texts)
        self.assertIn("https://openai.com", texts)

    def test_span_ids_sequential(self) -> None:
        """Test that span IDs are sequential."""
        text = "a中b"
        spans = split_spans(text)

        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0].span_id, "S0")
        self.assertEqual(spans[1].span_id, "S1")
        self.assertEqual(spans[2].span_id, "S2")

    def test_span_offsets_continuous(self) -> None:
        """Test that span offsets are continuous."""
        text = "hello世界123"
        spans = split_spans(text)

        current_end = 0
        for span in spans:
            self.assertEqual(span.start, current_end)
            current_end = span.end


class TestNormalizeWordPinyin(unittest.TestCase):
    """Tests for word pinyin normalization."""

    def test_remove_spaces(self) -> None:
        """Test removing syllable separator spaces."""
        self.assertEqual(normalize_word_pinyin("yín háng"), "yínháng")
        self.assertEqual(normalize_word_pinyin("zhōng guó"), "zhōngguó")

    def test_no_spaces(self) -> None:
        """Test text without spaces."""
        self.assertEqual(normalize_word_pinyin("zhōngguó"), "zhōngguó")

    def test_normalize_v_to_u_umlaut(self) -> None:
        """Test normalizing 'v' to 'ü'."""
        self.assertEqual(normalize_word_pinyin("nv"), "nü")
        self.assertEqual(normalize_word_pinyin("lv"), "lü")
        self.assertEqual(normalize_word_pinyin("NV"), "NÜ")

    def test_complex_normalization(self) -> None:
        """Test complex normalization - removes spaces and converts v to ü."""
        # Note: normalize_word_pinyin doesn't add tone marks, it only removes spaces and converts v to ü
        self.assertEqual(normalize_word_pinyin("nü han"), "nühan")
        self.assertEqual(normalize_word_pinyin("nv hai"), "nühai")


class TestIsWordLikeProtectedKind(unittest.TestCase):
    """Tests for word-like protected kind detection."""

    def test_url_is_word_like(self) -> None:
        """Test URL is word-like."""
        self.assertTrue(is_word_like_protected_kind("url"))

    def test_latin_is_word_like(self) -> None:
        """Test Latin is word-like."""
        self.assertTrue(is_word_like_protected_kind("latin"))

    def test_number_is_word_like(self) -> None:
        """Test Number is word-like."""
        self.assertTrue(is_word_like_protected_kind("number"))

    def test_punct_not_word_like(self) -> None:
        """Test punctuation is not word-like."""
        self.assertFalse(is_word_like_protected_kind("punct"))

    def test_space_not_word_like(self) -> None:
        """Test space is not word-like."""
        self.assertFalse(is_word_like_protected_kind("space"))

    def test_other_not_word_like(self) -> None:
        """Test other is not word-like."""
        self.assertFalse(is_word_like_protected_kind("other"))

    def test_none_not_word_like(self) -> None:
        """Test None is not word-like."""
        self.assertFalse(is_word_like_protected_kind(None))


class TestSpanDataclass(unittest.TestCase):
    """Tests for Span dataclass."""

    def test_span_creation(self) -> None:
        """Test Span creation."""
        span = Span(
            span_id="S0",
            type="han",
            start=0,
            end=2,
            text="中文",
        )
        self.assertEqual(span.span_id, "S0")
        self.assertEqual(span.type, "han")
        self.assertEqual(span.start, 0)
        self.assertEqual(span.end, 2)
        self.assertEqual(span.text, "中文")
        self.assertIsNone(span.kind)

    def test_protected_span_creation(self) -> None:
        """Test protected Span creation."""
        span = Span(
            span_id="S1",
            type="protected",
            start=2,
            end=7,
            text="hello",
            kind="latin",
        )
        self.assertEqual(span.type, "protected")
        self.assertEqual(span.kind, "latin")

    def test_span_immutability(self) -> None:
        """Test Span is frozen/immutable."""
        span = Span(
            span_id="S0",
            type="han",
            start=0,
            end=2,
            text="中文",
        )
        with self.assertRaises(AttributeError):
            span.text = "英文"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
