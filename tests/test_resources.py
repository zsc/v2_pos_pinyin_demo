"""Tests for resource loading and management."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pinyinize.resources import (
    PinyinResources,
    _load_char_base,
    _load_lexicon,
    _load_overrides_rules,
    _load_polyphone_candidates,
    _load_polyphone_disambig,
    _load_word_pinyin_map,
)


class TestLoadWordPinyinMap(unittest.TestCase):
    """Tests for _load_word_pinyin_map function."""

    def test_load_valid_words(self) -> None:
        """Test loading valid word-pinyin mappings."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "word.json"
            # Write as JSONL/streaming format (one object per line)
            path.write_text(
                '{"word": "银行", "pinyin": "yín háng"},\n'
                '{"word": "中国", "pinyin": "zhōng guó"},\n',
                encoding="utf-8",
            )
            result = _load_word_pinyin_map(path)
            self.assertEqual(result["银行"], "yín háng")
            self.assertEqual(result["中国"], "zhōng guó")

    def test_skip_non_han_words(self) -> None:
        """Test that words with non-Han characters are skipped."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "word.json"
            path.write_text(
                '{"word": "银行", "pinyin": "yín háng"},\n'
                '{"word": "OpenAI", "pinyin": "OpenAI"},\n'
                '{"word": "API接口", "pinyin": "API jiē kǒu"},\n',
                encoding="utf-8",
            )
            result = _load_word_pinyin_map(path)
            self.assertIn("银行", result)
            self.assertNotIn("OpenAI", result)
            self.assertNotIn("API接口", result)

    def test_skip_empty_word(self) -> None:
        """Test that empty words are skipped."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "word.json"
            path.write_text(
                '{"word": "", "pinyin": ""},\n'
                '{"word": "银行", "pinyin": "yín háng"},\n',
                encoding="utf-8",
            )
            result = _load_word_pinyin_map(path)
            self.assertNotIn("", result)
            self.assertIn("银行", result)

    def test_skip_invalid_json(self) -> None:
        """Test that lines with invalid JSON raise exception (current behavior)."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "word.json"
            path.write_text(
                '{"word": "银行", "pinyin": "yín háng"},\n'
                'invalid json line,\n'
                '{"word": "中国", "pinyin": "zhōng guó"},\n',
                encoding="utf-8",
            )
            # Current implementation raises exception on invalid JSON
            # This is acceptable behavior - invalid data should be fixed
            with self.assertRaises(json.JSONDecodeError):
                _load_word_pinyin_map(path)

    def test_streaming_json_lines(self) -> None:
        """Test loading streaming JSON lines format."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "word.json"
            path.write_text(
                "{\"word\": \"银行\", \"pinyin\": \"yín háng\"},\n"
                "{\"word\": \"中国\", \"pinyin\": \"zhōng guó\"},\n",
                encoding="utf-8",
            )
            result = _load_word_pinyin_map(path)
            self.assertIn("银行", result)
            self.assertIn("中国", result)


class TestLoadCharBase(unittest.TestCase):
    """Tests for _load_char_base function."""

    def test_load_valid_chars(self) -> None:
        """Test loading valid character-pinyin mappings."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "char_base.json"
            # Write as JSONL format
            path.write_text(
                '{"index": 1, "char": "中", "pinyin": ["zhōng", "zhòng"]},\n'
                '{"index": 2, "char": "国", "pinyin": ["guó"]},\n',
                encoding="utf-8",
            )
            result = _load_char_base(path)
            self.assertEqual(result["中"], ["zhōng", "zhòng"])
            self.assertEqual(result["国"], ["guó"])

    def test_skip_invalid_entries(self) -> None:
        """Test that invalid entries are skipped."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "char_base.json"
            path.write_text(
                '{"index": 1, "char": "中", "pinyin": ["zhōng"]},\n'
                '{"index": 2, "char": "国"},\n'  # Missing pinyin
                '{"index": 3, "pinyin": ["guó"]},\n',  # Missing char
                encoding="utf-8",
            )
            result = _load_char_base(path)
            self.assertIn("中", result)
            self.assertNotIn("国", result)
            self.assertNotIn("guó", result)


class TestLoadPolyphoneCandidates(unittest.TestCase):
    """Tests for _load_polyphone_candidates function."""

    def test_load_valid(self) -> None:
        """Test loading valid polyphone candidates."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "polyphone.json"
            path.write_text(
                json.dumps([
                    {"index": 1, "char": "中", "pinyin": ["zhōng", "zhòng"]},
                    {"index": 2, "char": "长", "pinyin": ["cháng", "zhǎng"]},
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            result = _load_polyphone_candidates(path)
            self.assertEqual(result["中"], ["zhōng", "zhòng"])
            self.assertEqual(result["长"], ["cháng", "zhǎng"])

    def test_skip_invalid_entries(self) -> None:
        """Test that invalid entries are skipped."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "polyphone.json"
            path.write_text(
                json.dumps([
                    {"index": 1, "char": "中", "pinyin": ["zhōng", "zhòng"]},
                    {"index": 2, "char": "长"},  # Missing pinyin
                    "invalid entry",
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            result = _load_polyphone_candidates(path)
            self.assertIn("中", result)
            self.assertNotIn("长", result)

    def test_non_array_input(self) -> None:
        """Test handling of non-array JSON input."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "polyphone.json"
            path.write_text(
                json.dumps({"not": "an array"}),
                encoding="utf-8",
            )
            result = _load_polyphone_candidates(path)
            self.assertEqual(result, {})


class TestLoadPolyphoneDisambig(unittest.TestCase):
    """Tests for _load_polyphone_disambig function."""

    def test_load_valid(self) -> None:
        """Test loading valid disambiguation data."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "polyphone_disambig.json"
            path.write_text(
                json.dumps({
                    "schema": "test",
                    "thresholds": {"min_support": 5, "min_prob": 0.85},
                    "items": [
                        {"char": "中", "candidates": ["zhōng", "zhòng"], "default": "zhōng"},
                        {"char": "长", "candidates": ["cháng", "zhǎng"], "default": "cháng"},
                    ],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            by_char, thresholds = _load_polyphone_disambig(path)
            self.assertIn("中", by_char)
            self.assertIn("长", by_char)
            self.assertEqual(thresholds["min_support"], 5)
            self.assertEqual(thresholds["min_prob"], 0.85)

    def test_default_thresholds(self) -> None:
        """Test that default thresholds are used when not specified."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "polyphone_disambig.json"
            path.write_text(
                json.dumps({
                    "items": [],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            by_char, thresholds = _load_polyphone_disambig(path)
            self.assertEqual(by_char, {})
            self.assertIsInstance(thresholds, dict)

    def test_invalid_thresholds_type(self) -> None:
        """Test handling of invalid thresholds type."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "polyphone_disambig.json"
            path.write_text(
                json.dumps({
                    "thresholds": "invalid",
                    "items": [],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            by_char, thresholds = _load_polyphone_disambig(path)
            self.assertIsInstance(thresholds, dict)


class TestLoadOverridesRules(unittest.TestCase):
    """Tests for _load_overrides_rules function."""

    def test_load_existing_file(self) -> None:
        """Test loading existing overrides file."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "overrides.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "rules": [
                        {"id": "rule1", "priority": 100},
                        {"id": "rule2", "priority": 50},
                    ],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            result = _load_overrides_rules(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["id"], "rule1")

    def test_create_missing_file(self) -> None:
        """Test creating overrides file when missing."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "overrides.json"
            self.assertFalse(path.exists())
            result = _load_overrides_rules(path)
            self.assertEqual(result, [])
            self.assertTrue(path.exists())
            # Verify the created file has proper structure
            content = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(content["schema_version"], 1)
            self.assertEqual(content["rules"], [])

    def test_non_array_rules(self) -> None:
        """Test handling of non-array rules field."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "overrides.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "rules": "not an array",
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            result = _load_overrides_rules(path)
            self.assertEqual(result, [])


class TestLoadLexicon(unittest.TestCase):
    """Tests for _load_lexicon function."""

    def test_load_with_items_field(self) -> None:
        """Test loading lexicon with items field."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lexicon.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "items": [
                        {"word": "银行", "pinyin": "yín háng"},
                        {"word": "中国", "pinyin": "zhōng guó"},
                    ],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            result = _load_lexicon(path)
            self.assertEqual(result["银行"], "yín háng")
            self.assertEqual(result["中国"], "zhōng guó")

    def test_load_simple_dict(self) -> None:
        """Test loading simple dict format."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lexicon.json"
            path.write_text(
                json.dumps({
                    "银行": "yín háng",
                    "中国": "zhōng guó",
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            result = _load_lexicon(path)
            self.assertEqual(result["银行"], "yín háng")
            self.assertEqual(result["中国"], "zhōng guó")

    def test_skip_non_han_words(self) -> None:
        """Test that words with non-Han characters are skipped."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lexicon.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "items": [
                        {"word": "银行", "pinyin": "yín háng"},
                        {"word": "OpenAI", "pinyin": "OpenAI"},  # Should be skipped
                    ],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            result = _load_lexicon(path)
            self.assertIn("银行", result)
            self.assertNotIn("OpenAI", result)

    def test_missing_file(self) -> None:
        """Test handling of missing file."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lexicon.json"
            result = _load_lexicon(path)
            self.assertEqual(result, {})


class TestPinyinResources(unittest.TestCase):
    """Tests for PinyinResources class."""

    def _create_test_files(self, root: Path) -> None:
        """Create all necessary test files."""
        root.joinpath("word.json").write_text(
            '{"word": "银行", "pinyin": "yín háng"},\n',
            encoding="utf-8",
        )
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "银", "pinyin": ["yín"]},\n'
            '{"index": 2, "char": "行", "pinyin": ["xíng", "háng"]},\n',
            encoding="utf-8",
        )
        root.joinpath("polyphone.json").write_text(
            json.dumps([
                {"index": 1, "char": "行", "pinyin": ["xíng", "háng"]},
            ], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps({
                "schema": "test",
                "thresholds": {"min_support": 5, "min_prob": 0.85},
                "items": [],
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("overrides.json").write_text(
            json.dumps({"schema_version": 1, "rules": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("lexicon.json").write_text(
            json.dumps({"schema_version": 1, "items": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_load_from_dir(self) -> None:
        """Test loading all resources from directory."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_files(root)
            resources = PinyinResources.load_from_dir(root)

            self.assertIn("银行", resources.word_pinyin)
            self.assertIn("银", resources.char_base)
            self.assertIn("行", resources.polyphone_candidates)
            self.assertIsInstance(resources.disambig_thresholds, dict)

    def test_combined_word_pinyin(self) -> None:
        """Test combining word and lexicon pinyin."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_files(root)
            # Add lexicon entry
            root.joinpath("lexicon.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "items": [
                        {"word": "中国", "pinyin": "zhōng guó"},
                    ],
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            resources = PinyinResources.load_from_dir(root)
            combined = resources.combined_word_pinyin()

            self.assertIn("银行", combined)
            self.assertIn("中国", combined)
            self.assertEqual(combined["银行"], "yín háng")
            self.assertEqual(combined["中国"], "zhōng guó")

    def test_lexicon_overrides_word(self) -> None:
        """Test that lexicon entries override word entries."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_files(root)
            # Add conflicting lexicon entry
            root.joinpath("lexicon.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "items": [
                        {"word": "银行", "pinyin": "different"},  # Should override
                    ],
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            resources = PinyinResources.load_from_dir(root)
            combined = resources.combined_word_pinyin()

            self.assertEqual(combined["银行"], "different")


if __name__ == "__main__":
    unittest.main()
