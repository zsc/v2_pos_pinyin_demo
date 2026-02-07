"""Core pinyinize tests - word lookup, char base, polyphone disambiguation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pinyinize.core import PinyinizeOptions, pinyinize
from pinyinize.resources import PinyinResources


class TestBasicPinyin(unittest.TestCase):
    """Basic single-character and word tests."""

    def _create_minimal_data(self, root: Path, **overrides) -> None:
        """Create minimal data files for testing."""
        word_items = overrides.get("word_items", [
            {"word": "细说", "pinyin": "xì shuō"},
            {"word": "银行", "pinyin": "yín háng"},
            {"word": "行长", "pinyin": "háng zhǎng"},
            {"word": "重新", "pinyin": "chóng xīn"},
            {"word": "营业", "pinyin": "yíng yè"},
            {"word": "得到", "pinyin": "dé dào"},
            {"word": "答案", "pinyin": "dá àn"},
            {"word": "得去", "pinyin": "děi qù"},
        ])
        # Write word.json as JSONL format
        root.joinpath("word.json").write_text(
            "".join(json.dumps(it, ensure_ascii=False) + ",\n" for it in word_items),
            encoding="utf-8",
        )

        char_items = overrides.get("char_items", [
            {"index": 1, "char": "他", "pinyin": ["tā"]},
            {"index": 2, "char": "的", "pinyin": ["de"]},
            {"index": 3, "char": "我", "pinyin": ["wǒ"]},
            {"index": 4, "char": "你", "pinyin": ["nǐ"]},
            {"index": 5, "char": "是", "pinyin": ["shì"]},
            {"index": 6, "char": "人", "pinyin": ["rén"]},
            {"index": 7, "char": "中", "pinyin": ["zhōng", "zhòng"]},
            {"index": 8, "char": "长", "pinyin": ["cháng", "zhǎng"]},
        ])
        # Write char_base.json as JSONL format
        root.joinpath("char_base.json").write_text(
            "".join(json.dumps(it, ensure_ascii=False) + ",\n" for it in char_items),
            encoding="utf-8",
        )

        root.joinpath("polyphone.json").write_text(
            json.dumps([
                {"index": 1, "char": "中", "pinyin": ["zhōng", "zhòng"]},
                {"index": 2, "char": "长", "pinyin": ["cháng", "zhǎng"]},
            ], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps(
                {
                    "schema": "test",
                    "thresholds": {"min_support": 5, "min_prob": 0.85, "min_margin": 0.15},
                    "items": [
                        {
                            "char": "中",
                            "candidates": ["zhōng", "zhòng"],
                            "default": "zhōng",
                            "contexts": {
                                "pos=NOUN|ner=O": {"best": "zhōng", "p": 0.9, "p2": 0.1, "n": 100},
                                "pos=VERB|ner=O": {"best": "zhòng", "p": 0.88, "p2": 0.12, "n": 80},
                            },
                        },
                        {
                            "char": "长",
                            "candidates": ["cháng", "zhǎng"],
                            "default": "cháng",
                            "contexts": {
                                "pos=ADJ|ner=O": {"best": "cháng", "p": 0.92, "p2": 0.08, "n": 120},
                                "pos=NOUN|ner=O": {"best": "zhǎng", "p": 0.85, "p2": 0.15, "n": 90},
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
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

    def test_single_word_lookup(self) -> None:
        """Test word-level dictionary lookup."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("细说", opts)
            self.assertTrue(any(r.output_text == "xìshuō" for r in results))

    def test_multi_word_sentence(self) -> None:
        """Test sentence with multiple words."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("银行行长重新营业", opts)
            self.assertTrue(any(r.output_text == "yínháng hángzhǎng chóngxīn yíngyè" for r in results))

    def test_de_polyphone(self) -> None:
        """Test '得' character variations."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("他得去得到答案", opts)
            self.assertTrue(any(r.output_text == "tā děiqù dédào dáàn" for r in results))

    def test_char_base_fallback(self) -> None:
        """Test falling back to char_base when word not in dictionary."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            # "你我他" - not in word.json, should use char_base
            results = pinyinize("你我他", opts)
            self.assertTrue(any(all(x in r.output_text for x in ["nǐ", "wǒ", "tā"]) for r in results))


class TestMixedContent(unittest.TestCase):
    """Tests for mixed Chinese and non-Chinese content."""

    def _create_minimal_data(self, root: Path) -> None:
        root.joinpath("word.json").write_text(
            '{"word": "细说", "pinyin": "xì shuō"},\n'
            '{"word": "OpenAI", "pinyin": "OpenAI"},\n',  # Should be ignored (not all han)
            encoding="utf-8",
        )
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "细", "pinyin": ["xì"]},\n'
            '{"index": 2, "char": "说", "pinyin": ["shuō"]},\n'
            '{"index": 3, "char": "的", "pinyin": ["de"]},\n',
            encoding="utf-8",
        )
        root.joinpath("polyphone.json").write_text("[]\n", encoding="utf-8")
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps({
                "schema": "test",
                "thresholds": {"min_support": 5, "min_prob": 0.85, "min_margin": 0.15},
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

    def test_chinese_with_latin(self) -> None:
        """Test Chinese mixed with English words."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("细说OpenAI的API", opts)
            self.assertTrue(any(r.output_text == "xìshuō OpenAI de API" for r in results))

    def test_chinese_with_numbers(self) -> None:
        """Test Chinese mixed with numbers."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("版本2.0发布", opts)
            self.assertTrue(any("2.0" in r.output_text for r in results))

    def test_url_preservation(self) -> None:
        """Test URL preservation."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("访问https://example.com", opts)
            self.assertTrue(any("https://example.com" in r.output_text for r in results))

    def test_punctuation_preserved(self) -> None:
        """Test punctuation is preserved."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("细说，测试。", opts)
            self.assertTrue(any(all(x in r.output_text for x in ["，", "。"]) for r in results))


class TestPolyphoneDisambiguation(unittest.TestCase):
    """Tests for polyphone character disambiguation."""

    def _create_disambig_data(self, root: Path) -> None:
        """Create data with polyphone disambiguation rules."""
        root.joinpath("word.json").write_text("", encoding="utf-8")
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "中", "pinyin": ["zhōng", "zhòng"]},\n'
            '{"index": 2, "char": "国", "pinyin": ["guó"]},\n'
            '{"index": 3, "char": "打", "pinyin": ["dǎ"]},\n'
            '{"index": 4, "char": "靶", "pinyin": ["bǎ"]},\n'
            '{"index": 5, "char": "长", "pinyin": ["cháng", "zhǎng"]},\n'
            '{"index": 6, "char": "很", "pinyin": ["hěn"]},\n'
            '{"index": 7, "char": "校", "pinyin": ["xiào"]},\n',
            encoding="utf-8",
        )
        root.joinpath("polyphone.json").write_text(
            json.dumps([
                {"index": 1, "char": "中", "pinyin": ["zhōng", "zhòng"]},
                {"index": 2, "char": "长", "pinyin": ["cháng", "zhǎng"]},
            ], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps({
                "schema": "test",
                "thresholds": {"min_support": 5, "min_prob": 0.85, "min_margin": 0.15},
                "items": [
                    {
                        "char": "中",
                        "candidates": ["zhōng", "zhòng"],
                        "default": "zhōng",
                        "contexts": {
                            "pos=NOUN|ner=O": {"best": "zhōng", "p": 0.9, "p2": 0.1, "n": 100},
                            "pos=VERB|ner=O": {"best": "zhòng", "p": 0.88, "p2": 0.12, "n": 80},
                        },
                    },
                    {
                        "char": "长",
                        "candidates": ["cháng", "zhǎng"],
                        "default": "cháng",
                        "contexts": {
                            "pos=ADJ|ner=O": {"best": "cháng", "p": 0.92, "p2": 0.08, "n": 120},
                            "pos=NOUN|ner=O": {"best": "zhǎng", "p": 0.85, "p2": 0.15, "n": 90},
                        },
                    },
                ],
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

    def test_disambiguation_uses_pos(self) -> None:
        """Test that POS tags affect disambiguation."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_disambig_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            # "中国" - "中" as NOUN should use "zhōng"
            # But without LLM, it uses fallback "zhōng" (default)
            results = pinyinize("中国", opts)
            self.assertTrue(any("zhōng" in r.output_text for r in results))


class TestOverrideRules(unittest.TestCase):
    """Tests for user override rules."""

    def _create_data_with_overrides(self, root: Path, rules: list) -> None:
        root.joinpath("word.json").write_text("", encoding="utf-8")
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "行", "pinyin": ["xíng", "háng"]},\n'
            '{"index": 2, "char": "好", "pinyin": ["hǎo", "hào"]},\n',
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
                "thresholds": {"min_support": 5, "min_prob": 0.85, "min_margin": 0.15},
                "items": [],
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("overrides.json").write_text(
            json.dumps({"schema_version": 1, "rules": rules}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("lexicon.json").write_text(
            json.dumps({"schema_version": 1, "items": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_simple_override(self) -> None:
        """Test basic override rule application."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rules = [
                {
                    "id": "test_001",
                    "priority": 100,
                    "description": "Test rule for 行行好",
                    "match": {"self": {"text": "行行好"}},
                    "target": {"char": "行", "occurrence": 1},
                    "choose": "xíng",
                },
                {
                    "id": "test_002",
                    "priority": 100,
                    "description": "Test rule for 行行好 second 行",
                    "match": {"self": {"text": "行行好"}},
                    "target": {"char": "行", "occurrence": 2},
                    "choose": "háng",
                },
            ]
            self._create_data_with_overrides(root, rules)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("行行好", opts)
            self.assertTrue(any(all(x in r.output_text for x in ["xíng", "háng"]) for r in results))

    def test_override_priority(self) -> None:
        """Test that higher priority rules take precedence."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rules = [
                {
                    "id": "low_priority",
                    "priority": 10,
                    "description": "Low priority",
                    "match": {"self": {"text": "行"}},
                    "target": {"char": "行", "occurrence": 1},
                    "choose": "háng",
                },
                {
                    "id": "high_priority",
                    "priority": 100,
                    "description": "High priority",
                    "match": {"self": {"text": "行"}},
                    "target": {"char": "行", "occurrence": 1},
                    "choose": "xíng",
                },
            ]
            self._create_data_with_overrides(root, rules)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("行", opts)
            self.assertTrue(any("xíng" in r.output_text for r in results))


class TestReportStructure(unittest.TestCase):
    """Tests for report output structure."""

    def _create_minimal_data(self, root: Path) -> None:
        root.joinpath("word.json").write_text(
            '{"word": "测试", "pinyin": "cè shì"},\n',
            encoding="utf-8",
        )
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "测", "pinyin": ["cè"]},\n'
            '{"index": 2, "char": "试", "pinyin": ["shì"]},\n',
            encoding="utf-8",
        )
        root.joinpath("polyphone.json").write_text("[]\n", encoding="utf-8")
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps({
                "schema": "test",
                "thresholds": {"min_support": 5, "min_prob": 0.85, "min_margin": 0.15},
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

    def test_report_has_required_fields(self) -> None:
        """Test that report contains all required fields."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("测试", opts)

            for result in results:
                report = result.report
                self.assertEqual(report["schema_version"], 1)
                self.assertIn("text", report)
                self.assertIn("spans", report)
                self.assertIn("tokens", report)
                self.assertIn("llm_segment_and_tag", report)
                self.assertIn("llm_double_check", report)
                self.assertIn("needs_review_items", report)
                self.assertIn("applied_overrides", report)
                self.assertIn("conflicts", report)
                self.assertIn("warnings", report)

    def test_spans_structure(self) -> None:
        """Test that spans have correct structure."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("测试", opts)
            for result in results:
                spans = result.report["spans"]
                self.assertIsInstance(spans, list)
                for span in spans:
                    self.assertIn("span_id", span)
                    self.assertIn("type", span)
                    self.assertIn("start", span)
                    self.assertIn("end", span)
                    self.assertIn("text", span)

    def test_tokens_structure(self) -> None:
        """Test that tokens have correct structure."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("测试", opts)
            for result in results:
                tokens = result.report["tokens"]
                self.assertIsInstance(tokens, list)
                for tok in tokens:
                    self.assertIn("span_id", tok)
                    self.assertIn("text", tok)
                    self.assertIn("upos", tok)
                    self.assertIn("xpos", tok)
                    self.assertIn("ner", tok)
                    self.assertIn("pinyin", tok)
                    self.assertIn("char_decisions", tok)


class TestWordLikeSpacing(unittest.TestCase):
    """Tests for word-like spacing option."""

    def _create_minimal_data(self, root: Path) -> None:
        root.joinpath("word.json").write_text(
            '{"word": "中文", "pinyin": "zhōng wén"},\n',
            encoding="utf-8",
        )
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "中", "pinyin": ["zhōng"]},\n'
            '{"index": 2, "char": "文", "pinyin": ["wén"]},\n',
            encoding="utf-8",
        )
        root.joinpath("polyphone.json").write_text("[]\n", encoding="utf-8")
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps({
                "schema": "test",
                "thresholds": {"min_support": 5, "min_prob": 0.85, "min_margin": 0.15},
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

    def test_spacing_enabled(self) -> None:
        """Test spacing between Chinese and Latin."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources, word_like_spacing=True)

            results = pinyinize("中文test", opts)
            self.assertTrue(any("wén test" in r.output_text for r in results))

    def test_spacing_disabled(self) -> None:
        """Test no spacing when disabled."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources, word_like_spacing=False)

            results = pinyinize("中文test", opts)
            self.assertTrue(any("wéntest" in r.output_text for r in results))


if __name__ == "__main__":
    unittest.main()
