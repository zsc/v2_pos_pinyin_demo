"""Integration tests for pinyinize system.

These tests verify end-to-end functionality with realistic scenarios.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pinyinize.core import PinyinizeOptions, pinyinize
from pinyinize.resources import PinyinResources


class TestAcceptanceCriteria(unittest.TestCase):
    """Tests based on the acceptance criteria from CLAUDE.md."""

    def _create_complete_data(self, root: Path) -> None:
        """Create complete test data covering acceptance criteria."""
        # word.json - comprehensive word list
        word_items = [
            {"word": "细说", "pinyin": "xì shuō"},
            {"word": "银行", "pinyin": "yín háng"},
            {"word": "行长", "pinyin": "háng zhǎng"},
            {"word": "重新", "pinyin": "chóng xīn"},
            {"word": "营业", "pinyin": "yíng yè"},
            {"word": "得到", "pinyin": "dé dào"},
            {"word": "答案", "pinyin": "dá àn"},
            {"word": "得去", "pinyin": "děi qù"},
            {"word": "同行", "pinyin": "tóng háng"},
            {"word": "行走", "pinyin": "xíng zǒu"},
            {"word": "重要", "pinyin": "zhòng yào"},
            {"word": "重复", "pinyin": "chóng fù"},
            {"word": "快乐", "pinyin": "kuài lè"},
            {"word": "音乐", "pinyin": "yīn yuè"},
            {"word": "目的", "pinyin": "mù dì"},
            {"word": "的确", "pinyin": "dí què"},
        ]
        root.joinpath("word.json").write_text(
            '{"word": "细说", "pinyin": "xì shuō"},\n'
            '{"word": "银行", "pinyin": "yín háng"},\n'
            '{"word": "行长", "pinyin": "háng zhǎng"},\n'
            '{"word": "重新", "pinyin": "chóng xīn"},\n'
            '{"word": "营业", "pinyin": "yíng yè"},\n'
            '{"word": "得到", "pinyin": "dé dào"},\n'
            '{"word": "答案", "pinyin": "dá àn"},\n'
            '{"word": "得去", "pinyin": "děi qù"},\n'
            '{"word": "同行", "pinyin": "tóng háng"},\n'
            '{"word": "行走", "pinyin": "xíng zǒu"},\n'
            '{"word": "重要", "pinyin": "zhòng yào"},\n'
            '{"word": "重复", "pinyin": "chóng fù"},\n'
            '{"word": "快乐", "pinyin": "kuài lè"},\n'
            '{"word": "音乐", "pinyin": "yīn yuè"},\n'
            '{"word": "目的", "pinyin": "mù dì"},\n'
            '{"word": "的确", "pinyin": "dí què"},\n',
            encoding="utf-8",
        )

        # char_base.json - character mappings
        char_items = [
            {"index": 1, "char": "细", "pinyin": ["xì"]},
            {"index": 2, "char": "说", "pinyin": ["shuō"]},
            {"index": 3, "char": "银", "pinyin": ["yín"]},
            {"index": 4, "char": "行", "pinyin": ["xíng", "háng"]},
            {"index": 5, "char": "长", "pinyin": ["cháng", "zhǎng"]},
            {"index": 6, "char": "重", "pinyin": ["zhòng", "chóng"]},
            {"index": 7, "char": "新", "pinyin": ["xīn"]},
            {"index": 8, "char": "营", "pinyin": ["yíng"]},
            {"index": 9, "char": "业", "pinyin": ["yè"]},
            {"index": 10, "char": "得", "pinyin": ["de", "dé", "děi"]},
            {"index": 11, "char": "到", "pinyin": ["dào"]},
            {"index": 12, "char": "答", "pinyin": ["dá"]},
            {"index": 13, "char": "案", "pinyin": ["àn"]},
            {"index": 14, "char": "去", "pinyin": ["qù"]},
            {"index": 15, "char": "他", "pinyin": ["tā"]},
            {"index": 16, "char": "我", "pinyin": ["wǒ"]},
            {"index": 17, "char": "你", "pinyin": ["nǐ"]},
            {"index": 18, "char": "的", "pinyin": ["de", "dí", "dì"]},
            {"index": 19, "char": "同", "pinyin": ["tóng"]},
            {"index": 20, "char": "走", "pinyin": ["zǒu"]},
            {"index": 21, "char": "要", "pinyin": ["yào"]},
            {"index": 22, "char": "复", "pinyin": ["fù"]},
            {"index": 23, "char": "快", "pinyin": ["kuài"]},
            {"index": 24, "char": "乐", "pinyin": ["lè", "yuè"]},
            {"index": 25, "char": "音", "pinyin": ["yīn"]},
            {"index": 26, "char": "目", "pinyin": ["mù"]},
            {"index": 27, "char": "的", "pinyin": ["dí", "dì", "de"]},
            {"index": 28, "char": "确", "pinyin": ["què"]},
            {"index": 29, "char": "中", "pinyin": ["zhōng", "zhòng"]},
            {"index": 30, "char": "国", "pinyin": ["guó"]},
        ]
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "细", "pinyin": ["xì"]},\n'
            '{"index": 2, "char": "说", "pinyin": ["shuō"]},\n'
            '{"index": 3, "char": "银", "pinyin": ["yín"]},\n'
            '{"index": 4, "char": "行", "pinyin": ["xíng", "háng"]},\n'
            '{"index": 5, "char": "长", "pinyin": ["cháng", "zhǎng"]},\n'
            '{"index": 6, "char": "重", "pinyin": ["zhòng", "chóng"]},\n'
            '{"index": 7, "char": "新", "pinyin": ["xīn"]},\n'
            '{"index": 8, "char": "营", "pinyin": ["yíng"]},\n'
            '{"index": 9, "char": "业", "pinyin": ["yè"]},\n'
            '{"index": 10, "char": "得", "pinyin": ["de", "dé", "děi"]},\n'
            '{"index": 11, "char": "到", "pinyin": ["dào"]},\n'
            '{"index": 12, "char": "答", "pinyin": ["dá"]},\n'
            '{"index": 13, "char": "案", "pinyin": ["àn"]},\n'
            '{"index": 14, "char": "去", "pinyin": ["qù"]},\n'
            '{"index": 15, "char": "他", "pinyin": ["tā"]},\n'
            '{"index": 16, "char": "我", "pinyin": ["wǒ"]},\n'
            '{"index": 17, "char": "你", "pinyin": ["nǐ"]},\n'
            '{"index": 18, "char": "的", "pinyin": ["de", "dí", "dì"]},\n'
            '{"index": 19, "char": "同", "pinyin": ["tóng"]},\n'
            '{"index": 20, "char": "走", "pinyin": ["zǒu"]},\n'
            '{"index": 21, "char": "要", "pinyin": ["yào"]},\n'
            '{"index": 22, "char": "复", "pinyin": ["fù"]},\n'
            '{"index": 23, "char": "快", "pinyin": ["kuài"]},\n'
            '{"index": 24, "char": "乐", "pinyin": ["lè", "yuè"]},\n'
            '{"index": 25, "char": "音", "pinyin": ["yīn"]},\n'
            '{"index": 26, "char": "目", "pinyin": ["mù"]},\n'
            '{"index": 27, "char": "的", "pinyin": ["dí", "dì", "de"]},\n'
            '{"index": 28, "char": "确", "pinyin": ["què"]},\n'
            '{"index": 29, "char": "中", "pinyin": ["zhōng", "zhòng"]},\n'
            '{"index": 30, "char": "国", "pinyin": ["guó"]},\n',
            encoding="utf-8",
        )

        # polyphone.json - polyphone definitions
        root.joinpath("polyphone.json").write_text(
            json.dumps([
                {"index": 1, "char": "行", "pinyin": ["xíng", "háng"]},
                {"index": 2, "char": "长", "pinyin": ["cháng", "zhǎng"]},
                {"index": 3, "char": "重", "pinyin": ["zhòng", "chóng"]},
                {"index": 4, "char": "得", "pinyin": ["de", "dé", "děi"]},
                {"index": 5, "char": "的", "pinyin": ["de", "dí", "dì"]},
                {"index": 6, "char": "乐", "pinyin": ["lè", "yuè"]},
                {"index": 7, "char": "中", "pinyin": ["zhōng", "zhòng"]},
            ], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # polyphone_disambig.json - disambiguation rules
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps({
                "schema": "complete_test",
                "thresholds": {"min_support": 5, "min_prob": 0.85, "min_margin": 0.15},
                "items": [
                    {
                        "char": "行",
                        "candidates": ["xíng", "háng"],
                        "default": "xíng",
                        "contexts": {
                            "pos=NOUN|ner=O": {"best": "háng", "p": 0.88, "p2": 0.12, "n": 100},
                            "pos=VERB|ner=O": {"best": "xíng", "p": 0.90, "p2": 0.10, "n": 120},
                        },
                    },
                    {
                        "char": "长",
                        "candidates": ["cháng", "zhǎng"],
                        "default": "cháng",
                        "contexts": {
                            "pos=ADJ|ner=O": {"best": "cháng", "p": 0.92, "p2": 0.08, "n": 150},
                            "pos=NOUN|ner=O": {"best": "zhǎng", "p": 0.87, "p2": 0.13, "n": 110},
                        },
                    },
                    {
                        "char": "重",
                        "candidates": ["zhòng", "chóng"],
                        "default": "zhòng",
                        "contexts": {
                            "pos=ADJ|ner=O": {"best": "zhòng", "p": 0.91, "p2": 0.09, "n": 130},
                            "pos=ADV|ner=O": {"best": "chóng", "p": 0.89, "p2": 0.11, "n": 95},
                        },
                    },
                    {
                        "char": "得",
                        "candidates": ["de", "dé", "děi"],
                        "default": "de",
                        "contexts": {
                            "pos=PART|ner=O": {"best": "de", "p": 0.95, "p2": 0.03, "n": 200},
                            "pos=VERB|ner=O": {"best": "dé", "p": 0.88, "p2": 0.08, "n": 85},
                            "pos=AUX|ner=O": {"best": "děi", "p": 0.86, "p2": 0.10, "n": 70},
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

    def test_criterion_1_basic(self) -> None:
        """Test criterion 1: 细说 -> xìshuō."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_complete_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("细说", opts)
            self.assertTrue(any(r.output_text == "xìshuō" for r in results))

    def test_criterion_2_bank_director(self) -> None:
        """Test criterion 2: 银行行长重新营业 -> yínháng hángzhǎng chóngxīn yíngyè."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_complete_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("银行行长重新营业", opts)
            self.assertTrue(any(r.output_text == "yínháng hángzhǎng chóngxīn yíngyè" for r in results))

    def test_criterion_3_de_polyphone(self) -> None:
        """Test criterion 3: 他得去得到答案 -> tā děiqù dédào dáàn."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_complete_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("他得去得到答案", opts)
            self.assertTrue(any(r.output_text == "tā děiqù dédào dáàn" for r in results))

    def test_criterion_4_mixed_content(self) -> None:
        """Test criterion 4: Mixed content preservation."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_complete_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("细说OpenAI的API v2.0：https://openai.com", opts)
            self.assertTrue(any("https://openai.com" in r.output_text for r in results))
            self.assertTrue(any("OpenAI" in r.output_text for r in results))
            self.assertTrue(any("v2.0" in r.output_text for r in results))

    def test_url_character_exact(self) -> None:
        """Test that URL is preserved character-by-character."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_complete_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            url = "https://openai.com/api/v2?key=value"
            results = pinyinize(f"访问{url}即可", opts)
            self.assertTrue(any(url in r.output_text for r in results))

    def test_report_has_decision_sources(self) -> None:
        """Test report tracks decision sources."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_complete_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("细说", opts)
            for result in results:
                report = result.report
                self.assertTrue(len(report["tokens"]) > 0)
                for token in report["tokens"]:
                    self.assertIn("char_decisions", token)
                    for decision in token["char_decisions"]:
                        self.assertIn("resolved_by", decision)
                        valid_sources = [
                            "word", "char_base", "polyphone_disambig",
                            "override", "llm_double_check", "user", "fallback", "unknown"
                        ]
                        self.assertIn(decision["resolved_by"], valid_sources)


class TestRealWorldScenarios(unittest.TestCase):
    """Real-world usage scenarios."""

    def _create_scenario_data(self, root: Path) -> None:
        """Create data for scenario tests."""
        root.joinpath("word.json").write_text(
            '{"word": "中国银行", "pinyin": "zhōng guó yín háng"},\n'
            '{"word": "工商银行", "pinyin": "gōng shāng yín háng"},\n'
            '{"word": "建设银行", "pinyin": "jiàn shè yín háng"},\n'
            '{"word": "农业银行", "pinyin": "nóng yè yín háng"},\n'
            '{"word": "中国人民银行", "pinyin": "zhōng guó rén mín yín háng"},\n'
            '{"word": "总经理", "pinyin": "zǒng jīng lǐ"},\n'
            '{"word": "董事长", "pinyin": "dǒng shì zhǎng"},\n'
            '{"word": "部长", "pinyin": "bù zhǎng"},\n'
            '{"word": "校长", "pinyin": "xiào zhǎng"},\n'
            '{"word": "市长", "pinyin": "shì zhǎng"},\n',
            encoding="utf-8",
        )

        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "中", "pinyin": ["zhōng", "zhòng"]},\n'
            '{"index": 2, "char": "国", "pinyin": ["guó"]},\n'
            '{"index": 3, "char": "银", "pinyin": ["yín"]},\n'
            '{"index": 4, "char": "行", "pinyin": ["xíng", "háng"]},\n'
            '{"index": 5, "char": "工", "pinyin": ["gōng"]},\n'
            '{"index": 6, "char": "商", "pinyin": ["shāng"]},\n'
            '{"index": 7, "char": "建", "pinyin": ["jiàn"]},\n'
            '{"index": 8, "char": "设", "pinyin": ["shè"]},\n'
            '{"index": 9, "char": "农", "pinyin": ["nóng"]},\n'
            '{"index": 10, "char": "业", "pinyin": ["yè"]},\n'
            '{"index": 11, "char": "人", "pinyin": ["rén"]},\n'
            '{"index": 12, "char": "民", "pinyin": ["mín"]},\n'
            '{"index": 13, "char": "总", "pinyin": ["zǒng"]},\n'
            '{"index": 14, "char": "经", "pinyin": ["jīng"]},\n'
            '{"index": 15, "char": "理", "pinyin": ["lǐ"]},\n'
            '{"index": 16, "char": "董", "pinyin": ["dǒng"]},\n'
            '{"index": 17, "char": "事", "pinyin": ["shì"]},\n'
            '{"index": 18, "char": "长", "pinyin": ["cháng", "zhǎng"]},\n'
            '{"index": 19, "char": "部", "pinyin": ["bù"]},\n'
            '{"index": 20, "char": "校", "pinyin": ["xiào", "jiào"]},\n'
            '{"index": 21, "char": "市", "pinyin": ["shì"]},\n',
            encoding="utf-8",
        )

        root.joinpath("polyphone.json").write_text(
            json.dumps([
                {"index": 1, "char": "中", "pinyin": ["zhōng", "zhòng"]},
                {"index": 2, "char": "行", "pinyin": ["xíng", "háng"]},
                {"index": 3, "char": "长", "pinyin": ["cháng", "zhǎng"]},
                {"index": 4, "char": "校", "pinyin": ["xiào", "jiào"]},
            ], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps({
                "schema": "scenario_test",
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

    def test_bank_names(self) -> None:
        """Test major Chinese bank names."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_scenario_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            test_cases = [
                ("中国银行", "zhōngguóyínháng"),
                ("工商银行", "gōngshāngyínháng"),
                ("建设银行", "jiànshèyínháng"),
                ("农业银行", "nóngyèyínháng"),
            ]

            for input_text, expected in test_cases:
                results = pinyinize(input_text, opts)
                self.assertTrue(any(r.output_text == expected for r in results), f"Failed for {input_text}")

    def test_job_titles(self) -> None:
        """Test job title pronunciations."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_scenario_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            test_cases = [
                ("总经理", "zǒngjīnglǐ"),
                ("董事长", "dǒngshìzhǎng"),
                ("部长", "bùzhǎng"),
                ("校长", "xiàozhǎng"),
                ("市长", "shìzhǎng"),
            ]

            for input_text, expected in test_cases:
                results = pinyinize(input_text, opts)
                self.assertTrue(any(r.output_text == expected for r in results), f"Failed for {input_text}")


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def _create_minimal_data(self, root: Path) -> None:
        root.joinpath("word.json").write_text("", encoding="utf-8")
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "测", "pinyin": ["cè"]},\n'
            '{"index": 2, "char": "试", "pinyin": ["shì"]},\n',
            encoding="utf-8",
        )
        root.joinpath("polyphone.json").write_text("[]\n", encoding="utf-8")
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps({
                "schema": "edge_test",
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

    def test_empty_input(self) -> None:
        """Test empty input."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("", opts)
            self.assertTrue(any(r.output_text == "" for r in results))

    def test_only_spaces(self) -> None:
        """Test input with only spaces."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("   ", opts)
            self.assertTrue(any(r.output_text == "   " for r in results))

    def test_only_punctuation(self) -> None:
        """Test input with only punctuation."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("，。！？", opts)
            self.assertTrue(any(r.output_text == "，。！？" for r in results))

    def test_emoji(self) -> None:
        """Test input with emoji."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("测试😀", opts)
            self.assertTrue(any("😀" in r.output_text for r in results))

    def test_numbers_and_chinese(self) -> None:
        """Test numbers mixed with Chinese."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            results = pinyinize("2024年测试", opts)
            self.assertTrue(any("2024" in r.output_text for r in results))
            self.assertTrue(any("cè" in r.output_text for r in results))


if __name__ == "__main__":
    unittest.main()
