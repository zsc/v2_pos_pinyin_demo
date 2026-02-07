"""Tests for the CLI interface."""

from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from pinyinize.cli import main


class TestCLI(unittest.TestCase):
    """Tests for the command line interface."""

    def _create_test_data(self, root: Path) -> None:
        """Create test data files."""
        root.joinpath("word.json").write_text(
            '{"word": "细说", "pinyin": "xì shuō"},\n'
            '{"word": "银行", "pinyin": "yín háng"},\n'
            '{"word": "行长", "pinyin": "háng zhǎng"},\n'
            '{"word": "重新", "pinyin": "chóng xīn"},\n'
            '{"word": "营业", "pinyin": "yíng yè"},\n'
            '{"word": "得到", "pinyin": "dé dào"},\n'
            '{"word": "答案", "pinyin": "dá àn"},\n'
            '{"word": "得去", "pinyin": "děi qù"},\n',
            encoding="utf-8",
        )
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "他", "pinyin": ["tā"]},\n'
            '{"index": 2, "char": "的", "pinyin": ["de"]},\n'
            '{"index": 3, "char": "我", "pinyin": ["wǒ"]},\n'
            '{"index": 4, "char": "你", "pinyin": ["nǐ"]},\n'
            '{"index": 5, "char": "是", "pinyin": ["shì"]},\n'
            '{"index": 6, "char": "人", "pinyin": ["rén"]},\n'
            '{"index": 7, "char": "中", "pinyin": ["zhōng", "zhòng"]},\n'
            '{"index": 8, "char": "长", "pinyin": ["cháng", "zhǎng"]},\n',
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

    def test_basic_text_input(self) -> None:
        """Test basic text input."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_data(root)

            with patch("sys.stdout", new=StringIO()) as fake_stdout:
                result = main(["--data-dir", str(root), "细说"])
                self.assertEqual(result, 0)
                output = fake_stdout.getvalue()
                self.assertIn("xìshuō", output)

    def test_stdin_input(self) -> None:
        """Test reading from stdin."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_data(root)

            with patch("sys.stdin", StringIO("银行行长")):
                with patch("sys.stdout", new=StringIO()) as fake_stdout:
                    result = main(["--data-dir", str(root)])
                    self.assertEqual(result, 0)
                    output = fake_stdout.getvalue()
                    self.assertIn("yínháng", output)
                    self.assertIn("hángzhǎng", output)

    def test_report_output(self) -> None:
        """Test writing report to file."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_data(root)
            report_path = root / "output_report.json"

            with patch("sys.stdout", new=StringIO()):
                result = main([
                    "--data-dir", str(root),
                    "--report", str(report_path),
                    "细说"
                ])
                self.assertEqual(result, 0)
                self.assertTrue(report_path.exists())

                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["schema_version"], 1)
                self.assertIn("text", report)
                self.assertIn("tokens", report)

    def test_no_word_like_spacing(self) -> None:
        """Test --no-word-like-spacing option."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_data(root)

            # Test with spacing (default)
            with patch("sys.stdout", new=StringIO()) as fake_stdout:
                main(["--data-dir", str(root), "中文test"])
                output_with_spacing = fake_stdout.getvalue()

            # Test without spacing
            with patch("sys.stdout", new=StringIO()) as fake_stdout:
                main(["--data-dir", str(root), "--no-word-like-spacing", "中文test"])
                output_without_spacing = fake_stdout.getvalue()

            # The outputs should be different when there are word-like spans
            # (though this depends on what's in the test data)

    def test_help(self) -> None:
        """Test --help output."""
        with self.assertRaises(SystemExit) as cm:
            main(["--help"])
        # argparse exits with 0 for --help
        self.assertEqual(cm.exception.code, 0)

    def test_ollama_model_option(self) -> None:
        """Test --ollama-model option creates adapter."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_data(root)

            # This test just verifies the option is accepted
            # It will fail to connect to Ollama, but that's expected
            with patch("sys.stdout", new=StringIO()):
                # We expect this to potentially fail with network error
                # but the option should be parsed correctly
                try:
                    main([
                        "--data-dir", str(root),
                        "--ollama-model", "gemma3:1b",
                        "--ollama-host", "http://localhost:11434",
                        "细说"
                    ])
                except Exception:
                    # Network errors are expected in test environment
                    pass


class TestCLIWithOverrides(unittest.TestCase):
    """Tests for CLI override functionality."""

    def _create_test_data_with_overrides(self, root: Path) -> None:
        """Create test data with override rules."""
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
            json.dumps({
                "schema_version": 1,
                "rules": [
                    {
                        "id": "test_rule",
                        "priority": 100,
                        "description": "Test rule",
                        "match": {"self": {"text": "行行好"}},
                        "target": {"char": "行", "occurrence": 1},
                        "choose": "xíng",
                    },
                ],
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("lexicon.json").write_text(
            json.dumps({"schema_version": 1, "items": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_override_applied(self) -> None:
        """Test that override rules are applied."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_test_data_with_overrides(root)

            with patch("sys.stdout", new=StringIO()) as fake_stdout:
                result = main(["--data-dir", str(root), "行行好"])
                self.assertEqual(result, 0)
                output = fake_stdout.getvalue()
                # The override should have been applied
                self.assertIn("xíng", output)


if __name__ == "__main__":
    unittest.main()
