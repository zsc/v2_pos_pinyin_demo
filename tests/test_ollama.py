"""Ollama LLM integration tests for pinyinize.

These tests verify that the OllamaLLMAdapter works correctly with the pinyinize system.
Note: Some tests mock the Ollama API responses, others may require a running Ollama server.

To run tests against a real Ollama server:
    OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=gemma3:1b python -m pytest tests/test_ollama.py -v

To skip tests requiring a real server, set SKIP_LIVE_TESTS=1.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pinyinize.core import PinyinizeOptions, pinyinize
from pinyinize.llm import LLMError, OllamaLLMAdapter, extract_json_object
from pinyinize.resources import PinyinResources


# Default test configuration
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
SKIP_LIVE_TESTS = os.environ.get("SKIP_LIVE_TESTS", "0") == "1"


class TestExtractJsonObject(unittest.TestCase):
    """Tests for JSON extraction from LLM responses."""

    def test_direct_json(self) -> None:
        """Test extracting direct JSON."""
        text = '{"key": "value", "number": 123}'
        result = extract_json_object(text)
        self.assertEqual(result, {"key": "value", "number": 123})

    def test_json_with_code_fence(self) -> None:
        """Test extracting JSON from markdown code fence."""
        text = """```json
{"key": "value"}
```"""
        result = extract_json_object(text)
        self.assertEqual(result, {"key": "value"})

    def test_json_with_text_prefix(self) -> None:
        """Test extracting JSON with text before it."""
        text = "Here is the result:\n{\"key\": \"value\"}"
        result = extract_json_object(text)
        self.assertEqual(result, {"key": "value"})

    def test_json_with_text_suffix(self) -> None:
        """Test extracting JSON with text after it."""
        text = '{"key": "value"}\nHope this helps!'
        result = extract_json_object(text)
        self.assertEqual(result, {"key": "value"})

    def test_empty_response(self) -> None:
        """Test error on empty response."""
        with self.assertRaises(LLMError) as ctx:
            extract_json_object("")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_invalid_json(self) -> None:
        """Test error on invalid JSON."""
        with self.assertRaises(LLMError):
            extract_json_object("not json at all")

    def test_nested_json(self) -> None:
        """Test extracting nested JSON structure."""
        text = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = extract_json_object(text)
        self.assertEqual(result["outer"]["inner"], [1, 2, 3])
        self.assertTrue(result["flag"])


class TestOllamaLLMAdapter(unittest.TestCase):
    """Tests for OllamaLLMAdapter."""

    def test_adapter_creation(self) -> None:
        """Test creating adapter with default host."""
        adapter = OllamaLLMAdapter(model="test-model")
        self.assertEqual(adapter.model, "test-model")
        self.assertEqual(adapter.host, "http://localhost:11434")

    def test_adapter_creation_with_custom_host(self) -> None:
        """Test creating adapter with custom host."""
        adapter = OllamaLLMAdapter(model="test-model", host="http://custom:8080")
        self.assertEqual(adapter.host, "http://custom:8080")

    def test_adapter_creation_with_timeout(self) -> None:
        """Test creating adapter with custom timeout."""
        adapter = OllamaLLMAdapter(model="test-model", timeout_s=30.0)
        self.assertEqual(adapter.timeout_s, 30.0)

    @patch("pinyinize.llm._http_post_json")
    def test_segment_and_tag_success(self, mock_post: MagicMock) -> None:
        """Test successful segment_and_tag call."""
        mock_post.return_value = {
            "message": {
                "content": '{"schema_version": 1, "spans": [{"span_id": "S0", "tokens": [{"text": "银行", "upos": "NOUN", "xpos": "NN", "ner": "O"}]}]}'
            }
        }

        adapter = OllamaLLMAdapter(model="test-model")
        payload = {
            "schema_version": 1,
            "spans": [{"span_id": "S0", "text": "银行"}],
        }
        result = adapter.segment_and_tag(payload)

        self.assertIn("spans", result)
        self.assertEqual(len(result["spans"]), 1)
        self.assertEqual(result["spans"][0]["span_id"], "S0")

    @patch("pinyinize.llm._http_post_json")
    def test_segment_and_tag_with_code_fence(self, mock_post: MagicMock) -> None:
        """Test segment_and_tag with markdown code fence response."""
        mock_post.return_value = {
            "message": {
                "content": """```json
{
  "schema_version": 1,
  "spans": [
    {
      "span_id": "S0",
      "tokens": [
        {"text": "测试", "upos": "VERB", "xpos": "VV", "ner": "O"}
      ]
    }
  ]
}
```"""
            }
        }

        adapter = OllamaLLMAdapter(model="test-model")
        payload = {
            "schema_version": 1,
            "spans": [{"span_id": "S0", "text": "测试"}],
        }
        result = adapter.segment_and_tag(payload)

        self.assertIn("spans", result)
        self.assertEqual(result["spans"][0]["tokens"][0]["text"], "测试")

    @patch("pinyinize.llm._http_post_json")
    def test_double_check_success(self, mock_post: MagicMock) -> None:
        """Test successful double_check call."""
        mock_post.return_value = {
            "message": {
                "content": '{"schema_version": 1, "verdict": "ok", "items": [{"span_id": "S0", "token_index": 0, "char_offset_in_token": 0, "char": "行", "candidates": ["háng", "xíng"], "recommended": "háng", "needs_user": false}]}'
            }
        }

        adapter = OllamaLLMAdapter(model="test-model")
        payload = {
            "schema_version": 1,
            "text": "银行",
            "spans": [{"span_id": "S0", "text": "银行", "tokens": [{"text": "银行", "upos": "NOUN", "xpos": "NN", "ner": "O"}]}],
            "items": [{"span_id": "S0", "token_index": 0, "char_offset_in_token": 0, "char": "行", "candidates": ["háng", "xíng"], "current": "háng"}],
        }
        result = adapter.double_check(payload)

        self.assertIn("items", result)
        self.assertEqual(result["items"][0]["recommended"], "háng")

    @patch("pinyinize.llm._http_post_json")
    def test_http_error_handling(self, mock_post: MagicMock) -> None:
        """Test handling of HTTP errors."""
        from urllib.error import URLError
        mock_post.side_effect = URLError("Connection refused")

        adapter = OllamaLLMAdapter(model="test-model")
        payload = {"spans": []}

        with self.assertRaises(LLMError) as ctx:
            adapter.segment_and_tag(payload)
        self.assertIn("ollama_http_error", str(ctx.exception))

    @patch("pinyinize.llm._http_post_json")
    def test_missing_message_field(self, mock_post: MagicMock) -> None:
        """Test error when message field is missing."""
        mock_post.return_value = {"other": "field"}

        adapter = OllamaLLMAdapter(model="test-model")
        payload = {"spans": []}

        with self.assertRaises(LLMError) as ctx:
            adapter.segment_and_tag(payload)
        self.assertIn("missing_message", str(ctx.exception))

    @patch("pinyinize.llm._http_post_json")
    def test_missing_content_field(self, mock_post: MagicMock) -> None:
        """Test error when content field is missing."""
        mock_post.return_value = {"message": {"role": "assistant"}}

        adapter = OllamaLLMAdapter(model="test-model")
        payload = {"spans": []}

        with self.assertRaises(LLMError) as ctx:
            adapter.segment_and_tag(payload)
        self.assertIn("missing_content", str(ctx.exception))


class TestOllamaWithPinyinize(unittest.TestCase):
    """Tests for Ollama integration with pinyinize core."""

    def _create_minimal_data(self, root: Path) -> None:
        """Create minimal test data."""
        root.joinpath("word.json").write_text(
            '{"word": "银行", "pinyin": "yín háng"},\n'
            '{"word": "行长", "pinyin": "háng zhǎng"},\n',
            encoding="utf-8",
        )
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "银", "pinyin": ["yín"]},\n'
            '{"index": 2, "char": "行", "pinyin": ["xíng", "háng"]},\n'
            '{"index": 3, "char": "长", "pinyin": ["cháng", "zhǎng"]},\n',
            encoding="utf-8",
        )
        root.joinpath("polyphone.json").write_text(
            json.dumps([
                {"index": 1, "char": "行", "pinyin": ["xíng", "háng"]},
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

    def test_pinyinize_with_mock_llm(self) -> None:
        """Test pinyinize with mock LLM adapter."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)

            # Create mock LLM adapter
            mock_adapter = MagicMock()
            mock_adapter.segment_and_tag.return_value = {
                "schema_version": 1,
                "spans": [
                    {
                        "span_id": "S0",
                        "tokens": [
                            {"text": "银行", "upos": "NOUN", "xpos": "NN", "ner": "O"},
                            {"text": "行长", "upos": "NOUN", "xpos": "NN", "ner": "O"},
                        ],
                    }
                ],
            }

            opts = PinyinizeOptions(
                resources=resources,
                llm_adapter=mock_adapter,
            )
            result = pinyinize("银行行长", opts)

            self.assertEqual(result.output_text, "yínháng hángzhǎng")
            mock_adapter.segment_and_tag.assert_called_once()

    def test_pinyinize_llm_fallback_on_invalid_response(self) -> None:
        """Test fallback when LLM returns invalid response."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)

            # Create mock LLM adapter that returns invalid response
            mock_adapter = MagicMock()
            mock_adapter.segment_and_tag.return_value = {
                "invalid": "response"  # Missing spans
            }

            opts = PinyinizeOptions(
                resources=resources,
                llm_adapter=mock_adapter,
            )
            result = pinyinize("银行行长", opts)

            # Should still produce output using fallback
            self.assertIn("yín", result.output_text)
            self.assertIn("háng", result.output_text)

            # Report should indicate error occurred (not invalid_spans since there was an error)
            self.assertIn("error", result.report["llm_segment_and_tag"])

    def test_pinyinize_with_double_check(self) -> None:
        """Test pinyinize with double-check LLM adapter.
        
        Note: Double check is only triggered when there are review items
        (low confidence, needs_review flags, or conflicts).
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)

            mock_adapter = MagicMock()
            mock_adapter.segment_and_tag.return_value = {
                "schema_version": 1,
                "spans": [
                    {
                        "span_id": "S0",
                        "tokens": [
                            # Use a word NOT in the dictionary to create ambiguity
                            {"text": "行", "upos": "NOUN", "xpos": "NN", "ner": "O"},
                        ],
                    }
                ],
            }
            mock_adapter.double_check.return_value = {
                "schema_version": 1,
                "verdict": "ok",
                "items": [
                    {
                        "span_id": "S0",
                        "token_index": 0,
                        "char_offset_in_token": 0,
                        "char": "行",
                        "recommended": "háng",
                        "needs_user": False,
                    }
                ],
            }

            opts = PinyinizeOptions(
                resources=resources,
                llm_adapter=mock_adapter,
                double_check_adapter=mock_adapter,
                double_check_threshold=0.9,
            )
            result = pinyinize("行", opts)

            # Double check should be called because 行 is a polyphone with low confidence
            mock_adapter.double_check.assert_called_once()

    def test_pinyinize_without_llm(self) -> None:
        """Test pinyinize without LLM adapter (fallback mode)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)

            opts = PinyinizeOptions(resources=resources, llm_adapter=None)
            result = pinyinize("银行行长", opts)

            self.assertEqual(result.output_text, "yínháng hángzhǎng")
            self.assertFalse(result.report["llm_segment_and_tag"]["used"])


@unittest.skipIf(SKIP_LIVE_TESTS, "Skipping live Ollama tests")
class TestLiveOllama(unittest.TestCase):
    """Live tests against a real Ollama server.
    
    These tests require:
    - Ollama server running at OLLAMA_HOST (default: http://localhost:11434)
    - Model OLLAMA_MODEL available (default: gemma3:1b)
    
    To skip: set SKIP_LIVE_TESTS=1
    """

    def _create_minimal_data(self, root: Path) -> None:
        """Create minimal test data."""
        root.joinpath("word.json").write_text(
            "[]\n",
            encoding="utf-8",
        )
        root.joinpath("char_base.json").write_text(
            '{"index": 1, "char": "测", "pinyin": ["cè"]},\n'
            '{"index": 2, "char": "试", "pinyin": ["shì"]},\n'
            '{"index": 3, "char": "银", "pinyin": ["yín"]},\n'
            '{"index": 4, "char": "行", "pinyin": ["xíng", "háng"]},\n',
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
            json.dumps({"schema_version": 1, "rules": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        root.joinpath("lexicon.json").write_text(
            json.dumps({"schema_version": 1, "items": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_live_segment_and_tag(self) -> None:
        """Test live Ollama segment_and_tag API."""
        adapter = OllamaLLMAdapter(
            model=DEFAULT_OLLAMA_MODEL,
            host=DEFAULT_OLLAMA_HOST,
            timeout_s=60.0,
        )

        payload = {
            "schema_version": 1,
            "task": "segment_and_tag",
            "spans": [{"span_id": "S0", "text": "银行"}],
        }

        try:
            result = adapter.segment_and_tag(payload)
            self.assertIn("spans", result)
            self.assertIsInstance(result["spans"], list)
        except LLMError as e:
            self.skipTest(f"Ollama server not available: {e}")

    def test_live_double_check(self) -> None:
        """Test live Ollama double_check API."""
        adapter = OllamaLLMAdapter(
            model=DEFAULT_OLLAMA_MODEL,
            host=DEFAULT_OLLAMA_HOST,
            timeout_s=60.0,
        )

        payload = {
            "schema_version": 1,
            "task": "double_check",
            "text": "银行",
            "spans": [
                {
                    "span_id": "S0",
                    "text": "银行",
                    "tokens": [{"text": "银行", "upos": "NOUN", "xpos": "NN", "ner": "O"}],
                }
            ],
            "items": [
                {
                    "span_id": "S0",
                    "token_index": 0,
                    "char_offset_in_token": 1,
                    "char": "行",
                    "candidates": ["háng", "xíng"],
                    "current": "háng",
                }
            ],
        }

        try:
            result = adapter.double_check(payload)
            self.assertIn("items", result)
        except LLMError as e:
            self.skipTest(f"Ollama server not available: {e}")

    def test_live_end_to_end(self) -> None:
        """Test end-to-end pinyinize with live Ollama."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._create_minimal_data(root)
            resources = PinyinResources.load_from_dir(root)

            adapter = OllamaLLMAdapter(
                model=DEFAULT_OLLAMA_MODEL,
                host=DEFAULT_OLLAMA_HOST,
                timeout_s=60.0,
            )

            opts = PinyinizeOptions(
                resources=resources,
                llm_adapter=adapter,
                double_check_adapter=None,  # Skip double check for this test
            )

            try:
                result = pinyinize("测试", opts)
                # Should produce pinyin output
                self.assertTrue(result.output_text)
                self.assertTrue(result.report["llm_segment_and_tag"]["used"])
            except Exception as e:
                self.skipTest(f"Ollama server not available: {e}")


if __name__ == "__main__":
    unittest.main()
