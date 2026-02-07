"""Main test suite for pinyinize - combines all acceptance criteria tests."""

import json
import tempfile
import unittest
from pathlib import Path

from pinyinize.core import PinyinizeOptions, pinyinize
from pinyinize.resources import PinyinResources


class TestPinyinize(unittest.TestCase):
    """Original acceptance criteria tests."""

    def _write_min_data(self, root: Path) -> None:
        word_items = [
            {"word": "细说", "pinyin": "xì shuō"},
            {"word": "银行", "pinyin": "yín háng"},
            {"word": "行长", "pinyin": "háng zhǎng"},
            {"word": "重新", "pinyin": "chóng xīn"},
            {"word": "营业", "pinyin": "yíng yè"},
            {"word": "得到", "pinyin": "dé dào"},
            {"word": "答案", "pinyin": "dá àn"},
            {"word": "得去", "pinyin": "děi qù"},
        ]
        root.joinpath("word.json").write_text(
            "[\n"
            + "\n".join(json.dumps(it, ensure_ascii=False) + "," for it in word_items[:-1])
            + "\n"
            + json.dumps(word_items[-1], ensure_ascii=False)
            + "\n]\n",
            encoding="utf-8",
        )

        char_items = [
            {"index": 1, "char": "他", "pinyin": ["tā"]},
            {"index": 2, "char": "的", "pinyin": ["de"]},
        ]
        root.joinpath("char_base.json").write_text(
            "\n".join(json.dumps(it, ensure_ascii=False) + "," for it in char_items) + "\n",
            encoding="utf-8",
        )

        root.joinpath("polyphone.json").write_text("[]\n", encoding="utf-8")
        root.joinpath("polyphone_disambig.json").write_text(
            json.dumps(
                {
                    "schema": "test",
                    "thresholds": {"min_support": 5, "min_prob": 0.85, "min_margin": 0.15},
                    "items": [],
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
            json.dumps(
                {
                    "schema_version": 1,
                    "items": [
                        {"word": "行长", "pinyin": "háng zhǎng"},
                        {"word": "得去", "pinyin": "děi qù"},
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_acceptance_cases(self) -> None:
        """Test all acceptance criteria from CLAUDE.md section 15."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_min_data(root)
            resources = PinyinResources.load_from_dir(root)
            opts = PinyinizeOptions(resources=resources)

            # Criterion 1: 基础
            self.assertEqual(pinyinize("细说", opts).output_text, "xìshuō")

            # Criterion 2: 行/长/重
            self.assertEqual(
                pinyinize("银行行长重新营业", opts).output_text,
                "yínháng hángzhǎng chóngxīn yíngyè",
            )

            # Criterion 3: 得
            self.assertEqual(
                pinyinize("他得去得到答案", opts).output_text,
                "tā děiqù dédào dáàn",
            )

            # Criterion 4: 混排
            self.assertEqual(
                pinyinize("细说OpenAI的API v2.0：https://openai.com", opts).output_text,
                "xìshuō OpenAI de API v2.0：https://openai.com",
            )


if __name__ == "__main__":
    unittest.main()

