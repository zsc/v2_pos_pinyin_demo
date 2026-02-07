from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import is_han, normalize_pinyin


def _load_word_pinyin_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s == "[" or s == "]":
                continue
            if s.endswith(","):
                s = s[:-1]
            obj = json.loads(s)
            word = obj.get("word")
            pinyin = obj.get("pinyin")
            if not isinstance(word, str) or not isinstance(pinyin, str):
                continue
            if not word:
                continue
            if not all(is_han(ch) for ch in word):
                continue
            out[word] = normalize_pinyin(pinyin)
    return out


def _load_char_base(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.endswith(","):
                s = s[:-1]
            obj = json.loads(s)
            ch = obj.get("char")
            pinyin = obj.get("pinyin")
            if isinstance(ch, str) and isinstance(pinyin, list) and all(
                isinstance(x, str) for x in pinyin
            ):
                out[ch] = [normalize_pinyin(x) for x in pinyin]
    return out


def _load_polyphone_candidates(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return out
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        ch = obj.get("char")
        pinyin = obj.get("pinyin")
        if isinstance(ch, str) and isinstance(pinyin, list) and all(
            isinstance(x, str) for x in pinyin
        ):
            out[ch] = [normalize_pinyin(x) for x in pinyin]
    return out


def _load_polyphone_disambig(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("items", [])
    by_char: dict[str, Any] = {}
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("char"), str):
                by_char[it["char"]] = it
    thresholds = raw.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        thresholds = {}
    return by_char, thresholds


def _load_overrides_rules(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        path.write_text(
            json.dumps({"schema_version": 1, "rules": []}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    rules = raw.get("rules", [])
    return rules if isinstance(rules, list) else []


def _load_lexicon(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "items" in raw:
        items = raw.get("items")
        out: dict[str, str] = {}
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                w = it.get("word")
                p = it.get("pinyin")
                if isinstance(w, str) and isinstance(p, str) and w and all(
                    is_han(ch) for ch in w
                ):
                    out[w] = normalize_pinyin(p)
        return out
    if isinstance(raw, dict):
        out2: dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, str) and k and all(
                is_han(ch) for ch in k
            ):
                out2[k] = normalize_pinyin(v)
        return out2
    return {}


@dataclass(frozen=True)
class PinyinResources:
    word_pinyin: dict[str, str]
    lexicon_pinyin: dict[str, str]
    char_base: dict[str, list[str]]
    polyphone_candidates: dict[str, list[str]]
    polyphone_disambig: dict[str, Any]
    disambig_thresholds: dict[str, Any]
    overrides_rules: list[dict[str, Any]]

    @staticmethod
    def load_from_dir(
        data_dir: str | Path,
        *,
        word_json: str = "word.json",
        char_base_json: str = "char_base.json",
        polyphone_json: str = "polyphone.json",
        polyphone_disambig_json: str = "polyphone_disambig.json",
        overrides_json: str = "overrides.json",
        lexicon_json: str = "lexicon.json",
    ) -> "PinyinResources":
        base = Path(data_dir)
        word_path = base / word_json
        char_base_path = base / char_base_json
        polyphone_path = base / polyphone_json
        disambig_path = base / polyphone_disambig_json
        overrides_path = base / overrides_json
        lexicon_path = base / lexicon_json

        word_pinyin = _load_word_pinyin_map(word_path)
        lexicon_pinyin = _load_lexicon(lexicon_path)
        char_base = _load_char_base(char_base_path)
        polyphone_candidates = _load_polyphone_candidates(polyphone_path)
        polyphone_disambig, thresholds = _load_polyphone_disambig(disambig_path)
        overrides_rules = _load_overrides_rules(overrides_path)

        return PinyinResources(
            word_pinyin=word_pinyin,
            lexicon_pinyin=lexicon_pinyin,
            char_base=char_base,
            polyphone_candidates=polyphone_candidates,
            polyphone_disambig=polyphone_disambig,
            disambig_thresholds=thresholds,
            overrides_rules=overrides_rules,
        )

    def combined_word_pinyin(self) -> dict[str, str]:
        merged = dict(self.word_pinyin)
        merged.update(self.lexicon_pinyin)
        return merged
