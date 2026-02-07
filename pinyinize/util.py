from __future__ import annotations

import unicodedata


def is_han(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    # CJK Unified Ideographs + extensions + compatibility ideographs.
    return (
        (0x3400 <= cp <= 0x4DBF)
        or (0x4E00 <= cp <= 0x9FFF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0x2CEB0 <= cp <= 0x2EBEF)
    )


def is_space(ch: str) -> bool:
    return ch.isspace()


def is_ascii_letter(ch: str) -> bool:
    o = ord(ch)
    return (65 <= o <= 90) or (97 <= o <= 122)


def is_ascii_digit(ch: str) -> bool:
    o = ord(ch)
    return 48 <= o <= 57


def is_punct_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def normalize_word_pinyin(pinyin: str) -> str:
    # Spec: remove syllable separator spaces; keep tone marks.
    return normalize_pinyin(pinyin.replace(" ", ""))


def normalize_pinyin(pinyin: str) -> str:
    # Normalize IPA "ɡ" (U+0261) used in some datasets to ASCII "g".
    # Also normalize ü in case upstream uses v.
    return pinyin.replace("ɡ", "g").replace("v", "ü").replace("V", "Ü")


def is_word_like_protected_kind(kind: str | None) -> bool:
    return kind in {"url", "latin", "number"}
