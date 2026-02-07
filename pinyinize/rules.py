from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import Rule, Token


@dataclass(frozen=True)
class AppliedRule:
    rule_id: str
    token_start: int
    token_end: int
    token_text: str
    target_char: str
    occurrence: int | str
    choose: str


def _match_part(part: dict[str, Any], tok: Token) -> bool:
    txt = tok.text
    if "text" in part and part["text"] != txt:
        return False
    if "text_in" in part:
        vals = part["text_in"]
        if isinstance(vals, list) and txt not in vals:
            return False
    if "regex" in part:
        rx = part["regex"]
        if isinstance(rx, str) and not re.search(rx, txt):
            return False
    if "upos_in" in part:
        vals = part["upos_in"]
        if isinstance(vals, list) and tok.upos not in vals:
            return False
    if "xpos_in" in part:
        vals = part["xpos_in"]
        if isinstance(vals, list) and tok.xpos not in vals:
            return False
    if "ner_in" in part:
        vals = part["ner_in"]
        if isinstance(vals, list) and tok.ner not in vals:
            return False
    if "contains" in part:
        vals = part["contains"]
        if isinstance(vals, list):
            for ch in vals:
                if isinstance(ch, str) and ch and ch not in txt:
                    return False
    return True


def rule_matches(rule: Rule, tok: Token, prev_tok: Token | None, next_tok: Token | None) -> bool:
    match = rule.get("match") or {}
    if not isinstance(match, dict):
        return False

    self_part = match.get("self")
    if self_part is not None:
        if not isinstance(self_part, dict):
            return False
        if not _match_part(self_part, tok):
            return False

    prev_part = match.get("prev")
    if prev_part is not None:
        if prev_tok is None:
            return False
        if not isinstance(prev_part, dict):
            return False
        if not _match_part(prev_part, prev_tok):
            return False

    next_part = match.get("next")
    if next_part is not None:
        if next_tok is None:
            return False
        if not isinstance(next_part, dict):
            return False
        if not _match_part(next_part, next_tok):
            return False

    return True


def sort_rules(rules: list[Rule]) -> list[Rule]:
    def key(r: Rule) -> tuple[int, str]:
        prio = r.get("priority")
        rid = r.get("id")
        prio_int = int(prio) if isinstance(prio, int) else 0
        rid_str = str(rid) if rid else ""
        # priority DESC, id ASC
        return (-prio_int, rid_str)

    return sorted(rules, key=key)
