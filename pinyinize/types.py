from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


SpanType = Literal["han", "protected"]
ProtectedKind = Literal[
    "url",
    "latin",
    "number",
    "space",
    "punct",
    "symbol",
    "other",
]


@dataclass(frozen=True)
class Span:
    span_id: str
    type: SpanType
    start: int
    end: int
    text: str
    kind: ProtectedKind | None = None


@dataclass(frozen=True)
class Token:
    span_id: str
    index_in_span: int
    start: int
    end: int
    text: str
    upos: str = "X"
    xpos: str = "UNK"
    ner: str = "O"


@dataclass
class CharDecision:
    char: str
    offset_in_token: int
    candidates: list[str]
    chosen: str
    resolved_by: Literal[
        "word",
        "char_base",
        "polyphone_disambig",
        "override",
        "llm_double_check",
        "user",
        "fallback",
        "unknown",
    ]
    confidence: float | None = None
    rule_id: str | None = None
    needs_review: bool = False
    conflict: bool = False
    notes: list[str] = field(default_factory=list)


class RuleMatchPart(TypedDict, total=False):
    text: str
    text_in: list[str]
    regex: str
    upos_in: list[str]
    xpos_in: list[str]
    ner_in: list[str]
    contains: list[str]


class RuleMatch(TypedDict, total=False):
    self: RuleMatchPart
    prev: RuleMatchPart
    next: RuleMatchPart


class RuleTarget(TypedDict):
    char: str
    occurrence: int | Literal["all"]


class Rule(TypedDict, total=False):
    id: str
    priority: int
    description: str
    match: RuleMatch
    target: RuleTarget
    choose: str
    confidence: float
    meta: dict[str, Any]
