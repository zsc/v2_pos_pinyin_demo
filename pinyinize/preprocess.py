from __future__ import annotations

import re

from .types import ProtectedKind, Span
from .util import is_ascii_digit, is_ascii_letter, is_han, is_punct_or_symbol, is_space


_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def split_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    i = 0
    span_idx = 0
    n = len(text)

    def push_span(
        span_type: str,
        start: int,
        end: int,
        kind: ProtectedKind | None = None,
    ) -> None:
        nonlocal span_idx
        if start >= end:
            return
        spans.append(
            Span(
                span_id=f"S{span_idx}",
                type=span_type,  # type: ignore[arg-type]
                kind=kind,
                start=start,
                end=end,
                text=text[start:end],
            )
        )
        span_idx += 1

    while i < n:
        m = _URL_RE.match(text, i)
        if m:
            push_span("protected", i, m.end(), kind="url")
            i = m.end()
            continue

        ch = text[i]
        if is_han(ch):
            j = i + 1
            while j < n and is_han(text[j]):
                j += 1
            push_span("han", i, j)
            i = j
            continue

        if is_space(ch):
            j = i + 1
            while j < n and is_space(text[j]):
                j += 1
            push_span("protected", i, j, kind="space")
            i = j
            continue

        if is_ascii_letter(ch):
            j = i + 1
            while j < n:
                cj = text[j]
                if is_ascii_letter(cj) or is_ascii_digit(cj) or cj in {"_", "-"}:
                    j += 1
                    continue
                break
            push_span("protected", i, j, kind="latin")
            i = j
            continue

        if is_ascii_digit(ch):
            j = i + 1
            while j < n:
                cj = text[j]
                if is_ascii_digit(cj) or cj in {".", "%"}:
                    j += 1
                    continue
                break
            push_span("protected", i, j, kind="number")
            i = j
            continue

        kind: ProtectedKind = "punct" if is_punct_or_symbol(ch) else "other"
        push_span("protected", i, i + 1, kind=kind)
        i += 1

    return spans

