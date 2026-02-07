from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .preprocess import split_spans
from .resources import PinyinResources
from .rules import AppliedRule, rule_matches, sort_rules
from .types import CharDecision, Rule, Span, Token
from .util import is_word_like_protected_kind, normalize_pinyin, normalize_word_pinyin


_ALLOWED_UPOS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
}

_ALLOWED_NER = {"O", "PER", "LOC", "ORG", "MISC"}


@dataclass(frozen=True)
class PinyinizeOptions:
    resources: PinyinResources
    interactive: bool = False
    # If True, add spaces between han pinyin and adjacent protected word-like spans.
    word_like_spacing: bool = True
    llm_adapter: Any | None = None
    double_check_adapter: Any | None = None
    double_check_threshold: float = 0.85
    debug: bool = False


@dataclass(frozen=True)
class PinyinizeResult:
    output_text: str
    report: dict[str, Any]


def _build_max_len_by_first_char(words: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for w in words.keys():
        if not w:
            continue
        fc = w[0]
        out[fc] = max(out.get(fc, 0), len(w))
    return out


def _segment_fmm(text: str, words: dict[str, str], max_len_by_fc: dict[str, int]) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        fc = text[i]
        max_len = max_len_by_fc.get(fc, 1)
        matched = None
        matched_len = 0
        for L in range(min(max_len, n - i), 0, -1):
            cand = text[i : i + L]
            if cand in words:
                matched = cand
                matched_len = L
                break
        if matched is None:
            tokens.append(text[i])
            i += 1
        else:
            tokens.append(matched)
            i += matched_len
    return tokens


def _tokens_from_spans_fallback(spans: list[Span], word_pinyin: dict[str, str]) -> list[Token]:
    max_len_by_fc = _build_max_len_by_first_char(word_pinyin)
    tokens: list[Token] = []
    for sp in spans:
        if sp.type != "han":
            continue
        seg = _segment_fmm(sp.text, word_pinyin, max_len_by_fc)
        cursor = sp.start
        for idx, t in enumerate(seg):
            start = cursor
            end = cursor + len(t)
            tokens.append(
                Token(
                    span_id=sp.span_id,
                    index_in_span=idx,
                    start=start,
                    end=end,
                    text=t,
                    upos="X",
                    xpos="UNK",
                    ner="O",
                )
            )
            cursor = end
    return tokens


def _tokens_from_spans_llm_or_fallback(
    spans: list[Span],
    word_pinyin: dict[str, str],
    llm_adapter: Any | None,
) -> tuple[list[Token], dict[str, Any]]:
    """
    Returns (tokens, llm_meta).
    llm_meta contains request/response/error/invalid span ids for reporting.
    """
    han_spans = [sp for sp in spans if sp.type == "han"]
    if not llm_adapter or not han_spans:
        return _tokens_from_spans_fallback(spans, word_pinyin), {"used": False}

    request = {
        "schema_version": 1,
        "task": "segment_and_tag",
        "tagset": {"upos": "UDv2", "xpos": "CTB", "ner": "CoNLL"},
        "spans": [{"span_id": sp.span_id, "text": sp.text} for sp in han_spans],
    }
    meta: dict[str, Any] = {"used": True, "request": request}

    try:
        segment_fn = getattr(llm_adapter, "segment_and_tag", None)
        if not callable(segment_fn):
            meta["error"] = "llm_adapter_missing_segment_and_tag"
            return _tokens_from_spans_fallback(spans, word_pinyin), meta
        response = segment_fn(request)
    except Exception as e:  # noqa: BLE001
        meta["error"] = f"llm_segment_and_tag_exception:{e}"
        return _tokens_from_spans_fallback(spans, word_pinyin), meta

    meta["response"] = response
    if not isinstance(response, dict):
        meta["error"] = "llm_response_not_object"
        return _tokens_from_spans_fallback(spans, word_pinyin), meta

    resp_spans = response.get("spans")
    if not isinstance(resp_spans, list):
        meta["error"] = "llm_response_missing_spans"
        return _tokens_from_spans_fallback(spans, word_pinyin), meta

    # span_id -> list[token dict]
    by_span_id: dict[str, list[dict[str, Any]]] = {}
    for s in resp_spans:
        if not isinstance(s, dict):
            continue
        sid = s.get("span_id")
        toks = s.get("tokens")
        if isinstance(sid, str) and isinstance(toks, list):
            by_span_id[sid] = [t for t in toks if isinstance(t, dict)]

    max_len_by_fc = _build_max_len_by_first_char(word_pinyin)
    invalid_spans: list[str] = []
    tokens: list[Token] = []

    def fallback_for_span(sp: Span) -> None:
        seg = _segment_fmm(sp.text, word_pinyin, max_len_by_fc)
        cursor = sp.start
        for idx, t in enumerate(seg):
            start = cursor
            end = cursor + len(t)
            tokens.append(
                Token(
                    span_id=sp.span_id,
                    index_in_span=idx,
                    start=start,
                    end=end,
                    text=t,
                    upos="X",
                    xpos="UNK",
                    ner="O",
                )
            )
            cursor = end

    for sp in spans:
        if sp.type != "han":
            continue
        llm_toks = by_span_id.get(sp.span_id)
        if not llm_toks:
            invalid_spans.append(sp.span_id)
            fallback_for_span(sp)
            continue

        texts: list[str] = []
        ok = True
        for t in llm_toks:
            tt = t.get("text")
            upos = t.get("upos")
            xpos = t.get("xpos")
            ner = t.get("ner")
            if not isinstance(tt, str) or not tt:
                ok = False
                break
            if not isinstance(upos, str) or upos not in _ALLOWED_UPOS:
                ok = False
                break
            if not isinstance(xpos, str) or not xpos:
                ok = False
                break
            if not isinstance(ner, str) or ner not in _ALLOWED_NER:
                ok = False
                break
            texts.append(tt)

        if not ok or "".join(texts) != sp.text:
            invalid_spans.append(sp.span_id)
            fallback_for_span(sp)
            continue

        cursor = sp.start
        for idx, t in enumerate(llm_toks):
            tt = t["text"]
            start = cursor
            end = cursor + len(tt)
            tokens.append(
                Token(
                    span_id=sp.span_id,
                    index_in_span=idx,
                    start=start,
                    end=end,
                    text=tt,
                    upos=t["upos"],
                    xpos=t["xpos"],
                    ner=t["ner"],
                )
            )
            cursor = end

    meta["invalid_spans"] = invalid_spans
    meta["warnings"] = response.get("warnings", [])
    return tokens, meta


def _char_candidates(char_base: dict[str, list[str]], ch: str) -> list[str]:
    cands = char_base.get(ch)
    return list(cands) if cands else []


def _polyphone_pick(
    disambig: dict[str, Any],
    thresholds: dict[str, Any],
    ch: str,
    *,
    upos: str,
    ner: str,
) -> tuple[str, float | None, bool]:
    """
    Returns (chosen, confidence, confident_enough).
    """
    item = disambig.get(ch)
    if not isinstance(item, dict):
        return "", None, False

    default = item.get("default")
    candidates = item.get("candidates")
    if not isinstance(candidates, list) or not all(isinstance(x, str) for x in candidates):
        candidates = []

    ctxs = item.get("contexts") or {}
    if not isinstance(ctxs, dict):
        ctxs = {}
    key = f"pos={upos}|ner={ner}"
    ctx = ctxs.get(key)
    if not isinstance(ctx, dict):
        if isinstance(default, str) and default:
            return default, None, False
        if candidates:
            return candidates[0], None, False
        return "", None, False

    best = ctx.get("best")
    p = ctx.get("p")
    p2 = ctx.get("p2")
    n = ctx.get("n")
    if not isinstance(best, str) or not best:
        if isinstance(default, str) and default:
            return default, None, False
        if candidates:
            return candidates[0], None, False
        return "", None, False

    min_support = thresholds.get("min_support", 5)
    min_prob = thresholds.get("min_prob", 0.85)
    min_margin = thresholds.get("min_margin", 0.15)
    try:
        n_int = int(n) if n is not None else 0
        p_f = float(p) if p is not None else 0.0
        p2_f = float(p2) if p2 is not None else 0.0
    except (TypeError, ValueError):
        n_int = 0
        p_f = 0.0
        p2_f = 0.0

    confident = (
        n_int >= int(min_support)
        and p_f >= float(min_prob)
        and (p_f - p2_f) >= float(min_margin)
    )
    return best, p_f if p is not None else None, confident


def _analyze_token(
    tok: Token,
    word_pinyin: dict[str, str],
    resources: PinyinResources,
) -> tuple[str, list[CharDecision], list[str]]:
    warnings: list[str] = []
    word_entry = word_pinyin.get(tok.text)
    if isinstance(word_entry, str):
        syllables = [s for s in word_entry.split() if s]
        if len(syllables) == len(tok.text):
            decisions: list[CharDecision] = []
            for i, ch in enumerate(tok.text):
                decisions.append(
                    CharDecision(
                        char=ch,
                        offset_in_token=i,
                        candidates=[syllables[i]],
                        chosen=syllables[i],
                        resolved_by="word",
                        confidence=1.0,
                    )
                )
            token_pinyin = normalize_word_pinyin(word_entry)
            return token_pinyin, decisions, warnings
        warnings.append(
            f"word_pinyin_alignment_mismatch: token='{tok.text}' syllables={len(syllables)} chars={len(tok.text)}"
        )

    decisions2: list[CharDecision] = []
    out_parts: list[str] = []
    for i, ch in enumerate(tok.text):
        cands = _char_candidates(resources.char_base, ch)
        if not cands:
            decisions2.append(
                CharDecision(
                    char=ch,
                    offset_in_token=i,
                    candidates=[],
                    chosen=ch,
                    resolved_by="unknown",
                    confidence=0.0,
                    needs_review=True,
                    notes=["char_not_in_char_base"],
                )
            )
            out_parts.append(ch)
            continue

        if len(cands) == 1:
            decisions2.append(
                CharDecision(
                    char=ch,
                    offset_in_token=i,
                    candidates=cands,
                    chosen=cands[0],
                    resolved_by="char_base",
                    confidence=1.0,
                )
            )
            out_parts.append(cands[0])
            continue

        best, conf, confident = _polyphone_pick(
            resources.polyphone_disambig,
            resources.disambig_thresholds,
            ch,
            upos=tok.upos,
            ner=tok.ner,
        )
        chosen = best if best else (cands[0] if cands else "")
        resolved_by: Literal["polyphone_disambig", "fallback"] = (
            "polyphone_disambig" if best else "fallback"
        )
        dec = CharDecision(
            char=ch,
            offset_in_token=i,
            candidates=cands,
            chosen=chosen,
            resolved_by=resolved_by,
            confidence=conf,
            needs_review=not confident,
        )
        if not confident:
            dec.notes.append("low_confidence_or_low_support")
        decisions2.append(dec)
        out_parts.append(chosen)

    return "".join(out_parts), decisions2, warnings


def _apply_overrides(
    tokens: list[Token],
    token_decisions: dict[tuple[int, int], list[CharDecision]],
    overrides_rules: list[dict[str, Any]],
) -> tuple[list[AppliedRule], list[dict[str, Any]]]:
    applied: list[AppliedRule] = []
    conflicts: list[dict[str, Any]] = []

    parsed_rules: list[Rule] = []
    for r in overrides_rules:
        if isinstance(r, dict) and isinstance(r.get("id"), str):
            parsed_rules.append(r)  # type: ignore[arg-type]

    # Only consider prev/next within the same han span.
    span_to_tokens: dict[str, list[Token]] = {}
    for tok in tokens:
        span_to_tokens.setdefault(tok.span_id, []).append(tok)

    for rule in sort_rules(parsed_rules):
        target = rule.get("target")
        choose = rule.get("choose")
        rid = rule.get("id")
        if not isinstance(target, dict) or not isinstance(choose, str) or not isinstance(rid, str):
            continue
        choose = normalize_pinyin(choose)
        target_char = target.get("char")
        occurrence = target.get("occurrence")
        if not isinstance(target_char, str) or not target_char:
            continue

        for span_tokens in span_to_tokens.values():
            for i, tok in enumerate(span_tokens):
                if target_char not in tok.text:
                    continue
                prev_tok = span_tokens[i - 1] if i > 0 else None
                next_tok = span_tokens[i + 1] if i + 1 < len(span_tokens) else None
                if not rule_matches(rule, tok, prev_tok, next_tok):
                    continue

                key = (tok.start, tok.end)
                decisions = token_decisions.get(key)
                if not decisions:
                    continue

                positions = [d.offset_in_token for d in decisions if d.char == target_char]
                if not positions:
                    continue

                def apply_at(pos: int) -> None:
                    dec = decisions[pos]
                    if dec.char != target_char:
                        return
                    if dec.chosen == choose:
                        dec.notes.append(f"override_reaffirm:{rid}")
                        # Still mark as override to prevent lower-priority rules from changing it
                        dec.resolved_by = "override"
                        dec.rule_id = rid
                        return
                    if dec.resolved_by == "override" and dec.rule_id and dec.rule_id != rid:
                        dec.conflict = True
                        conflicts.append(
                            {
                                "type": "override_conflict",
                                "token": tok.text,
                                "token_start": tok.start,
                                "token_end": tok.end,
                                "char": target_char,
                                "offset_in_token": pos,
                                "existing_rule_id": dec.rule_id,
                                "existing_choose": dec.chosen,
                                "new_rule_id": rid,
                                "new_choose": choose,
                            }
                        )
                        return

                    dec.chosen = choose
                    dec.resolved_by = "override"
                    dec.rule_id = rid
                    dec.needs_review = False
                    applied.append(
                        AppliedRule(
                            rule_id=rid,
                            token_start=tok.start,
                            token_end=tok.end,
                            token_text=tok.text,
                            target_char=target_char,
                            occurrence=str(occurrence),
                            choose=choose,
                        )
                    )

                if occurrence == "all":
                    for pos in positions:
                        apply_at(pos)
                    continue

                if isinstance(occurrence, int) and occurrence >= 1:
                    if occurrence <= len(positions):
                        apply_at(positions[occurrence - 1])
                    continue

    return applied, conflicts


def _collect_review_items(
    tokens: list[Token],
    token_decisions: dict[tuple[int, int], list[CharDecision]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for tok in tokens:
        decisions = token_decisions.get((tok.start, tok.end)) or []
        for d in decisions:
            low_conf = d.confidence is not None and d.confidence < threshold
            if d.needs_review or d.conflict or low_conf:
                items.append(
                    {
                        "span_id": tok.span_id,
                        "token_index": tok.index_in_span,
                        "token_text": tok.text,
                        "token_start": tok.start,
                        "token_end": tok.end,
                        "char_offset_in_token": d.offset_in_token,
                        "char": d.char,
                        "candidates": d.candidates,
                        "chosen": d.chosen,
                        "confidence": d.confidence,
                        "needs_review": d.needs_review,
                        "conflict": d.conflict,
                    }
                )
    return items


def _apply_llm_double_check(
    adapter: Any,
    *,
    text: str,
    spans: list[Span],
    tokens: list[Token],
    token_decisions: dict[tuple[int, int], list[CharDecision]],
    review_items: list[dict[str, Any]],
) -> dict[str, Any]:
    meta: dict[str, Any] = {"used": False}
    if not adapter or not review_items:
        return meta

    double_check_fn = getattr(adapter, "double_check", None)
    if not callable(double_check_fn):
        meta["error"] = "llm_adapter_missing_double_check"
        return meta

    # Group tokens by span to keep schema close to spec.
    span_to_tokens: dict[str, list[Token]] = {}
    for tok in tokens:
        span_to_tokens.setdefault(tok.span_id, []).append(tok)

    payload = {
        "schema_version": 1,
        "task": "double_check",
        "text": text,
        "spans": [
            {
                "span_id": sp.span_id,
                "text": sp.text,
                "tokens": [
                    {"text": t.text, "upos": t.upos, "xpos": t.xpos, "ner": t.ner}
                    for t in (span_to_tokens.get(sp.span_id) or [])
                ],
            }
            for sp in spans
            if sp.type == "han"
        ],
        "items": [
            {
                "span_id": it["span_id"],
                "token_index": it["token_index"],
                "char_offset_in_token": it["char_offset_in_token"],
                "char": it["char"],
                "candidates": it["candidates"],
                "current": it["chosen"],
            }
            for it in review_items
        ],
    }
    meta["used"] = True
    meta["request"] = payload

    try:
        response = double_check_fn(payload)
    except Exception as e:  # noqa: BLE001
        meta["error"] = f"llm_double_check_exception:{e}"
        return meta

    meta["response"] = response
    if not isinstance(response, dict):
        meta["error"] = "llm_double_check_response_not_object"
        return meta

    resp_items = response.get("items")
    if not isinstance(resp_items, list):
        meta["error"] = "llm_double_check_missing_items"
        return meta

    # Build a quick index for tokens.
    tok_by_span_and_index: dict[tuple[str, int], Token] = {}
    for tok in tokens:
        tok_by_span_and_index[(tok.span_id, tok.index_in_span)] = tok

    applied: list[dict[str, Any]] = []
    needs_user: list[dict[str, Any]] = []
    warnings: list[str] = []

    for it in resp_items:
        if not isinstance(it, dict):
            continue
        sid = it.get("span_id")
        token_index = it.get("token_index")
        char_offset = it.get("char_offset_in_token")
        char = it.get("char")
        recommended = it.get("recommended")
        reason = it.get("reason")
        needs_user_flag = it.get("needs_user", False)

        if not isinstance(sid, str) or not isinstance(token_index, int) or not isinstance(char_offset, int):
            continue
        tok = tok_by_span_and_index.get((sid, token_index))
        if not tok:
            warnings.append(f"double_check_item_token_not_found:{sid}:{token_index}")
            continue
        decisions = token_decisions.get((tok.start, tok.end)) or []
        if not (0 <= char_offset < len(decisions)):
            warnings.append(f"double_check_item_char_offset_oob:{sid}:{token_index}:{char_offset}")
            continue
        dec = decisions[char_offset]
        if isinstance(char, str) and char and dec.char != char:
            warnings.append(
                f"double_check_item_char_mismatch:{sid}:{token_index}:{char_offset}:expected={dec.char}:got={char}"
            )

        if bool(needs_user_flag):
            dec.needs_review = True
            dec.notes.append("llm_double_check_needs_user")
            needs_user.append(
                {
                    "span_id": sid,
                    "token_index": token_index,
                    "char_offset_in_token": char_offset,
                    "char": dec.char,
                    "candidates": dec.candidates,
                    "recommended": normalize_pinyin(recommended) if isinstance(recommended, str) else recommended,
                    "reason": reason,
                }
            )
            continue

        if isinstance(recommended, str) and recommended:
            recommended_norm = normalize_pinyin(recommended)
            dec.chosen = recommended_norm
            dec.resolved_by = "llm_double_check"
            dec.needs_review = False
            if isinstance(reason, str) and reason:
                dec.notes.append(f"llm_reason:{reason}")
            applied.append(
                {
                    "span_id": sid,
                    "token_index": token_index,
                    "char_offset_in_token": char_offset,
                    "char": dec.char,
                    "recommended": recommended_norm,
                    "reason": reason,
                }
            )

    meta["applied"] = applied
    meta["needs_user"] = needs_user
    meta["warnings"] = warnings
    return meta


def pinyinize(text: str, options: PinyinizeOptions) -> PinyinizeResult:
    import sys
    
    def _debug_step(step_name: str, data: Any) -> None:
        if options.debug:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[DEBUG] {step_name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            if isinstance(data, (list, dict)):
                print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
            else:
                print(data, file=sys.stderr)
    
    spans = split_spans(text)
    _debug_step("Step 1: Preprocessing - Split spans", [
        {"span_id": sp.span_id, "type": sp.type, "kind": sp.kind, 
         "start": sp.start, "end": sp.end, "text": sp.text}
        for sp in spans
    ])
    
    resources = options.resources
    combined_word_pinyin = resources.combined_word_pinyin()

    tokens, llm_meta = _tokens_from_spans_llm_or_fallback(
        spans, combined_word_pinyin, options.llm_adapter
    )
    _debug_step("Step 2: Tokenization (LLM or Fallback)", {
        "llm_used": llm_meta.get("used", False),
        "llm_error": llm_meta.get("error"),
        "invalid_spans": llm_meta.get("invalid_spans", []),
        "tokens": [
            {"span_id": tok.span_id, "index_in_span": tok.index_in_span,
             "start": tok.start, "end": tok.end, "text": tok.text,
             "upos": tok.upos, "xpos": tok.xpos, "ner": tok.ner}
            for tok in tokens
        ]
    })

    token_pinyin: dict[tuple[int, int], str] = {}
    token_decisions: dict[tuple[int, int], list[CharDecision]] = {}
    warnings: list[str] = []

    debug_token_analysis: list[dict[str, Any]] = []
    for tok in tokens:
        py, decisions, w = _analyze_token(tok, combined_word_pinyin, resources)
        token_pinyin[(tok.start, tok.end)] = py
        token_decisions[(tok.start, tok.end)] = decisions
        warnings.extend(w)
        
        if options.debug:
            debug_token_analysis.append({
                "token_text": tok.text,
                "start": tok.start,
                "end": tok.end,
                "upos": tok.upos,
                "ner": tok.ner,
                "final_pinyin": py,
                "char_decisions": [
                    {
                        "char": d.char,
                        "offset": d.offset_in_token,
                        "candidates": d.candidates,
                        "chosen": d.chosen,
                        "resolved_by": d.resolved_by,
                        "confidence": d.confidence,
                        "needs_review": d.needs_review,
                    }
                    for d in decisions
                ]
            })
    
    _debug_step("Step 3: Token Analysis (Word/CharBase/Polyphone lookup)", debug_token_analysis)

    applied_rules, conflicts = _apply_overrides(tokens, token_decisions, resources.overrides_rules)
    _debug_step("Step 4: Apply Overrides", {
        "applied_rules": [
            {"rule_id": r.rule_id, "token": r.token_text, "char": r.target_char, "choose": r.choose}
            for r in applied_rules
        ],
        "conflicts": conflicts
    })

    # LLM double check (optional)
    review_items_before = _collect_review_items(
        tokens, token_decisions, threshold=options.double_check_threshold
    )
    _debug_step("Step 5: Items needing review before double-check", review_items_before)
    
    double_check_meta = _apply_llm_double_check(
        options.double_check_adapter,
        text=text,
        spans=spans,
        tokens=tokens,
        token_decisions=token_decisions,
        review_items=review_items_before,
    )
    _debug_step("Step 6: LLM Double Check", {
        "used": double_check_meta.get("used"),
        "error": double_check_meta.get("error"),
        "applied": double_check_meta.get("applied", []),
        "needs_user": double_check_meta.get("needs_user", []),
        "warnings": double_check_meta.get("warnings", []),
    })

    # Rebuild token pinyin after overrides.
    for tok in tokens:
        key = (tok.start, tok.end)
        decisions = token_decisions.get(key) or []
        token_pinyin[key] = "".join(d.chosen for d in decisions)

    # Output stitching.
    out_parts: list[str] = []
    han_span_to_tokens: dict[str, list[Token]] = {}
    for tok in tokens:
        han_span_to_tokens.setdefault(tok.span_id, []).append(tok)

    def han_span_output(span_id: str) -> str:
        toks = han_span_to_tokens.get(span_id) or []
        return " ".join(token_pinyin[(t.start, t.end)] for t in toks)

    prev_kind: str | None = None
    prev_was_han = False

    for sp in spans:
        if sp.type == "han":
            han_out = han_span_output(sp.span_id)
            if options.word_like_spacing and out_parts:
                if not prev_was_han and is_word_like_protected_kind(prev_kind):
                    if not out_parts[-1].endswith((" ", "\t", "\n")):
                        out_parts.append(" ")
            out_parts.append(han_out)
            prev_kind = None
            prev_was_han = True
            continue

        if options.word_like_spacing and out_parts:
            if prev_was_han and is_word_like_protected_kind(sp.kind):
                if not out_parts[-1].endswith((" ", "\t", "\n")) and not sp.text.startswith(
                    (" ", "\t", "\n")
                ):
                    out_parts.append(" ")
        out_parts.append(sp.text)
        prev_kind = sp.kind
        prev_was_han = False

    output_text = "".join(out_parts)
    _debug_step("Step 7: Output Stitching", {
        "output_parts": out_parts,
        "final_output": output_text
    })

    review_items_after = _collect_review_items(
        tokens, token_decisions, threshold=options.double_check_threshold
    )

    report_tokens: list[dict[str, Any]] = []
    for tok in tokens:
        key = (tok.start, tok.end)
        report_tokens.append(
            {
                "span_id": tok.span_id,
                "index_in_span": tok.index_in_span,
                "start": tok.start,
                "end": tok.end,
                "text": tok.text,
                "upos": tok.upos,
                "xpos": tok.xpos,
                "ner": tok.ner,
                "pinyin": token_pinyin.get(key, ""),
                "char_decisions": [
                    {
                        "char": d.char,
                        "offset_in_token": d.offset_in_token,
                        "candidates": d.candidates,
                        "chosen": d.chosen,
                        "resolved_by": d.resolved_by,
                        "confidence": d.confidence,
                        "rule_id": d.rule_id,
                        "needs_review": d.needs_review,
                        "conflict": d.conflict,
                        "notes": d.notes,
                    }
                    for d in (token_decisions.get(key) or [])
                ],
            }
        )

    report = {
        "schema_version": 1,
        "text": text,
        "spans": [
            {
                "span_id": sp.span_id,
                "type": sp.type,
                "kind": sp.kind,
                "start": sp.start,
                "end": sp.end,
                "text": sp.text,
            }
            for sp in spans
        ],
        "tokens": report_tokens,
        "llm_segment_and_tag": llm_meta,
        "llm_double_check": double_check_meta,
        "needs_review_items": review_items_after,
        "unresolved_fallback": bool(review_items_after) and not (
            double_check_meta.get("used") and not double_check_meta.get("error")
        ),
        "applied_overrides": [
            {
                "rule_id": a.rule_id,
                "token_text": a.token_text,
                "token_start": a.token_start,
                "token_end": a.token_end,
                "target_char": a.target_char,
                "occurrence": a.occurrence,
                "choose": a.choose,
            }
            for a in applied_rules
        ],
        "conflicts": conflicts,
        "warnings": warnings,
    }

    return PinyinizeResult(output_text=output_text, report=report)
