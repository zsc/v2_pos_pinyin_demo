from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

from .core import PinyinizeOptions, pinyinize
from .llm import OllamaLLMAdapter
from .resources import PinyinResources


_OVERRIDE_ID_RE = re.compile(r"^override_(\d{4}-\d{2}-\d{2})_(\d{4})$")


def _next_override_id(existing_rules: list[dict]) -> str:
    today = _dt.date.today().isoformat()
    max_serial = 0
    for r in existing_rules:
        rid = r.get("id") if isinstance(r, dict) else None
        if not isinstance(rid, str):
            continue
        m = _OVERRIDE_ID_RE.match(rid)
        if not m:
            continue
        date_s, serial_s = m.group(1), m.group(2)
        if date_s != today:
            continue
        try:
            max_serial = max(max_serial, int(serial_s))
        except ValueError:
            continue
    return f"override_{today}_{max_serial + 1:04d}"


def _load_overrides(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "rules": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_overrides(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prompt_choice(candidates: list[str], default_idx: int = 0) -> int | None:
    if not candidates:
        return None
    prompt = f"Choose 1-{len(candidates)} (default {default_idx + 1}, enter=default, s=skip): "
    while True:
        s = input(prompt).strip().lower()
        if s == "":
            return default_idx
        if s == "s":
            return None
        try:
            v = int(s)
        except ValueError:
            continue
        if 1 <= v <= len(candidates):
            return v - 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pinyinize")
    parser.add_argument("text", nargs="?", help="Input text. If omitted, read from stdin.")
    parser.add_argument("--data-dir", default=".", help="Directory containing *.json data files.")
    parser.add_argument("--report", default=None, help="Write report JSON to this path.")
    parser.add_argument(
        "--no-word-like-spacing",
        action="store_true",
        help="Do not insert spaces around latin/number/url spans.",
    )
    parser.add_argument("--ollama-model", default=None, help="Use Ollama for LLM segment+tag and double-check.")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--no-double-check", action="store_true", help="Disable LLM double-check step.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for unresolved readings and write overrides.json.")
    parser.add_argument("--debug", action="store_true", help="Print intermediate processing steps.")
    args = parser.parse_args(argv)

    text = args.text
    if text is None:
        text = sys.stdin.read()

    data_dir = Path(args.data_dir)
    resources = PinyinResources.load_from_dir(data_dir)

    llm_adapter = None
    double_check_adapter = None
    if args.ollama_model:
        llm_adapter = OllamaLLMAdapter(model=args.ollama_model, host=args.ollama_host)
        if not args.no_double_check:
            double_check_adapter = llm_adapter

    opts = PinyinizeOptions(
        resources=resources,
        word_like_spacing=not args.no_word_like_spacing,
        llm_adapter=llm_adapter,
        double_check_adapter=double_check_adapter,
        debug=args.debug,
    )
    res = pinyinize(text, opts)

    if args.interactive and res.report.get("needs_review_items"):
        overrides_path = data_dir / "overrides.json"
        overrides = _load_overrides(overrides_path)
        rules = overrides.get("rules")
        if not isinstance(rules, list):
            rules = []
            overrides["rules"] = rules

        for item in res.report["needs_review_items"]:
            if not isinstance(item, dict):
                continue
            token_text = item.get("token_text")
            token_start = item.get("token_start")
            token_end = item.get("token_end")
            char = item.get("char")
            char_offset = item.get("char_offset_in_token")
            candidates = item.get("candidates") or []
            chosen = item.get("chosen")
            if not (
                isinstance(token_text, str)
                and isinstance(token_start, int)
                and isinstance(token_end, int)
                and isinstance(char, str)
                and isinstance(char_offset, int)
                and isinstance(candidates, list)
            ):
                continue
            if not candidates:
                continue

            left = max(0, token_start - 12)
            right = min(len(text), token_end + 12)
            ctx = text[left:right]
            print("\n---")
            print(f"context: {ctx}")
            print(f"token: '{token_text}' [{token_start},{token_end})")
            print(f"char: '{char}' offset={char_offset}")
            for i, c in enumerate(candidates, start=1):
                mark = "*" if c == chosen else " "
                print(f"  {mark}{i}) {c}")

            default_idx = 0
            if isinstance(chosen, str) and chosen in candidates:
                default_idx = candidates.index(chosen)
            picked = _prompt_choice(candidates, default_idx=default_idx)
            if picked is None:
                continue
            choose = candidates[picked]

            occurrence = token_text[: char_offset + 1].count(char)
            rid = _next_override_id(rules)
            rule = {
                "id": rid,
                "priority": 100000,
                "description": f"user override: {char}({token_text})={choose}",
                "match": {"self": {"text": token_text}},
                "target": {"char": char, "occurrence": occurrence},
                "choose": choose,
                "meta": {"created_at": _dt.date.today().isoformat(), "source": "user", "example": ctx},
            }
            rules.append(rule)
            print(f"wrote override: {rid}")

        _write_overrides(overrides_path, overrides)
        # Reload resources + rerun so output reflects overrides.
        resources2 = PinyinResources.load_from_dir(data_dir)
        opts2 = PinyinizeOptions(
            resources=resources2,
            word_like_spacing=not args.no_word_like_spacing,
            llm_adapter=llm_adapter,
            double_check_adapter=double_check_adapter,
            debug=args.debug,
        )
        res = pinyinize(text, opts2)

    sys.stdout.write(res.output_text)
    if not res.output_text.endswith("\n"):
        sys.stdout.write("\n")

    if args.report:
        Path(args.report).write_text(
            json.dumps(res.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
