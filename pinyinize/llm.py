from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json_object(text: str) -> Any:
    """
    Best-effort extractor for LLM responses that should be strict JSON.
    Returns the decoded JSON value (usually a dict).
    """
    t = text.strip()
    if not t:
        raise LLMError("empty_llm_response")

    # Remove common ```json fences.
    if "```" in t:
        t = _CODE_FENCE_RE.sub("", t).strip()

    # Try direct JSON first.
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    # Try to extract the outermost JSON object.
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = t[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as e:
            raise LLMError(f"invalid_json_snippet:{e}") from e

    raise LLMError("no_json_object_found")


def _http_post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.load(resp)
    except urllib.error.URLError as e:
        raise LLMError(f"ollama_http_error:{e}") from e


@dataclass(frozen=True)
class OllamaLLMAdapter:
    model: str
    host: str = "http://localhost:11434"
    timeout_s: float = 60.0

    def _chat(self, *, system: str, user: str) -> str:
        url = f"{self.host.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        raw = _http_post_json(url, payload, timeout_s=self.timeout_s)
        msg = raw.get("message")
        if not isinstance(msg, dict):
            raise LLMError("ollama_missing_message")
        content = msg.get("content")
        if not isinstance(content, str):
            raise LLMError("ollama_missing_content")
        return content

    def _complete_json(self, *, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        user = json.dumps(payload, ensure_ascii=False, indent=2)
        content = self._chat(system=system, user=user)
        obj = extract_json_object(content)
        if not isinstance(obj, dict):
            raise LLMError("llm_response_not_object")
        return obj

    def segment_and_tag(self, payload: dict[str, Any]) -> dict[str, Any]:
        system = (
            "You are a Chinese NLP tagger.\n"
            "Task: segment each span text into tokens and tag each token with:\n"
            "- upos: UDv2 UPOS tag (ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X)\n"
            "- xpos: CTB tag (string, e.g., NN, VV, AD, etc.)\n"
            "- ner: CoNLL NER tag (O, PER, LOC, ORG, MISC)\n\n"
            "Output format:\n"
            '{\n'
            '  "spans": [\n'
            '    {\n'
            '      "span_id": "S0",\n'
            '      "tokens": [\n'
            '        {"text": "token1", "upos": "VERB", "xpos": "VV", "ner": "O"},\n'
            '        {"text": "token2", "upos": "NOUN", "xpos": "NN", "ner": "O"}\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "Rules:\n"
            "1. You MUST output STRICT JSON only. No extra text.\n"
            "2. For each span: concatenation of token.text MUST equal the original span.text exactly.\n"
            "3. Each token must have text, upos, xpos, and ner fields."
        )
        return self._complete_json(system=system, payload=payload)

    def double_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        system = (
            "You are helping to disambiguate Chinese polyphonic characters.\n"
            "Given input text, spans, tokens (with POS/NER), and a list of review items,\n"
            "return STRICT JSON only with recommended pinyin (tone marks) for each item.\n"
            "If context is insufficient or ambiguous, set needs_user=true for that item.\n"
            "No extra text."
        )
        return self._complete_json(system=system, payload=payload)

