#!/usr/bin/env python
"""Benchmark pinyinize with Ollama LLM against Chinese-TTS-Dataset."""

import json
import re
import sys
from pathlib import Path
from pinyinize.core import PinyinizeOptions, pinyinize
from pinyinize.resources import PinyinResources
from pinyinize.llm import OllamaLLMAdapter

# Configuration
DATASET_PATH = Path("Chinese-TTS-Dataset/Chinese Polyphonic Characters.json")
MAX_CASES = 200
OLLAMA_MODEL = "gemma3:1b"
OLLAMA_HOST = "http://localhost:11434"

# Load dataset
print(f"Loading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Loaded {len(data)} total cases, will test {MAX_CASES}")

# Load resources
print("Loading pinyin resources...")
resources = PinyinResources.load_from_dir(Path("."))

# Setup Ollama adapter
print(f"Connecting to Ollama at {OLLAMA_HOST} (model: {OLLAMA_MODEL})...")
llm_adapter = OllamaLLMAdapter(model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout_s=120)

opts = PinyinizeOptions(
    resources=resources,
    llm_adapter=llm_adapter,
    double_check_adapter=llm_adapter,
    double_check_threshold=0.85,
)

# Tone conversion
TONE_NUM_TO_MARK = {
    ("a", "1"): "ā",
    ("a", "2"): "á",
    ("a", "3"): "ǎ",
    ("a", "4"): "à",
    ("e", "1"): "ē",
    ("e", "2"): "é",
    ("e", "3"): "ě",
    ("e", "4"): "è",
    ("i", "1"): "ī",
    ("i", "2"): "í",
    ("i", "3"): "ǐ",
    ("i", "4"): "ì",
    ("o", "1"): "ō",
    ("o", "2"): "ó",
    ("o", "3"): "ǒ",
    ("o", "4"): "ò",
    ("u", "1"): "ū",
    ("u", "2"): "ú",
    ("u", "3"): "ǔ",
    ("u", "4"): "ù",
    ("v", "1"): "ǖ",
    ("v", "2"): "ǘ",
    ("v", "3"): "ǚ",
    ("v", "4"): "ǜ",
    ("ü", "1"): "ǖ",
    ("ü", "2"): "ǘ",
    ("ü", "3"): "ǚ",
    ("ü", "4"): "ǜ",
}


def num_to_mark(syllable: str) -> str:
    if not syllable:
        return syllable
    
    tone = None
    if syllable[-1].isdigit():
        tone = syllable[-1]
        syllable = syllable[:-1]
    
    syllable = syllable.replace("u:", "ü").replace("v", "ü")

    if not tone or tone in ("0", "5"):
        return syllable

    s = syllable.lower()

    def mark_at(idx: int) -> str:
        v = syllable[idx]
        marked = TONE_NUM_TO_MARK.get((v, tone)) or TONE_NUM_TO_MARK.get((v.lower(), tone))
        if not marked:
            return syllable
        return syllable[:idx] + marked + syllable[idx + 1 :]

    # Pinyin tone placement:
    # 1) a/o/e take precedence
    for v in ("a", "o", "e"):
        pos = s.find(v)
        if pos != -1:
            return mark_at(pos)
    # 2) iu/ui mark second vowel
    pos = s.find("iu")
    if pos != -1:
        return mark_at(pos + 1)
    pos = s.find("ui")
    if pos != -1:
        return mark_at(pos + 1)
    # 3) otherwise mark the last vowel
    vowels = "aeiouü"
    last = -1
    for i, ch in enumerate(s):
        if ch in vowels:
            last = i
    return mark_at(last) if last != -1 else syllable

def normalize_expected(expected: str) -> str:
    # Replace punctuation with spaces to avoid merging syllables like "jiao3，cheng2".
    s = re.sub(r'[，。、；：？！“”‘’（）【】《》]', ' ', expected)
    s = re.sub(r'[\s,;.!?"\'()[\]{}]+', ' ', s)
    syllables = s.split()
    return ''.join(num_to_mark(syl) for syl in syllables if syl)

def normalize_actual(actual: str) -> str:
    s = re.sub(r'[，。、；：？！""''（）【】《》]', '', actual)
    s = re.sub(r'\s+', '', s)
    return s.replace('v', 'ü').replace('ɡ', 'g')

# Run benchmark with progress reporting
results = {
    'total': 0,
    'correct': 0,
    'incorrect': 0,
    'good_cases': [],
    'bad_cases': [],
}

print("\n" + "=" * 70)
print("STARTING BENCHMARK WITH OLLAMA LLM")
print("=" * 70 + "\n")

for idx, item in enumerate(data[:MAX_CASES], 1):
    text = item['text']
    expected = item['pinyin']
    item_id = item['id']
    
    print(f"[{idx}/{MAX_CASES}] Testing {item_id}...", end=' ', flush=True)
    
    try:
        pinyin_results = pinyinize(text, opts)
        candidates = pinyin_results or []
        actual_by_segmenter = [(r.segmenter, r.output_text) for r in candidates]
        
        exp_norm = normalize_expected(expected)
        act_norms = [(seg, normalize_actual(out)) for (seg, out) in actual_by_segmenter]
        matched_seg = None
        for seg, norm in act_norms:
            if exp_norm == norm:
                matched_seg = seg
                break
        is_correct = matched_seg is not None
        
        results['total'] += 1
        
        if is_correct:
            results['correct'] += 1
            status = f"✓ PASS ({matched_seg})" if matched_seg else "✓ PASS"
            if len(results['good_cases']) < 5:
                results['good_cases'].append({
                    'id': item_id,
                    'text': text[:40] + '...' if len(text) > 40 else text,
                })
        else:
            results['incorrect'] += 1
            first_seg, first_norm = act_norms[0] if act_norms else (None, "")
            first_seg_s = f" ({first_seg})" if first_seg else ""
            status = "✗ FAIL: " + exp_norm + '\n' + (first_norm + first_seg_s)
            if len(results['bad_cases']) < 10:
                act_norm = first_norm
                # Find first diff position
                diff_pos = 0
                for i, (e, a) in enumerate(zip(exp_norm, act_norm)):
                    if e != a:
                        diff_pos = i
                        break
                results['bad_cases'].append({
                    'id': item_id,
                    'text': text[:40] + '...' if len(text) > 40 else text,
                    'expected': exp_norm[:50],
                    'actual': act_norm[:50],
                    'diff_at': diff_pos,
                })
        
        accuracy = results['correct'] / results['total'] * 100
        print(f"{status} | Accuracy: {accuracy:.1f}% ({results['correct']}/{results['total']})")
        
    except Exception as e:
        results['incorrect'] += 1
        print(f"✗ ERROR: {e}")

# Final report
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
accuracy = results['correct'] / results['total'] * 100 if results['total'] > 0 else 0
print(f"Total cases:    {results['total']}")
print(f"Correct:        {results['correct']} ({accuracy:.1f}%)")
print(f"Incorrect:      {results['incorrect']} ({100-accuracy:.1f}%)")

if results['good_cases']:
    print("\n" + "=" * 70)
    print("GOOD CASES (Perfect Match)")
    print("=" * 70)
    for case in results['good_cases']:
        print(f"  {case['id']}: {case['text']}")

if results['bad_cases']:
    print("\n" + "=" * 70)
    print("BAD CASES (Mismatch)")
    print("=" * 70)
    for case in results['bad_cases']:
        print(f"\n  {case['id']}: {case['text']}")
        print(f"    Expected: {case['expected']}")
        print(f"    Actual:   {case['actual']}")

print("\n" + "=" * 70)
print("NOTES")
print("=" * 70)
print("""
Common sources of mismatch:
1. Tone sandhi (一, 不) - dataset applies contextual tone changes
2. Neutral tones (的, 了, 头) - dataset uses tone 0 for particles
3. Multi-syllable words - segmentation may differ
4. LLM segmentation errors - check llm_segment_and_tag in report
""")
