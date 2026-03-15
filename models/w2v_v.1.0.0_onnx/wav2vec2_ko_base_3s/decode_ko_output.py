#!/usr/bin/env python3
"""Decode wav2vec2-base-korean NPU output (output_0.dat) to Korean text.

The model outputs Korean jamo characters via CTC, which need to be composed
into syllables for readable Korean text.

Usage:
    python3 decode_ko_output.py output_0.dat
    python3 decode_ko_output.py output_0.dat --model nopad  # use nopad quantization params
    python3 decode_ko_output.py output_0.dat --gt "정답 텍스트"

Model output spec (from nbg_meta.json):
    Shape: [1, 149, 56] uint8 (original) or [1, 149, 56] (nopad variants)
    Original:  scale=0.06898809224367142, zp=76
    Nopad:     scale=0.104530, zp=143

Vocab (56 tokens):
    0-18:  초성 자음 (ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ)
    19-39: 중성 모음 (ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ)
    40-50: 복합 종성 (ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ)
    51:    space ' '
    52:    [UNK]
    53:    [PAD]  (CTC blank)
    54:    <s>
    55:    </s>
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Quantization parameters for different model variants ─────────────
QUANT_PARAMS = {
    "original": {"scale": 0.06898809224367142, "zp": 76, "shape": (149, 56)},
    "nopad":    {"scale": 0.104530, "zp": 143, "shape": (149, 56)},
    "nopad5":   {"scale": 0.104530, "zp": 143, "shape": (249, 56)},
}

# ── Korean Jamo composition tables ──────────────────────────────────
# Unicode Hangul Syllable = 0xAC00 + (초성 * 21 + 중성) * 28 + 종성
HANGUL_BASE = 0xAC00

# 초성 (leading consonants) - 19 chars
CHOSEONG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")

# 중성 (vowels) - 21 chars
JUNGSEONG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

# 종성 (trailing consonants) - 28 slots (0 = no jongseong)
# Index 0 means no trailing consonant
JONGSEONG = [
    "",   # 0: no jongseong
    "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ",  # 1-7
    "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",  # 8-15
    "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ",  # 16-22
    "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",  # 23-27
]

# Build reverse lookup: jamo char -> (type, index)
# type: 'cho' (choseong), 'jung' (jungseong), 'jong' (jongseong)
CHOSEONG_MAP = {c: i for i, c in enumerate(CHOSEONG)}
JUNGSEONG_MAP = {c: i for i, c in enumerate(JUNGSEONG)}
JONGSEONG_MAP = {c: i for i, c in enumerate(JONGSEONG) if c}  # skip empty


def classify_jamo(ch):
    """Classify a jamo character as choseong, jungseong, or jongseong candidate.

    Returns (type, index) where type is 'cho', 'jung', or 'jong'.
    Some consonants can be either choseong or jongseong depending on context.
    """
    if ch in JUNGSEONG_MAP:
        return ("jung", JUNGSEONG_MAP[ch])
    if ch in CHOSEONG_MAP:
        return ("cho", CHOSEONG_MAP[ch])
    if ch in JONGSEONG_MAP:
        # Compound jongseong (ㄳ, ㄵ, ㄶ, etc.) can only be jongseong
        return ("jong", JONGSEONG_MAP[ch])
    return None


def is_consonant(ch):
    """Check if char is a consonant (could be choseong or jongseong)."""
    return ch in CHOSEONG_MAP or ch in JONGSEONG_MAP


def is_vowel(ch):
    """Check if char is a vowel (jungseong)."""
    return ch in JUNGSEONG_MAP


def compose_jamo_to_syllables(jamo_str: str) -> str:
    """Compose a string of Korean jamo characters into syllables.

    Algorithm:
    Scan left-to-right, building syllables as (cho, jung, jong?).
    A consonant followed by a vowel starts choseong+jungseong.
    A consonant after a vowel becomes jongseong, UNLESS the next char is also
    a vowel, in which case it starts a new choseong.

    This handles the standard Korean syllable structure: C V (C).
    """
    if not jamo_str:
        return ""

    result = []
    i = 0
    n = len(jamo_str)

    while i < n:
        ch = jamo_str[i]

        # Non-jamo character (space, etc.) -> output directly
        if not is_consonant(ch) and not is_vowel(ch):
            result.append(ch)
            i += 1
            continue

        # Try to build a syllable: Choseong + Jungseong + (optional Jongseong)
        # Case 1: consonant followed by vowel -> start syllable
        if is_consonant(ch) and ch in CHOSEONG_MAP and i + 1 < n and is_vowel(jamo_str[i + 1]):
            cho_idx = CHOSEONG_MAP[ch]
            jung_idx = JUNGSEONG_MAP[jamo_str[i + 1]]
            jong_idx = 0  # no jongseong by default

            # Check for jongseong: next char is consonant
            if i + 2 < n and is_consonant(jamo_str[i + 2]):
                cand = jamo_str[i + 2]
                # If there's a vowel after the consonant candidate, this consonant
                # is the choseong of the NEXT syllable, not jongseong of current
                if i + 3 < n and is_vowel(jamo_str[i + 3]):
                    # consonant belongs to next syllable as choseong
                    # BUT: compound jongseong (ㄳ, ㄵ, etc.) can't be choseong
                    if cand in JONGSEONG_MAP and cand not in CHOSEONG_MAP:
                        # compound jongseong -> must be jongseong here
                        jong_idx = JONGSEONG_MAP[cand]
                        i += 3
                    else:
                        # simple consonant -> starts next syllable
                        i += 2
                else:
                    # No vowel after -> this consonant is jongseong
                    if cand in JONGSEONG_MAP:
                        jong_idx = JONGSEONG_MAP[cand]
                        i += 3
                    elif cand in CHOSEONG_MAP:
                        # Use single consonant jongseong mapping
                        # ㄱ,ㄲ,ㄴ,ㄷ,ㄹ,ㅁ,ㅂ,ㅅ,ㅆ,ㅇ,ㅈ,ㅊ,ㅋ,ㅌ,ㅍ,ㅎ
                        if cand in JONGSEONG_MAP:
                            jong_idx = JONGSEONG_MAP[cand]
                        i += 3
                    else:
                        i += 2
            else:
                i += 2

            # Compose the syllable
            code = HANGUL_BASE + (cho_idx * 21 + jung_idx) * 28 + jong_idx
            result.append(chr(code))

        # Case 2: vowel alone (no preceding consonant for this syllable)
        elif is_vowel(ch):
            # Standalone vowel: use ㅇ as placeholder choseong (index 11)
            jung_idx = JUNGSEONG_MAP[ch]
            cho_idx = CHOSEONG_MAP["ㅇ"]
            code = HANGUL_BASE + (cho_idx * 21 + jung_idx) * 28
            result.append(chr(code))
            i += 1

        # Case 3: consonant not followed by vowel -> output as-is
        else:
            result.append(ch)
            i += 1

    return "".join(result)


def ctc_greedy_decode(logits: np.ndarray, vocab: dict, blank_id: int = 53) -> str:
    """CTC greedy decode: take argmax, collapse repeated, remove blanks.

    Args:
        logits: float32 array of shape [T, V] (time_steps, vocab_size)
        vocab: dict mapping char -> id
        blank_id: CTC blank token id (default: 53 = [PAD])

    Returns:
        Decoded jamo string (before syllable composition)
    """
    # Build id -> char mapping
    id2char = {v: k for k, v in vocab.items()}

    # Argmax over vocab dimension
    pred_ids = np.argmax(logits, axis=-1)  # [T]

    # Collapse repeated tokens
    collapsed = []
    prev = -1
    for idx in pred_ids:
        if idx != prev:
            collapsed.append(int(idx))
        prev = idx

    # Remove blanks and special tokens
    special_ids = {blank_id, vocab.get("<s>", -1), vocab.get("</s>", -1)}
    decoded_ids = [i for i in collapsed if i not in special_ids]

    # Convert to characters
    chars = [id2char.get(i, "?") for i in decoded_ids]

    # Handle [UNK] token
    jamo_str = "".join(c if c != "[UNK]" else "" for c in chars)

    return jamo_str


def load_vocab(vocab_path: str = None) -> dict:
    """Load vocab.json."""
    if vocab_path is None:
        vocab_path = os.path.join(SCRIPT_DIR, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_output_file(
    output_path: str,
    model_variant: str = "original",
    vocab_path: str = None,
    gt_text: str = None,
    verbose: bool = True,
) -> dict:
    """Decode an output_0.dat file to Korean text.

    Args:
        output_path: Path to output_0.dat (uint8 binary)
        model_variant: "original", "nopad", or "nopad5"
        vocab_path: Path to vocab.json (default: same dir as script)
        gt_text: Ground truth text for comparison
        verbose: Print detailed info

    Returns:
        dict with decoded text, jamo, and optional CER
    """
    params = QUANT_PARAMS[model_variant]
    scale = params["scale"]
    zp = params["zp"]
    shape = params["shape"]

    # Load vocab
    vocab = load_vocab(vocab_path)

    # Load output_0.dat
    raw = np.fromfile(output_path, dtype=np.uint8)
    expected_size = shape[0] * shape[1]
    if len(raw) != expected_size:
        print(f"WARNING: Expected {expected_size} bytes, got {len(raw)}")
        print(f"  Trying to infer shape from file size...")
        vocab_size = shape[1]
        time_steps = len(raw) // vocab_size
        if len(raw) % vocab_size != 0:
            print(f"  ERROR: file size {len(raw)} not divisible by vocab_size {vocab_size}")
            sys.exit(1)
        shape = (time_steps, vocab_size)
        print(f"  Inferred shape: {shape}")

    # Reshape to [T, V]
    output_u8 = raw.reshape(shape)

    # Dequantize: float = (uint8 - zp) * scale
    logits = (output_u8.astype(np.float32) - zp) * scale

    if verbose:
        print(f"Output file: {output_path}")
        print(f"  Size: {len(raw)} bytes")
        print(f"  Shape: {shape}")
        print(f"  Model variant: {model_variant}")
        print(f"  Quant: scale={scale}, zp={zp}")
        print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

    # CTC greedy decode
    blank_id = vocab.get("[PAD]", 53)
    jamo_str = ctc_greedy_decode(logits, vocab, blank_id=blank_id)

    if verbose:
        print(f"\n  Jamo output: {jamo_str}")

    # Compose jamo -> syllables
    text = compose_jamo_to_syllables(jamo_str)

    if verbose:
        print(f"  Composed text: {text}")

    # Compare with ground truth
    result = {
        "jamo": jamo_str,
        "text": text,
        "model_variant": model_variant,
        "shape": shape,
    }

    if gt_text:
        # Simple CER (character-level, ignoring spaces)
        pred_nospace = text.replace(" ", "")
        gt_nospace = gt_text.replace(" ", "")
        cer = compute_cer(pred_nospace, gt_nospace)
        result["gt_text"] = gt_text
        result["cer"] = cer
        if verbose:
            print(f"\n  Ground truth: {gt_text}")
            print(f"  CER: {cer:.2%}")

    # Show top-5 predictions per time step for first few steps
    if verbose:
        id2char = {v: k for k, v in vocab.items()}
        print(f"\n  First 10 time steps (top-3 predictions):")
        pred_ids = np.argmax(logits, axis=-1)
        for t in range(min(10, shape[0])):
            top_k = np.argsort(logits[t])[::-1][:3]
            items = []
            for k in top_k:
                ch = id2char.get(k, "?")
                prob = logits[t, k]
                marker = "*" if k == pred_ids[t] else " "
                items.append(f"{marker}{ch}({k})={prob:.2f}")
            print(f"    t={t:3d}: {' | '.join(items)}")

    return result


def compute_cer(pred: str, ref: str) -> float:
    """Compute Character Error Rate using edit distance."""
    if not ref:
        return 1.0 if pred else 0.0

    # Dynamic programming edit distance
    n, m = len(ref), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / n


def batch_decode(test_data_dir: str, model_variant: str = "original", vocab_path: str = None):
    """Decode all output files in test_data_dir using manifest."""
    manifest_path = os.path.join(test_data_dir, "test_manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"No manifest found: {manifest_path}")
        print("Run prepare_ko_test_input.py --batch first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    vocab = load_vocab(vocab_path)
    results = []

    for entry in manifest:
        basename = entry["basename"]
        gt_text = entry.get("gt_text")
        # Look for corresponding output file
        output_path = os.path.join(test_data_dir, f"{basename}_output_0.dat")
        if not os.path.isfile(output_path):
            print(f"  SKIP (no output): {output_path}")
            continue

        print(f"\n{'─' * 50}")
        print(f"File: {basename}")
        result = decode_output_file(output_path, model_variant, vocab_path, gt_text)
        results.append(result)

    if results:
        cers = [r["cer"] for r in results if "cer" in r]
        if cers:
            print(f"\n{'=' * 50}")
            print(f"Average CER: {sum(cers)/len(cers):.2%} ({len(cers)} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Decode wav2vec2-base-korean NPU output to Korean text"
    )
    parser.add_argument("output_path", nargs="?",
                        help="Path to output_0.dat (uint8 binary)")
    parser.add_argument("--model", "-m", default="original",
                        choices=list(QUANT_PARAMS.keys()),
                        help="Model variant for dequantization (default: original)")
    parser.add_argument("--vocab", default=None,
                        help="Path to vocab.json (default: same dir as script)")
    parser.add_argument("--gt", default=None,
                        help="Ground truth text for CER calculation")
    parser.add_argument("--batch", default=None,
                        help="Batch decode from test_data directory")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress detailed output")
    args = parser.parse_args()

    if args.batch:
        batch_decode(args.batch, args.model, args.vocab)
    elif args.output_path:
        result = decode_output_file(
            args.output_path,
            model_variant=args.model,
            vocab_path=args.vocab,
            gt_text=args.gt,
            verbose=not args.quiet,
        )
        if args.quiet:
            print(result["text"])
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 decode_ko_output.py output_0.dat")
        print("  python3 decode_ko_output.py output_0.dat --model nopad --gt '안녕하세요'")
        print("  python3 decode_ko_output.py --batch test_data/ --model original")
        sys.exit(1)


if __name__ == "__main__":
    main()
