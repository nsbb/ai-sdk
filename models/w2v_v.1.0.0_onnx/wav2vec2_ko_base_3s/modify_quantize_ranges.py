#!/usr/bin/env python3
"""
양자화 파일(.quantize)의 activation 범위를 수정하여 uint8 정밀도 개선.

문제: 한국어 wav2vec2의 일부 activation이 range 200+ → scale > 1.0 → uint8에서 정보 파괴
해결: 큰 range를 가진 activation의 min/max를 클리핑하여 scale 축소

이 방법은 ONNX 모델을 수정하지 않고 양자화 파라미터만 조정.
Pegasus inference simulation으로 효과 검증 가능.
"""
import os
import re
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUANTIZE_FILE = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim_ma.quantize")


def parse_quantize(filepath):
    """quantize 파일을 파싱하여 구조화된 데이터로 반환"""
    entries = {}
    current_key = None
    current_entry = {}

    with open(filepath) as f:
        lines = f.readlines()

    header_lines = []
    in_header = True

    for line in lines:
        stripped = line.rstrip()

        if in_header and not stripped.startswith("  '"):
            header_lines.append(line)
            if stripped.startswith("quantize_parameters:"):
                in_header = False
            continue

        if stripped.startswith("  '") and stripped.endswith("':"):
            # 새 항목 시작
            if current_key:
                entries[current_key] = current_entry
            current_key = stripped.strip().rstrip(":")
            current_entry = {}
        elif ':' in stripped and current_key:
            key, val = stripped.strip().split(':', 1)
            val = val.strip()
            # 숫자 변환 시도
            try:
                if '.' in val:
                    val = float(val)
                elif val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                    val = int(val)
            except ValueError:
                pass
            current_entry[key] = val

    if current_key:
        entries[current_key] = current_entry

    return header_lines, entries


def write_quantize(filepath, header_lines, entries):
    """quantize 파일 저장"""
    with open(filepath, 'w') as f:
        for line in header_lines:
            f.write(line)

        for key, entry in entries.items():
            f.write(f"  {key}:\n")
            for k, v in entry.items():
                if isinstance(v, float):
                    f.write(f"    {k}: {v}\n")
                else:
                    f.write(f"    {k}: {v}\n")


def analyze_entries(entries):
    """양자화 파라미터 분석"""
    print(f"\nTotal entries: {len(entries)}")

    # scale별 분류
    high_scale = []
    for key, entry in entries.items():
        scale = entry.get('scale', 0)
        if isinstance(scale, (int, float)):
            if scale > 0.5:
                high_scale.append((key, entry))

    high_scale.sort(key=lambda x: -x[1].get('scale', 0))

    print(f"\nEntries with scale > 0.5 ({len(high_scale)}):")
    print(f"{'Name':<70} {'Scale':>10} {'Range':>12} {'min':>12} {'max':>12}")
    print("-" * 120)
    for key, entry in high_scale:
        name = key.strip("'")
        scale = entry.get('scale', 0)
        min_v = entry.get('min_value', 0)
        max_v = entry.get('max_value', 0)
        range_v = max_v - min_v if isinstance(min_v, (int, float)) and isinstance(max_v, (int, float)) else 0
        print(f"  {name[-66:]:<68} {scale:>10.4f} {range_v:>10.2f}   {min_v:>10.2f}   {max_v:>10.2f}")

    # attention 관련 텐서 분류
    attn_entries = [(k, e) for k, e in entries.items() if 'attention' in k.lower() or 'MatMul' in k]
    print(f"\nAttention-related entries: {len(attn_entries)}")

    return high_scale


def clip_ranges(entries, max_range=100.0, target_ops=None):
    """
    큰 range를 가진 activation의 min/max를 클리핑.
    max_range: 최대 허용 range (기본 100 → uint8 scale = 0.392)
    target_ops: None이면 모든 scale > threshold인 텐서, 아니면 특정 op만
    """
    modified = copy.deepcopy(entries)
    clipped_count = 0

    for key, entry in modified.items():
        scale = entry.get('scale', 0)
        min_v = entry.get('min_value', 0)
        max_v = entry.get('max_value', 0)

        if not isinstance(scale, (int, float)) or not isinstance(min_v, (int, float)):
            continue

        current_range = max_v - min_v
        if current_range <= max_range:
            continue

        if target_ops:
            if not any(op in key for op in target_ops):
                continue

        # 범위를 max_range로 축소 (중심 유지)
        center = (max_v + min_v) / 2.0
        new_min = center - max_range / 2.0
        new_max = center + max_range / 2.0

        new_scale = max_range / 255.0
        new_zp = int(round(-new_min / new_scale))
        new_zp = max(0, min(255, new_zp))

        entry['min_value'] = new_min
        entry['max_value'] = new_max
        entry['scale'] = new_scale
        entry['zero_point'] = new_zp

        clipped_count += 1

    return modified, clipped_count


def main():
    print("=" * 70)
    print("Quantize Range Modification for Korean Wav2Vec2")
    print("=" * 70)

    header, entries = parse_quantize(QUANTIZE_FILE)
    print(f"\nParsed: {QUANTIZE_FILE}")

    # 분석
    high_scale = analyze_entries(entries)

    # 실험 1: 모든 high-scale 텐서의 range를 100으로 제한
    for max_range in [200, 150, 100, 50]:
        modified, count = clip_ranges(entries, max_range=max_range)
        out_file = QUANTIZE_FILE.replace("_ma.quantize",
                                          f"_ma_rangeclip{max_range}.quantize")
        write_quantize(out_file, header, modified)
        print(f"\n  RangeClip={max_range}: {count} entries modified → {os.path.basename(out_file)}")

        # 수정된 엔트리의 새 scale 표시
        for key, entry in modified.items():
            if entry != entries.get(key, {}):
                name = key.strip("'")[-50:]
                print(f"    {name}: scale {entries[key].get('scale', 0):.4f} → {entry['scale']:.4f}")

    # 실험 2: attention MatMul 출력만 타겟
    for max_range in [100, 50, 30]:
        modified, count = clip_ranges(entries, max_range=max_range,
                                       target_ops=["MatMul", "attention"])
        out_file = QUANTIZE_FILE.replace("_ma.quantize",
                                          f"_ma_attnclip{max_range}.quantize")
        write_quantize(out_file, header, modified)
        print(f"\n  AttnClip={max_range}: {count} entries modified → {os.path.basename(out_file)}")
        for key, entry in modified.items():
            if entry != entries.get(key, {}):
                name = key.strip("'")[-50:]
                print(f"    {name}: scale {entries[key].get('scale', 0):.4f} → {entry['scale']:.4f}")

    print(f"\n{'='*70}")
    print("Next steps:")
    print("  1. pegasus inference --model-quantize <modified>.quantize --dtype quantized")
    print("  2. Compare uint8 simulation output with original")
    print("  3. If better, export to NB and test on device")


if __name__ == "__main__":
    main()
