#!/usr/bin/env python3
"""Fix ALL uncalibrated entries in PCQ quantize file.

For i8 asymmetric_affine internal nodes (Gather, Unsqueeze, Mul):
  Use range [-128, 127] → scale=1.0, zp=0 as conservative default.
  This maps i8 values directly to float with 1:1 correspondence.

For Constant :data entries: keep as-is.
"""

import re

RANGES = {
    "cached_avg_0": (-0.6520153284072876, 0.5231920480728149),
    "cached_avg_1": (-0.49359890818595886, 0.3702373802661896),
    "cached_avg_2": (-0.17677751183509827, 0.2493944764137268),
    "cached_avg_3": (-0.11874654144048691, 0.16640740633010864),
    "cached_avg_4": (-0.23172907531261444, 0.28056102991104126),
    "cached_key_0": (-3.3077738285064697, 4.127389907836914),
    "cached_key_1": (-2.991609811782837, 2.8925249576568604),
    "cached_key_2": (-2.106147289276123, 2.3220324516296387),
    "cached_key_3": (-2.735071897506714, 2.3771708011627197),
    "cached_key_4": (-3.4068922996520996, 3.6414847373962402),
    "cached_val_0": (-2.814129590988159, 2.7313547134399414),
    "cached_val_1": (-4.124510288238525, 3.5030229091644287),
    "cached_val_2": (-2.0935235023498535, 2.6646666526794434),
    "cached_val_3": (-2.47493314743042, 2.4281296730041504),
    "cached_val_4": (-3.102327346801758, 3.7375881671905518),
    "cached_val2_0": (-4.947299957275391, 6.2559404373168945),
    "cached_val2_1": (-3.7970688343048096, 2.8566412925720215),
    "cached_val2_2": (-3.015770435333252, 3.746561288833618),
    "cached_val2_3": (-2.045248031616211, 1.7102863788604736),
    "cached_val2_4": (-4.847288131713867, 4.75404167175293),
    "cached_conv1_0": (-23.791166305541992, 29.784202575683594),
    "cached_conv1_1": (-32.329010009765625, 26.419137954711914),
    "cached_conv1_2": (-28.322477340698242, 33.21759796142578),
    "cached_conv1_3": (-12.875473976135254, 13.865696907043457),
    "cached_conv1_4": (-27.124448776245117, 27.570236206054688),
    "cached_conv2_0": (-31.056745529174805, 31.64058494567871),
    "cached_conv2_1": (-28.334440231323242, 29.148311614990234),
    "cached_conv2_2": (-83.66825103759766, 59.86714553833008),
    "cached_conv2_3": (-19.42669105529785, 19.53230857849121),
    "cached_conv2_4": (-48.51869201660156, 55.735836029052734),
}


def compute_i8_params(min_val, max_val):
    scale = (max_val - min_val) / 255.0
    zp = round(-128 - min_val / scale)
    zp = max(-128, min(127, zp))
    return scale, zp


def get_state_name(entry_key):
    m = re.match(r"'@(cached_\w+_\d+)_\d+:out0'", entry_key)
    if m:
        return m.group(1)
    m = re.match(r"'@Reshape_(cached_\w+_\d+)_reshaped_\d+:out0'", entry_key)
    if m:
        return m.group(1)
    return None


def main():
    src = "zipformer_encoder_folded4_with_states_v6_pcq.quantize"
    dst = "zipformer_encoder_folded4_with_states_v6_pcq_fixed_all.quantize"

    with open(src) as f:
        lines = f.readlines()

    fixed_state = 0
    fixed_internal = 0
    kept_const = 0

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Check if any line in the entry has scale: 1.0
        is_uncalibrated = False
        entry_end = min(i + 10, len(lines))
        for j in range(i + 1, entry_end):
            if lines[j].strip() == 'scale: 1.0':
                is_uncalibrated = True
                break
            if lines[j].strip().startswith("'@"):
                break

        if is_uncalibrated:
            entry_key = line.strip()

            if ":data'" in entry_key:
                kept_const += 1
                i += 1
                continue

            state_name = get_state_name(entry_key)
            if state_name and state_name in RANGES:
                min_val, max_val = RANGES[state_name]
                scale, zp = compute_i8_params(min_val, max_val)
                # Replace values in the entry
                for k in range(i + 1, entry_end):
                    if 'max_value:' in lines[k]:
                        lines[k] = f"    max_value: {max_val}\n"
                    elif 'min_value:' in lines[k]:
                        lines[k] = f"    min_value: {min_val}\n"
                    elif lines[k].strip().startswith('scale:'):
                        lines[k] = f"    scale: {scale}\n"
                    elif 'zero_point:' in lines[k]:
                        lines[k] = f"    zero_point: {zp}\n"
                    elif lines[k].strip().startswith("'@"):
                        break
                fixed_state += 1
            elif entry_key.endswith(":out0':"):
                # Internal activation node — use conservative range [-128, 127]
                # scale = 1.0, zp = 0 (identity for int8)
                for k in range(i + 1, entry_end):
                    if 'max_value:' in lines[k]:
                        lines[k] = f"    max_value: 127.0\n"
                    elif 'min_value:' in lines[k]:
                        lines[k] = f"    min_value: -128.0\n"
                    elif lines[k].strip().startswith('scale:'):
                        lines[k] = f"    scale: 1.0\n"  # keep as-is
                    elif 'zero_point:' in lines[k]:
                        lines[k] = f"    zero_point: 0\n"  # fix from -128 to 0
                    elif lines[k].strip().startswith("'@"):
                        break
                fixed_internal += 1

        i += 1

    with open(dst, 'w') as f:
        f.writelines(lines)

    remaining = sum(1 for l in lines if l.strip() == 'scale: 1.0')
    print(f"Fixed state entries: {fixed_state}")
    print(f"Fixed internal entries: {fixed_internal}")
    print(f"Kept constant :data entries: {kept_const}")
    print(f"Remaining scale=1.0: {remaining}")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
