# Korean Citrinet INT8 NB Handoff (1-input, fixed length)

This package is for Android app integration of the latest working NB.

## 1) Files to use (exact paths)

- NB model:
  - `/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/android_app_handoff/network_binary.nb`
- Quant/meta:
  - `/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/android_app_handoff/nbg_meta.json`
- Tokenizer (must use Korean model tokenizer):
  - `/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/android_app_handoff/tokenizer.model`
  - `/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/android_app_handoff/vocab_ko.txt`
  - `/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/android_app_handoff/model_config_ko.yaml`
- Reference input for sanity check:
  - `/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/android_app_handoff/input_0_ref.dat`
  - `/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/android_app_handoff/input_0_ref_float.npy`

## 2) Model I/O contract

- Input tensor:
  - name: `audio_signal`
  - shape: `[1, 80, 1, 300]` (NCHW)
  - dtype: `int8`
  - quant: `scale=0.02096451073884964`, `zero_point=-37`
- Output tensor:
  - shape: `[1, 2049, 1, 38]` (NCHW)
  - dtype: `int8`
  - quant: `scale=0.11265987902879715`, `zero_point=127`

## 3) Audio preprocess (must match server)

- WAV input: mono, 16kHz.
- If stereo: average channels.
- Feature extraction:
  - `n_fft=512`, `window_size=0.025s`, `hop=0.01s`, `window=hann`
  - mel bins: `80`
  - normalize: `per_feature`
  - `pad_to=16`
- Time handling:
  - crop to 300 frames if longer
  - zero-pad to 300 frames if shorter
- Float feature tensor shape before quantization: `[1, 80, 1, 300]` (float32)

## 4) Input quantization

For each float feature `x`:

- `q = round(x / 0.02096451073884964) + (-37)`
- clamp to `[-128, 127]`
- store as signed int8

Expected byte size: `1*80*1*300 = 24000` bytes.

## 5) Output decode

- Output layout `[1, 2049, 1, 38]` -> logits matrix `[T=38, C=2049]`.
- CTC greedy:
  - argmax over class axis for each frame
  - collapse repeats
  - remove blank id `2048`
- Convert token ids to text with `tokenizer.model` (SentencePiece).
- Note: argmax can be done on raw int8 output directly (dequant not required for argmax).

## 6) Minimal sanity check flow

1. Run NB with `input_0_ref.dat`.
2. Confirm output tensor shape is `1x2049x1x38`.
3. Run CTC+SentencePiece decode.
4. Then switch app input from reference dat to live WAV preprocess path.

