# Wav2Vec2-base-960h T527 NPU Results

## Model Info
- **Model**: facebook/wav2vec2-base-960h (English STT)
- **Input**: [1, 80000] float32 → uint8 (5 seconds @ 16kHz)
- **Output**: [1, 249, 32] uint8 → CTC greedy decode
- **NB Size**: 87MB (uint8 quantization)
- **Quantization**: uint8 asymmetric_affine, moving_average (w=0.004), 51 calibration samples, reverse_channel=false

## Input Quantization Parameters
- Scale: 0.002860036
- Zero Point: 137

## Output Quantization Parameters
- Scale: 0.150270
- Zero Point: 186

## Test Results (50 LibriSpeech test-clean samples, T527 NPU)

### CER Summary (50 samples, vpm_run direct NPU)

| Metric | ONNX FP32 | NPU uint8 | Degradation |
|--------|-----------|-----------|-------------|
| **CER** | **9.74%** | **17.52%** | **+7.78%p** |
| **WER** | — | **27.38%** | — |
| **Exact match** | 25/50 (50%) | 6/50 (12%) | — |
| **Avg Inference** | — | **715ms** | — |

### Quantization Degradation
- NPU uint8 vs ONNX FP32 CER: **9.75%** (measures quantization loss only)
- ONNX FP32 baseline CER vs GT: **9.74%** (model's inherent accuracy with 5s truncation)
- NPU uint8 CER vs GT: **17.52%** (combined model + quantization error)

### Sample Results (selected)
| # | Ground Truth | NPU Output | CER |
|---|-------------|------------|-----|
| 7 | I AM CONVINCED OF WHAT I SAY SAID THE COUNT | I AM CONVINCED OF WHAT I SAY SAID THE COUNT | 0.0% |
| 25 | THIS THOUGHT HOWEVER DID NOT ENTER THE HEADS... | THIS THOUGHT HOWEVER DID NOT ENTER THE HEADS... | 0.0% |
| 37 | NO SOUND BROKE THE STILLNESS OF THE NIGHT | NO SOUND BROKE THE STILLNESS OF THE NIGHT | 0.0% |
| 8 | IT IS ANNOYANCE THEN | S N | 88.2% |
| 31 | THEN SHE SUDDENLY REMARKED | BER ESMAK | 73.9% |
| 28 | NOW LET'S DUST THE FURNITURE AND PICTURES | ALLITS DOS TH FIRNTURN PICTURE | 40.0% |

### Notes
- Short utterances (<2s) have higher CER due to the 5s zero-padding affecting model context
- Test data: LibriSpeech test-clean speaker 6930, streamed via HuggingFace datasets API
- Audio durations: 1.8s - 7.4s (padded/truncated to 5s = 80000 samples)

## Model Comparison: T527 NPU STT Models

| Model | Language | CER | WER | Inference | NB Size | Input |
|-------|----------|-----|-----|-----------|---------|-------|
| **KoCitrinet 300f int8** | Korean | **44.44%** | — | **120ms** | **~5MB** | 3s |
| **Wav2Vec2 uint8** | English | **17.52%** | **27.38%** | **715ms** | **87MB** | 5s |

### Analysis
- Wav2Vec2 has better CER (17.5% vs 44.4%) but English vs Korean is not a fair comparison (Korean is agglutinative, harder for character-level CTC)
- KoCitrinet is **6x faster** (120ms vs 715ms)
- KoCitrinet NB is **17x smaller** (5MB vs 87MB)
- Wav2Vec2 uint8 quantization adds **+7.8%p CER** over ONNX FP32 (9.7% → 17.5%)
- Both models run faster than real-time: Wav2Vec2 processes 5s in 0.72s (**7x RT**), KoCitrinet processes 3s in 0.12s (**25x RT**)

## Critical Bugs Fixed
1. **JNI float32→uint8 casting**: `processedAudio` (raw float32) was passed to NPU instead of `quantized_input` (properly quantized uint8)
2. **reverse_channel=true in inputmeta**: Corrupted calibration, causing garbage output despite correct code flow
3. **Output tensor layout**: [seq, vocab] not [vocab, seq] — wrong indexing produced random token selection
4. **Model dimensions**: MODEL_INPUT_LENGTH and OUTPUT_SEQ_LEN were wrong for the 5s model

## Failed Quantization Attempts
| Method | Status | Issue |
|--------|--------|-------|
| bf16 | gen_nbg segfault | NB too large (~181MB) |
| PCQ int8 | gen_nbg segfault | "Reshape tensor error" |
| int16 | NPU hangs | Driver incompatibility |
| fp32 | SRAM overflow | Model too large |
| hybrid | No effect | Flag didn't change precision |
| MLE | Acuity crash | Internal AttributeError |
| KL divergence | Quantize OK | Not exported to NB (similar quality to MA in simulation) |

## Reproduction

### vpm_run test
```bash
WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
$WIN_ADB push network_binary.nb /data/local/tmp/w2v_test/
$WIN_ADB push input_0000.dat /data/local/tmp/w2v_test/
$WIN_ADB push sample_0000.txt /data/local/tmp/w2v_test/
$WIN_ADB shell "cd /data/local/tmp/w2v_test && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample_0000.txt -b 0"
$WIN_ADB pull /data/local/tmp/w2v_test/output_0000.dat .
python3 eval_wav2vec_cer.py --output-dir . --gt ground_truth.txt
```

### Android app test
```bash
# Single file:
adb shell am start -n com.t527.awaiasr_2/com.t527.wav2vecdemo.Wav2VecTestActivity --es auto_test test.wav
# Batch test:
adb shell am start -n com.t527.awaiasr_2/com.t527.wav2vecdemo.Wav2VecTestActivity --es auto_en_batch ground_truth.txt
adb logcat -s Wav2VecTestActivity | grep BATCH_
```
