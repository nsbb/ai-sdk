# Android Build

```
mm
```

find in ~/out/target/product/xxx/vendor/bin/deepspeech2

Burn firmare，then will be found in device /vendor/bin/deepspeech2.



#  Run Demo


1. push deepspeech2 into device /data or sdcard dir

2. STFT get wav tensor:

   ```
   python3 pre_process.py --wav=1188-133604-0010.flac.wav
   ```

3. push deepspeech2.nb, 1188-133604-0010.flac\_756\_161\_1.tensor into device

4. run command `deepspeech2 deepspeech2.nb 1188-133604-0010.flac_756_161_1.tensor`

5. print result as,

```
./deepspeech2 nbg input
input tensor_file = 1188-133604-0010.flac_756_161_1.tensor
Original array: but  in  thiss  vignnye  copiiedd ffrom  turrrnerr  yoou haavve the  ttwo prrincciiplleess   brrougght  oout perrfecctllly
Modified array : but in this vignye copied from turner you have the two principles brought out perfectly
```

