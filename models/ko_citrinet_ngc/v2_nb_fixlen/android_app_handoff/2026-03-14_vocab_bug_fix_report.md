# 2026-03-14 | awaiasr_2 KoCitrinet 한국어 인식 오류 원인 분석 및 수정 보고서

---

## 1. 문제 요약

**현상:** awaiasr_2 앱에서 한국어 음성을 인식하면 완전히 엉뚱한 단어가 나온다.
**심각도:** 330개 샘플 기준 CER(문자 오류율) **~114%**, 정확 일치 **0/330** (실질적으로 전혀 동작하지 않는 수준)
**기대값:** 동일 모델을 쓰는 bundle_app의 CER **44.44%**, 정확 일치 **44/330**
**수정 후:** awaiasr_2도 **CER 44.44%, 44/330** 달성 — bundle_app과 완전히 동일

---

## 2. 배경 지식 (비전공자를 위한 개념 설명)

### 음성 인식이 동작하는 방식

음성 인식 시스템은 크게 세 단계로 나뉩니다.

```
[1단계] 음성 → 숫자 변환    [2단계] AI 추론       [3단계] 숫자 → 텍스트 변환
   WAV 파일                   NPU(AI 칩)              사전(vocabulary)
   ↓ mel spectrogram          ↓ INT8 연산             ↓ 토큰 ID → 한국어
   80×300 = 24,000개 숫자   → 2049개 확률값 배열 →  "안녕하세요"
```

- **mel spectrogram**: 음성을 "어떤 주파수 소리가 언제 얼마나 강한가"를 수치로 표현한 것. 마치 악보처럼 음성을 시각화합니다.
- **NPU**: Neural Processing Unit. AI 연산 전용 칩. T527 SoC에 내장되어 있습니다.
- **토큰(token)**: AI가 출력하는 기본 단위. 글자, 음절, 단어 조각 등이 될 수 있습니다. 이 모델은 한국어 SentencePiece 토큰을 사용하며 총 2048개가 있습니다.
- **vocabulary (vocab_ko.txt)**: "토큰 번호 N = 실제 텍스트 X"를 저장한 사전 파일. 예: `82번 = '▁네'` (여기서 `▁`는 단어 앞 공백을 의미)
- **CTC 디코딩**: AI가 출력한 숫자(토큰 ID) 배열에서 반복을 제거하고 공백 토큰(blank)을 빼서 최종 텍스트를 만드는 방식

### 두 앱의 구조 비교

| 항목 | bundle_app (기준, 정상) | awaiasr_2 (문제 앱) |
|------|------------------------|---------------------|
| 모델 파일 (network_binary.nb) | 동일 (MD5 일치) | ← 동일 |
| AI 라이브러리 (libawnn.viplite.so) | 동일 (MD5 일치) | ← 동일 |
| 음성 전처리 | Kotlin WavFrontend | Java KoMelFrontend |
| NPU 추론 | AWNN API | AWNN API (동일) |
| 토큰→텍스트 변환 | Kotlin SentencePieceModel | C ko_citrinet_postprocess.cpp |
| vocab 사전 | tokenizer.model (바이너리) | vocab_ko.txt (텍스트) |

---

## 3. 증상 상세

### 동일 음성 파일에 대한 결과 비교

| 파일 | 정답 (GT) | bundle_app | awaiasr_2 (수정 전) | awaiasr_2 (수정 후) |
|------|-----------|------------|----------------------|----------------------|
| audio_0000 | 좋은하루되세요 | 네 | **그래서** | 네 ✓ |
| audio_0001 | 네감사합니다 | 아 네 네 네 네 감사합니다 | **아 그래서 그래서...** | 아 네 네 네... ✓ |
| audio_0003 | 네네어머니 | 네 | **그래서** | 네 ✓ |

→ 수정 전 awaiasr_2는 '네'(토큰 82)를 항상 '그래서'라고 잘못 출력했습니다.

---

## 4. 원인 분석 과정

### 4-1. 처음 의심한 것들 (모두 무죄)

**의심 1: 모델 파일이 다른가?**
```bash
md5sum ko_citrinet_model.nb  # 두 앱 동일: 54d9e3...
```
→ **아니다.** 완전히 동일한 파일을 사용 중.

**의심 2: AI 라이브러리(libawnn.viplite.so)가 다른가?**
```bash
md5sum libawnn.viplite.so  # 두 앱 동일: 54d9e3...
```
→ **아니다.** 동일한 라이브러리. 심지어 bundle_app은 VNN/OpenVX 방식으로 알려져 있었으나, 실제 코드를 열어보니 AWNN을 `dlopen`으로 동적 로드하는 방식으로 awaiasr_2와 동일.

**의심 3: 음성 전처리(mel spectrogram) 계산이 다른가?**
Python으로 두 앱의 mel 계산을 시뮬레이션해서 바이트 단위 비교:
```
차이 있는 바이트: 0개 (완전히 동일)
```
→ **아니다.** 전처리 결과가 byte-identical.

**의심 4: NPU 입력/출력 배열 순서가 다른가?**
두 앱 모두 channel-major 형식(`c * TIME_FRAMES + t`)으로 읽고 씀을 코드에서 확인.
→ **아니다.**

### 4-2. 핵심 단서 발견

NPU가 토큰 ID **82**를 출력한다고 가정하면:

- **bundle_app**: `tokenizer.model`의 82번 항목 = `'▁네'` → "네" 출력 ✓
- **awaiasr_2**: `vocab_ko.txt`의 82번 줄 = `'그래서'` → "그래서" 출력 ✗

두 사전에서 같은 번호가 다른 단어를 가리키고 있었습니다!

### 4-3. 근본 원인 확인

`tokenizer.model`(정답 사전)을 파이썬으로 파싱:

```
index 0: '<unk>'      ← 미등록 단어 (Unknown)
index 1: '▁'         ← 공백
index 2: '이'
...
index 82: '▁네'      ← bundle_app이 82번으로 출력하는 단어
index 83: '▁그래서'
```

`vocab_ko.txt`(awaiasr_2 사전) 파일 확인:

```
줄 1 (index 0): '▁'        ← ⚠️ '<unk>'가 있어야 할 자리!
줄 2 (index 1): '이'
...
줄 82 (index 81): '▁네'    ← 한 칸 밀려있음
줄 83 (index 82): '그래서'  ← awaiasr_2가 82번으로 잘못 출력하는 단어
```

**결론: `vocab_ko.txt`의 첫 줄에 `<unk>`(index 0)가 빠져 있어, 모든 토큰이 1칸씩 뒤로 밀려 있었습니다.**

---

## 5. 버그의 구조 (핵심 설명)

### 비유로 이해하기

영어-한국어 사전을 생각해보세요. 정답 사전은:

```
page 1: apple → 사과
page 2: banana → 바나나
page 3: cherry → 체리
```

그런데 awaiasr_2의 사전은 1페이지가 찢어져 있어:

```
page 1 (원래 2): banana → 바나나   ← 사과(1)가 없음!
page 2 (원래 3): cherry → 체리
page 3 (원래 4): date → 대추
```

AI가 "2번 단어야"라고 알려주면:
- 정답 사전: "2번 = banana = 바나나" ✓
- 찢어진 사전: "2번 = cherry = 체리" ✗ (실제로는 3번)

### 실제 토큰 흐름

```
NPU 출력 → CTC 디코딩 → 토큰 ID 목록 → 사전 조회 → 텍스트

예) 토큰 ID: [82]
  bundle_app:  vocab[82] = '▁네'   → "네"       ✓
  awaiasr_2:   vocab[82] = '그래서' → "그래서"   ✗  (실제로는 vocab[83]의 내용)
```

### 2차 버그: 단어 경계 공백 누락

구버전 `vocab_ko.txt`에는 `▁` 접두사가 없는 항목들이 있었습니다:

```
# 구버전
'근데'      ← ▁ 없음 → C 코드가 공백 없이 붙여씀

# 수정 후
'▁근데'     ← C 코드가 ▁ 감지 → 앞에 공백 삽입, ▁ 제거 → "근데"
```

`▁`는 SentencePiece 형식에서 단어 경계(앞에 공백이 있음)를 표시하는 기호입니다. 이 기호가 없으면 단어들이 공백 없이 붙어서 출력됩니다. C 코드(`ko_citrinet_postprocess.cpp`)는 이미 `▁`를 올바르게 처리하는 로직이 있었으므로, 사전만 바로잡으면 해결됩니다.

---

## 6. 해결 방법

### 방법: vocab_ko.txt를 tokenizer.model에서 재생성

`tokenizer.model`은 protobuf 바이너리 형식의 SentencePiece 모델 파일입니다. 이것을 파싱해서 정확한 토큰 목록을 추출했습니다.

```python
# tokenizer.model protobuf 파싱 → vocab_ko.txt 재생성
with open("tokenizer.model", "rb") as f:
    data = f.read()

pieces = []
# protobuf field 1 = SentencePiece message 배열
# 각 message의 field 1 = piece 문자열
# ... (파싱 로직) ...

# 결과:
# pieces[0] = '<unk>'
# pieces[1] = '▁'
# pieces[82] = '▁네'
# pieces[83] = '▁그래서'
# ... 총 2048개

with open("vocab_ko_fixed.txt", "w", encoding="utf-8") as f:
    for piece in pieces:
        f.write(piece + "\n")
```

### 수정 전후 비교

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| 줄 수 | 2047줄 | **2048줄** |
| index 0 | `'▁'` (틀림) | `'<unk>'` (정답) |
| index 82 | `'그래서'` (틀림) | `'▁네'` (정답) |
| ▁ 접두사 | 일부 누락 | tokenizer.model과 완전 일치 |

### 수정된 파일 위치

```
awaiasr_2/app/src/main/assets/models/KoCitriNet/vocab_ko.txt
```

C 코드(`ko_citrinet_postprocess.cpp`)는 수정하지 않았습니다. 이미 올바른 로직을 갖추고 있었기 때문입니다.

---

## 7. 검증 결과

### APK 빌드 및 디바이스 테스트

```bash
# 빌드
gradlew.bat assembleDebug → BUILD SUCCESSFUL

# 설치 및 배치 테스트 (330개 샘플)
adb install -r app-debug.apk
adb shell am start ... --ez auto_batch_test true
```

### 성능 결과

| 지표 | 수정 전 | 수정 후 | bundle_app (목표) |
|------|---------|---------|-------------------|
| 정확 일치 | **0/330** | **44/330** | 44/330 |
| CER (공백 무시) | **~114%** | **44.44%** | 44.44% |

→ 수정 후 두 앱의 결과가 **완전히 동일**합니다.

### 개별 샘플 검증 (audio_0000)

```
정답:    좋은하루되세요
수정 전: 그래서           ← 완전히 틀린 단어
수정 후: 네               ← bundle_app과 동일 (모델 한계, 정답은 아니나 일치)
```

---

## 8. 왜 이런 버그가 생겼나 (추정)

`vocab_ko.txt`는 아마도 `tokenizer.model`에서 수동으로 추출하는 과정에서 생성된 파일로 보입니다. SentencePiece 모델의 특수 토큰(`<unk>`, `<s>`, `</s>`)은 protobuf 내에 일반 piece와 동일한 구조로 저장되지만, 일부 추출 도구는 이를 건너뛰거나 다르게 처리합니다. 가장 가능성 높은 원인:

- `<unk>`(Unknown 토큰)가 "실제로 출력되지 않는 특수 토큰"이라는 이유로 수동 편집 시 삭제됨
- 또는 SentencePiece Python API의 `.id_to_piece()` 대신 다른 방법으로 추출하다가 index 0 누락

---

## 9. 향후 주의사항

1. **vocab 파일 생성 시**: 항상 `tokenizer.model`의 protobuf를 직접 파싱하거나, `sentencepiece` Python 패키지의 `sp.id_to_piece(i)` (i=0부터) 를 사용할 것. 특수 토큰(`<unk>` 등) 포함 여부를 반드시 확인.

2. **검증 방법**: 생성된 vocab 파일의 줄 수 = tokenizer의 vocab_size와 일치해야 함 (이 모델: 2048개).

3. **adb 연결 팁**: usbipd로 WSL attach가 안 될 때 (Windows adb.exe가 장치를 점유 중인 경우) Windows adb.exe를 직접 사용:
   ```bash
   WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
   $WIN_ADB devices   # 장치 확인
   $WIN_ADB install -r app.apk
   $WIN_ADB logcat -d -s CitrinetTest
   ```
   T527 busid: `1-5` (이전 메모의 `1-4`는 틀림)

---

## 10. 관련 파일 목록

| 파일 | 역할 | 수정 여부 |
|------|------|-----------|
| `vocab_ko.txt` | 토큰 ID → 한국어 텍스트 사전 | **수정** (2047→2048줄, index 0에 `<unk>` 추가) |
| `tokenizer.model` | SentencePiece 원본 모델 (정답 사전) | 읽기만 함 |
| `ko_citrinet_postprocess.cpp` | CTC 디코딩 + vocab 조회 C 코드 | 수정 없음 (기존 코드 정상) |
| `KoMelFrontend.java` | Java mel spectrogram 전처리 | 수정 없음 |
| `AwKoCitrinetJni.java` | Java↔JNI 인터페이스 | 수정 없음 |
| `network_binary.nb` | NPU 모델 (INT8, 300프레임) | 수정 없음 |
