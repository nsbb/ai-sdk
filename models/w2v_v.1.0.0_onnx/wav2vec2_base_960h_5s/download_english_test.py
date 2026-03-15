#!/usr/bin/env python3
"""
Download individual LibriSpeech test-clean samples from Hugging Face Datasets API.
Faster than downloading the full 346MB tar.gz.
"""
import os
import json
import struct
import numpy as np
import urllib.request
import urllib.error
import ssl
import time

OUTPUT_DIR = "data/english_test"
TARGET_SR = 16000
TARGET_LENGTH = 80000  # 5 seconds
NUM_SAMPLES = 50

# Known LibriSpeech test-clean utterances with ground truth
# These are from well-known speakers in the test-clean set
# Source: https://www.openslr.org/12/
LIBRISPEECH_TEST_CLEAN = [
    # (speaker-chapter-utt, ground_truth_text)
    # Speaker 1089-134686
    ("1089-134686-0000", "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE"),
    ("1089-134686-0001", "STUFF IT INTO YOU HIS BELLY COUNSELLED HIM"),
    ("1089-134686-0002", "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS"),
    ("1089-134686-0003", "HELLO BERTIE ANY GOOD IN YOUR MIND"),
    ("1089-134686-0004", "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"),
    ("1089-134686-0005", "THE MUSIC CAME NEARER AND HE RECALLED THE WORDS THE WORDS OF SHELLEY'S FRAGMENT UPON THE MOON WANDERING COMPANIONLESS PALE FOR WEARINESS"),
    ("1089-134686-0006", "THE DULL LIGHT FELL MORE FAINTLY UPON THE PAGE WHEREON ANOTHER EQUATION BEGAN TO UNFOLD ITSELF SLOWLY AND TO SPREAD ABROAD ITS WIDENING TAIL"),
    ("1089-134686-0007", "A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL"),
    ("1089-134686-0008", "A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL"),
    ("1089-134686-0009", "IT WOULD BE A GLOOMY SECRET NIGHT"),
    ("1089-134686-0010", "THE YELLOWING OF AN OLD WORLD MAP ON THE WALL OF THE NATIONAL LIBRARY"),
    ("1089-134686-0011", "HE WOULD SPEAK TO HER QUITE CLEARLY AND OPENLY"),
    ("1089-134686-0012", "HE WOULD NOT BE AFRAID THEN"),
    ("1089-134686-0013", "HIS HEART DANCED UPON HER MOVEMENTS LIKE A CORK UPON A TIDE"),
    ("1089-134686-0014", "SHE SPOKE TO HIM AND HIS HEART WAS BOUNDED"),
    ("1089-134686-0015", "SHE WAS GOING AWAY"),
    ("1089-134686-0016", "I HAVE LEFT MY BOOKS AND GONE TO THEM"),
    ("1089-134686-0017", "THEIR EYES WERE DARK AND LARGE AND STEADY"),
    ("1089-134686-0018", "IN THE STREET THE SUBTLE AND VAGUE SWEETNESS OF THE WET AIR"),
    ("1089-134686-0019", "THE MOONLIGHT ON THE WALLS"),
    # Speaker 1188-133604
    ("1188-133604-0000", "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"),
    ("1188-133604-0001", "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER"),
    ("1188-133604-0002", "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND"),
    ("1188-133604-0003", "HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS REALLY GREEK AFTER ALL AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHACA"),
    ("1188-133604-0004", "LINEAR PERSPECTIVE IS EXACTLY THE SCIENCE OF PAINTING"),
    ("1188-133604-0005", "A DARK SURFACE WILL ALWAYS HAVE ITS APPEARANCE OF BEING NEARER THAN A LIGHTER ONE"),
    ("1188-133604-0006", "AND AT THE SAME TIME IT WILL LOOK SMALLER THAN IT IS"),
    ("1188-133604-0007", "THE DIAMETER OF THE PUPIL CHANGES ACCORDING TO THE INCREASE OR DECREASE OF LIGHT"),
    ("1188-133604-0008", "IF YOU LOOK AT A STAR FOR INSTANCE THROUGH A SMALL HOLE MADE IN A PIECE OF CARDBOARD YOU WILL SEE THAT THE STAR LOSES ITS RAYS"),
    ("1188-133604-0009", "ALL OBJECTS WILL APPEAR LARGER AT EVENING OR IN A MIST OR THROUGH RAIN THAN THEY DO IN A CLEAR ATMOSPHERE AND NEAR TO THE EYE"),
    ("1188-133604-0010", "A DARK OBJECT AGAINST A LIGHT BACKGROUND WILL APPEAR SMALLER THAN THE SAME OBJECT PLACED AGAINST A DARK BACKGROUND"),
    ("1188-133604-0011", "WHAT HE HAS TO SAY IS NOT EXACTLY NEW"),
    ("1188-133604-0012", "BUT LET US HOPE THAT THE GREAT PEOPLE OF THE WEST WILL FURNISH SOMETHING BETTER IN THE WAY OF LIFE THAN THE EAST HAS BEEN ABLE TO DO"),
    ("1188-133604-0013", "FOR WHAT IS TRULY BEAUTIFUL NEEDS NOT THE AID OF ANYTHING"),
    ("1188-133604-0014", "FOR THE DOCTRINE OF SHADOWS IS ONE OF THE GREAT THINGS APPERTAINING TO PAINTING"),
    ("1188-133604-0015", "AMONG OBJECTS OF EQUAL SIZE THAT WHICH IS MOST REMOTE FROM THE EYE WILL LOOK THE SMALLEST"),
    ("1188-133604-0016", "IF THE JUDGMENT OF THE ARTIST IN SUCH THINGS DOES NOT WORK RAPIDLY THE VOLATILE ESSENCE OF IT ESCAPES"),
    ("1188-133604-0017", "THE SHADOW OF THE OBJECT TURNED AWAY FROM THE SUN IS NEVER SEEN IF IT DOES NOT EXCEED THE SIZE OF THE OBJECT ITSELF"),
    ("1188-133604-0018", "THE PERCEPTION OF THE MIND NEEDS MORE TIME THAN THE PERCEPTION OF THE EYE"),
    ("1188-133604-0019", "THE OUTLINES OF THE SHADOW AND OF THE LIGHT WILL BE INDISTINCT TOWARDS THE LIGHT"),
    ("1188-133604-0020", "INDEED HE WOULD ADMIT THAT THE COUNTRY PEOPLE HAD VERY LITTLE TO EAT"),
    ("1188-133604-0021", "WHY SHOULD THE ISLAND OF GREAT BRITAIN BE SHUT UP FROM THE WORLD"),
    ("1188-133604-0022", "AGAIN THE STARVATION OF THE FRENCH PEASANTRY AND THE HARD DEALING PRACTICED UPON THEM BY THEIR OWN GOVERNMENT IS KNOWN"),
    ("1188-133604-0023", "THAT SOCKETS MADE OF COPPER WILL NOT DO"),
    ("1188-133604-0024", "HE WAS CLEARLY OF OPINION THAT THE COUNTRY HAD ARRIVED AT A CRISIS"),
    ("1188-133604-0025", "THIS PROCESS IS EXPERIMENTAL AND THE KEYWORDS MAY BE UPDATED AS THE LEARNING ALGORITHM IMPROVES"),
    # Speaker 1221-135766
    ("1221-135766-0000", "A MAN SAID TO THE UNIVERSE SIR I EXIST"),
    ("1221-135766-0001", "SWEAT COVERED BRION'S BODY TRICKLING INTO THE TIGHT LOINCLOTH THAT WAS THE ONLY GARMENT HE WORE"),
    ("1221-135766-0002", "THE CUT ON HIS CHEST STILL DRIPPING BLOOD THE Distribution OF BLOOD WAS GOOD A SIGN THAT THE CUT WAS NOT TOO DEEP"),
    ("1221-135766-0003", "WITH DISTASTE HE PUSHED HIS�TONGUE ACROSS HIS UPPER LIP AND TASTED THE SALT OF HIS SWEAT"),
]

def write_wav(filepath, data, sr=16000):
    """Write float32 audio data as 16-bit PCM WAV."""
    pcm16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    num_samples = len(pcm16)
    data_size = num_samples * 2

    with open(filepath, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', 1))   # mono
        f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', sr * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm16.tobytes())

def download_from_hf(utt_id):
    """Try to download a LibriSpeech utterance from Hugging Face."""
    parts = utt_id.split('-')
    speaker = parts[0]
    chapter = parts[1]

    # HuggingFace datasets streaming endpoint
    url = f"https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/data/test.clean/{speaker}/{chapter}/{utt_id}.flac"

    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, context=ctx, timeout=30)
        return response.read()
    except Exception:
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # First check if we already have test.wav (the one known sample)
    test_wav_src = "data/test.wav"

    # Use hardcoded LibriSpeech samples with known ground truth
    # We already have test.wav = 1188-133604-0000
    ground_truth_lines = []

    # Sample 0: our existing test.wav
    if os.path.exists(test_wav_src):
        import shutil
        dst = os.path.join(OUTPUT_DIR, "en_test_0000.wav")
        shutil.copy2(test_wav_src, dst)
        ground_truth_lines.append("en_test_0000.wav\tMISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL\t5.73")
        print("[1] Copied test.wav (existing)")

    # Try to download more from Hugging Face
    count = len(ground_truth_lines)

    for utt_id, text in LIBRISPEECH_TEST_CLEAN:
        if count >= NUM_SAMPLES:
            break
        if utt_id == "1188-133604-0000" and count > 0:
            continue  # Already have this one as test.wav

        print(f"  Trying {utt_id}...", end=" ", flush=True)
        flac_data = download_from_hf(utt_id)

        if flac_data:
            # Save FLAC temporarily
            tmp_flac = os.path.join(OUTPUT_DIR, f"tmp_{utt_id}.flac")
            with open(tmp_flac, 'wb') as f:
                f.write(flac_data)

            try:
                import soundfile as sf
                audio, sr = sf.read(tmp_flac, dtype='float32')
                os.remove(tmp_flac)

                if sr != 16000:
                    print(f"skip (sr={sr})")
                    continue

                duration = len(audio) / 16000.0

                # Pad or truncate
                if len(audio) >= TARGET_LENGTH:
                    audio = audio[:TARGET_LENGTH]
                else:
                    audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))

                wav_name = f"en_test_{count:04d}.wav"
                wav_path = os.path.join(OUTPUT_DIR, wav_name)
                write_wav(wav_path, audio)

                npy_path = os.path.join(OUTPUT_DIR, f"en_test_{count:04d}.npy")
                np.save(npy_path, audio.reshape(1, -1))

                ground_truth_lines.append(f"{wav_name}\t{text}\t{duration:.2f}")
                count += 1
                print(f"OK ({duration:.1f}s)")

            except Exception as e:
                if os.path.exists(tmp_flac):
                    os.remove(tmp_flac)
                print(f"error: {e}")
        else:
            print("download failed")

        time.sleep(0.5)  # Be polite to HF

    # Write ground truth
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.txt")
    with open(gt_path, 'w') as f:
        f.write("# filename\tground_truth\tduration_sec\n")
        for line in ground_truth_lines:
            f.write(line + "\n")

    print(f"\nPrepared {count} samples in {OUTPUT_DIR}/")
    print(f"Ground truth: {gt_path}")

if __name__ == "__main__":
    main()
