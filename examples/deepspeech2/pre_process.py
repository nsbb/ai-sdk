import os
import sys
import numpy as np
import librosa
from collections import namedtuple

AUDIO_SR = 16000  # sampling rate

def tf_wave_to_stft(wave):
    sample_rate = 16000
    window_size = 0.02
    window_stride = 0.01
    window = 'hamming'
    normalize = True

    n_fft = 320  # int(sample_rate * window_size)
    win_length = n_fft
    hop_length = 160  # int(sample_rate * window_stride)
    # STFT
    D = librosa.stft(wave, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)

    spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)
    return spect


def _get_files_stft_librosa(wav_filenames):
    # print('Processing stft_librosa...')
    mfccs = []
    lens = []
    n_context = 0
    n_input = 161

    audio_fname = wav_filenames
    wave = librosa.core.load(audio_fname, sr=AUDIO_SR)[0]

    mfcc = tf_wave_to_stft(wave).T

    feature_len = (len(mfcc) + 6 + 5) // 6 * 6

    if len(mfcc) < feature_len:
        needlen = feature_len - len(mfcc)
        INPUT_DIM = n_input
        a = np.array(([[0 for x in range(INPUT_DIM * (2 * n_context + 1))] for y in range(needlen)]))
        mfcc = np.concatenate((mfcc, a))
    mfccs.append(mfcc)
    lens.append(len(mfcc))
    a_mfccs = np.array(mfccs)
    a_lens = np.array(lens)
    return a_mfccs, a_lens

def pad(tensor, h=756):
    shape = tensor.shape
    if shape[1] > 756:
        return tensor[:,0:756,:]
    elif shape[1] < 756:
        npad = ((0, 0), (0, 756 - shape[1]), (0, 0))
        return np.pad(tensor, pad_width=npad, mode='constant', constant_values=0)
    return tensor

def parse_args():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('', '--wav', dest="wav_file", help="waf file name")

    (options, args) = parser.parse_args()
    if options.wav_file:
        return options

    parser.print_help()
    sys.exit(-1)

if __name__ == '__main__':
    options = parse_args()
    if os.path.exists(options.wav_file):
        print('Processing {}...'.format(options.wav_file))
        m, _ = _get_files_stft_librosa(options.wav_file)
        tensor = pad(m)
        tensor = np.transpose(tensor, [1, 2, 0])
        extension = '_' + '_'.join([str(i) for i in tensor.shape]) + '.tensor'
        tensor_file = options.wav_file.replace('.wav', extension)
        print('Save to {}'.format(tensor_file))
        tensor.tofile(tensor_file, '\n')

        tensor_file = tensor_file.replace('.tensor', '.npy')
        print('Save to {}'.format(tensor_file))
        np.save(tensor_file, tensor)
    else:
        print('{} not found!'.format(options.wav_file))
        sys.exit(-1)
