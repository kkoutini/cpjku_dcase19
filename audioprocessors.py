import torch

import numpy as np
import soundfile as sf
import librosa
from scipy import signal
import os


def processor_d18(file_path):
    n_fft = 2048  # 2048
    sr = 22050  # 22050  # 44100  # 32000
    mono = True  #
    log_spec = False
    n_mels = 256

    hop_length = 512
    fmax = None

    if mono:
        # this is the slowest part resampling
        sig, sr = librosa.load(file_path, sr=sr, mono=True)
        sig = sig[np.newaxis]
    else:
        sig, sr = librosa.load(file_path, sr=sr, mono=False)
        # sig, sf_sr = sf.read(file_path)
        # sig = np.transpose(sig, (1, 0))
        # sig = np.asarray([librosa.resample(s, sf_sr, sr) for s in sig])

    spectrograms = []
    for y in sig:

        # compute stftnp.asfortranarray(x)
        stft = librosa.stft(np.asfortranarray(y), n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                            pad_mode='reflect')

        # keep only amplitures
        stft = np.abs(stft)

        # spectrogram weighting
        if log_spec:
            stft = np.log10(stft + 1)
        else:
            freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
            stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=80.0)

        # apply mel filterbank
        spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

        # keep spectrogram
        spectrograms.append(np.asarray(spectrogram))

    spectrograms = np.asarray(spectrograms, dtype=np.float32)

    return spectrograms


def processor_d18_stereo(file_path):
    n_fft = 2048  # 2048
    sr = 22050  # 22050  # 44100  # 32000
    mono = False  #=
    log_spec = False
    n_mels = 256

    hop_length = 512
    fmax = None

    if mono:
        # this is the slowest part resampling
        sig, sr = librosa.load(file_path, sr=sr, mono=True)
        sig = sig[np.newaxis]
    else:
        sig, sr = librosa.load(file_path, sr=sr, mono=False)
        dpath, filename = os.path.split(file_path)
        #librosa.output.write_wav(dpath + "/../audio22k/" + filename, sig, sr)

        # sig, sf_sr = sf.read(file_path)
        # sig = np.transpose(sig, (1, 0))
        # sig = np.asarray([librosa.resample(s, sf_sr, sr) for s in sig])

    spectrograms = []
    for y in sig:

        # compute stft
        stft = librosa.stft(np.asfortranarray(y), n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                            pad_mode='reflect')

        # keep only amplitures
        stft = np.abs(stft)

        # spectrogram weighting
        if log_spec:
            stft = np.log10(stft + 1)
        else:
            freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
            stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=80.0)

        # apply mel filterbank
        spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

        # keep spectrogram
        spectrograms.append(np.asarray(spectrogram))

    spectrograms = np.asarray(spectrograms, dtype=np.float32)

    return spectrograms


