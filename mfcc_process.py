# tmux attach -s face
import os
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
import librosa
from tqdm import tqdm, trange

from faceio import get_date_directories

REQUIRED_FRAMERATE = 48000

frames_num_sum = 0
audio_perfs = get_date_directories('bili_audio',RATIO=1)

# emission audio id=425: /home/yunyangli/CMLR/audio_feat/s10/20150802.npy
with trange(0, len(audio_perfs)) as t:
    for day_idx in t:
        day = audio_perfs[day_idx]

        sources, target = day['sources'], day['target']
        t.set_description(target)

        mfcc_combined = None
        total_slices = 0

        for fname, slice_num in sources:
            sample_rate, data = wavfile.read(fname)
            data = data[:, 0]
            if np.mean(data) == 0:
                data = data[:, 1]
            if sample_rate != REQUIRED_FRAMERATE:
                nroutsamples = round(len(data) * REQUIRED_FRAMERATE/sample_rate)
                signal = sps.resample(data, nroutsamples)
            else:
                signal = data.astype(np.float64)
            sample_rate = REQUIRED_FRAMERATE
            
            # https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
            pre_emphasis = 0.97
            emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

            frame_size = 0.1
            frame_stride = 0.04
            frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
            signal_length = len(emphasized_signal)
            frame_length = int(round(frame_length))
            frame_step = int(round(frame_step))
            # num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
            num_frames = slice_num
    
            # pad_signal_length = num_frames * frame_step + frame_length
            z = np.zeros((int((frame_size-frame_stride)*sample_rate/2)))
            pad_signal = np.append(z, np.append(emphasized_signal, z)) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

            indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
            frames = pad_signal[indices.astype(np.int32, copy=False)]

            for i in range(len(frames)):
                import scipy
                from math import sqrt
                x = frames[i, :]
                x /= np.max(np.abs(x)) * sqrt(2)
                x *= np.hamming(frame_length)
                
                mfcc = librosa.feature.mfcc(y=x, sr=REQUIRED_FRAMERATE, n_mfcc=20)
                mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

                if mfcc_combined is not None:
                    mfcc_combined = np.concatenate((mfcc_combined, mfcc), axis=1)
                else:
                    mfcc_combined = mfcc

            total_slices += slice_num

        t.set_postfix(mfcc_combined=mfcc_combined.shape,
                      current_slices=total_slices)
        np.save(target, mfcc_combined)