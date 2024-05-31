# MIT License
# 
# Copyright (c) 2022 MiscellaneousStuff
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Kara One helper library.

This library has been heavily adapated from:
https://github.com/wjbladek/SilentSpeechClassifier/blob/
b7f5ffd2a314ee14678a3e141d7addbd5320b5d0/SSC.py#L284"""


 # Addon
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mne.preprocessing import ICA
 #
import os
import glob
from time import time

import torch
import mne
import librosa
import scipy

import pandas as pd
import soundfile as sf
import scipy.io as sio

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from .extra import *
MAPPING = {
    '/diy/': 0,
    '/iy/': 1,
    '/m/': 2,
    '/n/': 3,
    '/piy/': 4,
    '/tiy/': 5,
    '/uw/': 6,
    'gnaw': 7,
    'knew': 8,
    'pat': 9,
    'pot': 10
}
PATH_INBETWEEN = ""
DEFAULT_CHANNELS = [
    'FC6', 'FT8', 'C5', 'CP3', 'P3', 'T7', 'CP5', 'C3', 'CP1', 'C4']
DROP_CHANNELS = [
    'CB1', 'CB2', 'VEO', 'HEO', 'EKG', 'EMG', 'Trigger']

FEIS_CHANNELS = [
    'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']


import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def get_eeg_delta_feats(eeg_channel, fs=256, stft=False, debug=False):
    """
    Extract features from EEG signal in the delta frequency band (0.5-4 Hz).

    Parameters:
    eeg_channel (numpy.ndarray): EEG signal
    fs (int): Sampling frequency (default: 256 Hz)
    stft (bool): Whether to include STFT features (default: False)
    debug (bool): Whether to plot the features (default: False)

    Returns:
    list: List of extracted features in the delta band
    """
    # Butterworth bandpass filter (0.5-4 Hz)
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 4 / nyquist
    sos = butter(4, [low, high], btype='band', output='sos')
    x = sosfilt(sos, eeg_channel)

    # Waveform Length
    w = double_average(x)

    # Root Mean Square (RMS)
    p = x - w
    rms = librosa.feature.rms(y=np.abs(p), frame_length=256, hop_length=64, center=False)
    rms = np.squeeze(rms, 0)

    # Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(p, frame_length=256, hop_length=64, center=False)
    zcr = np.squeeze(zcr, 0)

    # Wavelength
    wavelength = librosa.util.frame(w, frame_length=256, hop_length=64).mean(axis=0)

    channel_features = [wavelength, rms, zcr]

    if stft:
        # Short-Time Fourier Transform (STFT)
        stft = np.abs(librosa.stft(x, n_fft=512, hop_length=64, center=False))
        stft = stft.reshape(-1) 
        channel_features.append(stft)

    if debug:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
        axs[0].plot(x)
        axs[0].set_title('EEG Signal (Delta Band)')
        axs[1].plot(wavelength)
        axs[1].set_title('Wavelength')
        axs[2].plot(rms)
        axs[2].set_title('Root Mean Square (RMS)')
        axs[3].plot(zcr)
        axs[3].set_title('Zero Crossing Rate (ZCR)')
        # axs[4].plot(wamp)
        # axs[4].set_title('Willison Amplitude (WAMP)')
        if stft:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(stft.reshape(1, -1), origin='lower', aspect='auto', interpolation='nearest')
            ax.set_title('Short-Time Fourier Transform (STFT)')
        plt.tight_layout()
        plt.show()

    return channel_features

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(6, 0.5, 'highpass', fs=fs)
    signal = scipy.signal.filtfilt(b, a, signal)
    return scipy.signal.filtfilt(b, a, signal)
def extract_gamma(signal, fs):
    b, a = scipy.signal.butter(6, 31, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)
def extract_delta(signal, fs):
    b, a = scipy.signal.butter(6, 4, 'lowpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)
def extract_beta(signal, fs):
    b, a = scipy.signal.butter(6, 15, 'lowpass', fs=fs)
    epoch = scipy.signal.filtfilt(b, a, signal)
    b, a = scipy.signal.butter(6, 31, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, epoch)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result
def normalize_data(data):
    return data - np.mean(data, axis=1, keepdims=True)
def load_audio(fname,
               n_mel_channels=128,
               filter_length=512,
               win_length=432,
               hop_length=160):
    audio, r = sf.read(fname)
    if r != 16_000:
        audio = librosa.resample(audio, orig_sr = r, target_sr = 16_000)
        r = 16_000
    assert r == 16_000

    audio_features = librosa.feature.melspectrogram(
        y=audio,
        sr=r,
        n_mels=n_mel_channels,
        center=False,
        n_fft=filter_length,
        win_length=win_length,
        hop_length=hop_length).T
    audio_features = np.log(audio_features + 1e-5)

    return audio, audio_features.astype(np.float32)

def read_eeg(eeg_path, channels_only=[], drop_channels=DROP_CHANNELS):
    print("PATH:", eeg_path)
    eeg_raw = mne.io.read_raw_cnt(eeg_path, preload=True) # .load_data()
    eeg_raw.drop_channels(drop_channels)
    if channels_only:
        eeg_raw.pick_channels(channels_only)
    print("EEG_RAW:", eeg_raw)
    return eeg_raw

def get_semg_feats_orig(eeg_data, hop_length=6, frame_length=16, stft=False, debug=False):
    xs = eeg_data - eeg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(eeg_data.shape[1]):
        x = xs[:, i]
        print(len(x))
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(w, frame_length=frame_length, hop_length=hop_length).mean(axis=0)
        p_w = librosa.feature.rms(w, frame_length=frame_length, hop_length=hop_length, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(r, frame_length=frame_length, hop_length=hop_length, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=frame_length, hop_length=hop_length, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=frame_length, hop_length=hop_length).mean(axis=0)

        if stft:
            s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=frame_length, hop_length=hop_length, center=False))

        if debug:
            plt.subplot(7,1,1)
            plt.plot(x)
            plt.subplot(7,1,2)
            plt.plot(w_h)
            plt.subplot(7,1,3)
            plt.plot(p_w)
            plt.subplot(7,1,4)
            plt.plot(p_r)
            plt.subplot(7,1,5)
            plt.plot(z_p)
            plt.subplot(7,1,6)
            plt.plot(r_h)

            plt.subplot(7,1,7)
            plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')

            plt.show()

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        if stft:
            frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)

import numpy as np
import librosa
from scipy.stats import entropy

def get_semg_feats(eeg_channel, stft=False, debug=False):
    xs = eeg_channel - eeg_channel.mean(axis=0, keepdims=True)
    x = xs
    


    w = double_average(x)
    p = x - w
    r = np.abs(p)

    w_frames = librosa.util.frame(w, frame_length=128, hop_length=32)
    r_frames = librosa.util.frame(r, frame_length=128, hop_length=32)

    w_h = w_frames.mean(axis=0)
    p_w = librosa.feature.rms(y=p, frame_length=128, hop_length=32, center=False)
    p_w = np.squeeze(p_w, 0)
    p_r = librosa.feature.rms(y=r, frame_length=128, hop_length=32, center=False)
    p_r = np.squeeze(p_r, 0)
    z_p = librosa.feature.zero_crossing_rate(p, frame_length=128, hop_length=32, center=False)
    z_p = np.squeeze(z_p, 0)
    r_h = r_frames.mean(axis=0)

    channel_features = []
    frames= librosa.util.frame(x, frame_length=128, hop_length=32)
    mean = frames.mean(axis=0)
    median = np.median(frames, axis=0)
    std = frames.std(axis=0)
    var = frames.var(axis=0)
    max_val = frames.max(axis=0)
    min_val = frames.min(axis=0)
    max_min = max_val - min_val
    sum_val = frames.sum(axis=0)
    spectral_entropy = [entropy(frame, base=2) for frame in frames.T]
    energy = np.sum(frames**2, axis=0)
    kurtosis = np.apply_along_axis(lambda x: scipy.stats.kurtosis(x, bias=False), 0, frames)
    skewness = np.apply_along_axis(lambda x: scipy.stats.skew(x, bias=False), 0, frames)
    channel_features.extend([mean, median, std, var, max_val, min_val, max_min, sum_val, spectral_entropy,energy, kurtosis, skewness])
    
    channel_features.extend([w_h, p_w, p_r, z_p, r_h])
    if stft:
        s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))
        channel_features.append(s)

    if debug:
        plt.subplot(7, 1, 1)
        plt.plot(x)
        plt.subplot(7, 1, 2)
        plt.plot(w_h)
        plt.subplot(7, 1, 3)
        plt.plot(p_w)
        plt.subplot(7, 1, 4)
        plt.plot(p_r)
        plt.subplot(7, 1, 5)
        plt.plot(z_p)
        plt.subplot(7, 1, 6)
        plt.plot(r_h)
        if stft:
            plt.subplot(7, 1, 7)
            plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')
        plt.show()

    return channel_features

def calculate_features(eeg_data, epoch_inds, prompts, condition_inds, prompts_list, \
                       end_idx=0, start_idx=0, powerline_freq=60, raw_only=False):
    
    offset  = int(eeg_data.info["sfreq"] / 2)
    X_raw   = []
    X_feats = []
    end_idx = end_idx if end_idx else len(prompts["prompts"][0])

    max_raw_size = 0
  
    for i, prompt in enumerate(prompts["prompts"][0][start_idx:end_idx]):
        if prompt[0] in prompts_list:
            t0 = time()
            start = epoch_inds[condition_inds][0][i][0][0] + offset
            end   = epoch_inds[condition_inds][0][i][0][1]

            delta_raw  = []

        
            for idx, ch in enumerate(eeg_data):
                epoch = eeg_data[idx][0][0][start:end]
              
                if epoch.shape[0] > max_raw_size:
                    max_raw_size = epoch.shape[0]
                epoch     = subsample(epoch, 600, 1000)
                delta = get_semg_feats(epoch,stft=False, debug=False)
                delta_raw.append(delta)
            g,b,d = 0,0,np.stack(delta_raw)
            
            channel_set = []
            if not raw_only:
                for sequence in d:

                    channel_set.extend(sequence)

          
            X_feats.append(channel_set)
            X_raw.append([])
            print("Calc: %0.3fs" % (time() - t0), i, prompt)
        
      
    return X_raw, X_feats




class KaraOneDataset(torch.utils.data.Dataset):
    num_features = 62 * 5 # (electrodes * features_per_electrode)

    def __init__(
            self,
            root_dir,
            pts=("MM05",),
            raw_only=False,
            start_idx=0,
            end_idx=0,
            n_mel_channels=128,
            eeg_types=[],
            channels_only=[]):
        
        self.root_dir = root_dir
        self.pts = pts
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.n_mel_channels = n_mel_channels
        self.eeg_types = eeg_types
        self.label = []
        self.transposed_tensors = []
        self.min = 0


        print("init() end_idx:", end_idx)

        info_dir = \
            os.path.join(
                root_dir,
                PATH_INBETWEEN,
                pts[0],
                f"kinect_data\\")
        
        with open(os.path.join(info_dir, f"{pts[0]}_p.txt")) as f:
            self.labels = f.read().split("\n")
            self.ids = range(len(self.labels))
            self.audios = [os.path.join(info_dir, f"{id_}.wav")
                            for id_ in self.ids]
        
        base_dir = os.path.join(
            self.root_dir,
            PATH_INBETWEEN,
            self.pts[0])

        for f in glob.glob(
            os.path.join(f"{base_dir}", "all_features_simple.mat")):
            prompts_to_extract = sio.loadmat(f)
        
        self.prompts = prompts = prompts_to_extract['all_features'][0,0]

        variable_names = ()
        for eeg_type in eeg_types:
            if eeg_type == "vocal":
                variable_names += ("speaking_inds",)
            elif eeg_type == "imagined":
                variable_names += ("thinking_inds",)

        for f in glob.glob(
            os.path.join(f"{base_dir}", "epoch_inds.mat")):
            self.epoch_inds = epoch_inds = \
                sio.loadmat(f, variable_names=variable_names)
   
        
        for f in glob.glob(
            os.path.join(f"{base_dir}", "*.cnt")):
            eeg_data = read_eeg(f, channels_only)
        
        self.eeg_data = eeg_data

        t0 = time()

        end_idx = end_idx if end_idx else len(self.prompts["prompts"][0])

        print("init-2() end_idx:", end_idx)

        self.Y_s = Y_s = self.prompts["prompts"][0][start_idx:end_idx]

 
        raw = eeg_data.pick_channels(DEFAULT_CHANNELS)
        n_time_samps = raw.n_times
        time_secs = raw.times
        ch_names = raw.ch_names
        n_chan = len(ch_names) 
        print(
            "the (cropped) sample data object has {} time samples and {} channels."
            "".format(n_time_samps, n_chan)
        )
        print("The last time sample is at {} seconds.".format(time_secs[-1]))
        print("The first few channel names are {}.".format(", ".join(ch_names)))
        print() 
        print("bad channels:", raw.info["bads"]) 
        print(raw.info["sfreq"], "Hz")  
        print(raw.info["description"], "\n")  

        order = 6
        freq_cutoff = 1
        transition_bandwidth = 0.7
        raw.filter(l_freq=freq_cutoff, h_freq=100, method='iir', iir_params=dict(order=order, ftype='butter', output='sos', bw=transition_bandwidth)) #1/3 from 1000

        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()
        notch_freqs = [60, 120, 180, 240, 300,360,420,480]
        raw.notch_filter(freqs=notch_freqs)
        if "imagined" in eeg_types:
            self.X_s_active_raw, self.X_s_active_feats = \
                calculate_features(
                    raw,
                    epoch_inds,
                    prompts,
                    "thinking_inds",
                    Y_s,
                    start_idx=start_idx,
                    raw_only=raw_only,
                    end_idx=end_idx)
            self.X_s_active_raw = np.asarray(self.X_s_active_raw)
            self.X_s_active_feats = np.asarray(self.X_s_active_feats)

        self.Y_s = np.hstack(self.Y_s)
        print("Calc: %0.3fs" % (time() - t0))
    
    def __getitem__(self, i):
        n_mel_channels = self.n_mel_channels
        eeg_types = self.eeg_types

        audio_raw, audio_features = \
            load_audio(self.audios[self.start_idx:self.end_idx][i],
                       n_mel_channels)

        data = {
            "label":       self.Y_s[i],
            "audio_raw":   audio_raw,
            "audio_feats": audio_features
        }
        
        if "imagined" in eeg_types:
            cur = {
                "eeg_active_raw":   np.transpose(self.X_s_active_raw[i]),
                "eeg_active_feats": np.transpose(self.X_s_active_feats[i]),
            }
            data = {**data, **cur}

        if "vocal" in eeg_types:
            print("[i] X_s_vocal_raw.shape:", self.X_s_vocal_raw.shape)
            print("[i] X_s_vocal_feats.shape:", self.X_s_vocal_feats.shape)
            cur = {
                "eeg_vocal_raw":   np.transpose(self.X_s_vocal_raw[i], (1, 0)),
                "eeg_vocal_feats": np.transpose(self.X_s_vocal_feats[i], (1, 0)),
            }
            data = {**data, **cur}
            
        return data

    def __len__(self):
        return len(self.Y_s)
    
    def get_data(self):
        label = [MAPPING[phoneme] for phoneme in  self.Y_s]
        print(self.X_s_active_feats.shape)
        return self.X_s_active_feats,label