#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import cv2
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#%%
base_dir = './CelebAMask-HQ'
img_list = sorted([x for x in os.listdir(base_dir + '/CelebA-HQ-img') if x != '.DS_Store'])
anno_base = sorted([x for x in os.listdir(base_dir + '/CelebAMask-HQ-mask-anno') if x != '.DS_Store'])
anno_list = []
for a in anno_base:
    anno_list += os.listdir(base_dir + f'/CelebAMask-HQ-mask-anno/{a}')
with open(base_dir + '/CelebAMask-HQ-attribute-anno.txt', 'r') as f:
    labels = f.readlines()
#%%
lables = pd.DataFrame(
    [x.split() for x in labels[2:]],
    columns=['file'] + labels[1].split()
)

smile = lables[['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Chubby', 'Narrow_Eyes']]
smile = smile.astype(float)
smile[smile == -1] = 0

attractive = lables[['Young', 'Male', 'Bags_Under_Eyes', 'Chubby', 'Heavy_Makeup', 'Receding_Hairline']]
attractive = attractive.astype(float)
attractive[attractive == -1] = 0
#%%
atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

smile_seg_map = [['skin'],  # High_Cheekbones
    ['mouth', 'u_lip', 'l_lip'], # Mouth_Slightly_Open
    ['skin', 'nose', 'neck', 'neck_l'], # Chubby
    ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g'], # Narrow_Eyes
    ['l_ear', 'r_ear', 'ear_r', 'cloth', 'hair', 'hat']] # etc

attractive_seg_map = [['l_eye', 'r_eye', 'eye_g'],  # Bags_Under_Eyes
    ['skin', 'nose', 'neck', 'neck_l'], # Chubby
    ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'u_lip', 'l_lip'], # Heavy_Makeup
    ['hair', 'hat'], # Receding_Hairline
    ['mouth', 'l_ear', 'r_ear', 'ear_r', 'cloth', 'hair', 'hat']] # etc
#%%
tmp = set()
for x in smile_seg_map:
    tmp = tmp.union(set(x))
assert tmp == set(atts)

tmp = set()
for x in attractive_seg_map:
    tmp = tmp.union(set(x))
assert tmp == set(atts)
#%%
img_size = 128

idx = int(img_list[0].split('.')[0])

img = cv2.imread(base_dir + '/CelebA-HQ-img/' + img_list[0])
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
img = cv2.resize(img, (img_size, img_size)) / 255

b = idx // 2000
filenames = []
for seg in smile_seg_map:
    filenames.append([base_dir + f'/CelebAMask-HQ-mask-anno/{b}/' + f'{idx:05d}_{a}.png' for a in seg])

seg_imgs = []
for fs in filenames:
    seg_imgs.append(
        np.concatenate(
            [cv2.resize(cv2.imread(f), (img_size, img_size)) 
             for f in fs if os.path.exists(f)], axis=-1
            ).sum(axis=-1, keepdims=True))
#%%
np.concatenate([img] + seg_imgs, axis=-1)
#%%
import random
import os

import torch

from modules.utils import (
    load_filepaths_and_text,
    load_wav_to_torch,
    intersperse
)

from text import (
    text_to_sequence, 
    cleaned_text_to_sequence
)
#%%
# 이건 global 변수인데 아직까지 사용되지는 않아서 정확히 목적은 모르겠음...
hann_window = {}
#%%
# torch로 변환된 wav 파일을 padding 후 STFT 형식으로 변환 (spectogram으로 변환)
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec
#%%
# (text, spectogram, wav)의 쌍을 생성하는 데이터셋
class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, config, seed):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners  = config["text_cleaners"]
        self.max_wav_value  = config["max_wav_value"]
        self.sampling_rate  = config["sampling_rate"]
        self.filter_length  = config["filter_length"] 
        self.hop_length     = config["hop_length"] 
        self.win_length     = config["win_length"]
        self.sampling_rate  = config["sampling_rate"]
        self.max_text_len   = config["max_text_len"]
        self.min_text_len   = config["min_text_len"]
        self.cleaned_text   = config["cleaned_text"]
        self.add_blank      = config["add_blank"]

        random.seed(seed)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)
#%%