import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import torchaudio
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from src.data.transforms import Crop, StatefulRandomHorizontalFlip
from PIL import Image
import librosa
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import glob
from scipy import signal
import torchvision
from torch.autograd import Variable
from librosa.filters import mel as librosa_mel_fn
from src.data.audio_processing import dynamic_range_compression, dynamic_range_decompression, griffin_lim
from src.data.stft import STFT
import math

log1e5 = math.log(1e-5)


class collater():

    def __init__(
        self,
        duration=3,
        sample_rate=16000,
        filter_length=640,
        hop_length=160,
        win_length=640,
        fps=25,
    ):
        self.duration = duration
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.fps = fps

        pass

    def __call__(self, batches):
        spec_per_second = math.ceil(self.sample_rate / self.hop_length)
        melspec_b = [batch[0][:int(spec_per_second * self.duration)] for batch in batches if batch[0] is not None]
        spec_b = [batch[1][:int(spec_per_second * self.duration)] for batch in batches if batch[1] is not None]
        vid_b = [batch[2][:int(self.duration * self.fps)] for batch in batches if batch[2] is not None]
        num_v_frames_b = [torch.tensor(int(self.duration * self.fps)) for batch in batches if batch[3] is not None]
        audio_b = [batch[4][:int(self.duration * self.sample_rate)] for batch in batches if batch[4] is not None]
        num_a_frames_b = [
            torch.tensor(int(spec_per_second * self.duration)) for batch in batches if batch[5] is not None
        ]

        pad_m = lambda melspec: [0, 0, 0, int(self.duration * spec_per_second) - melspec.shape[0]]
        pad_s = lambda spec: [0, 0, 0, int(self.duration * spec_per_second) - spec.shape[0]]
        pad_v = lambda vid: [0, 0, 0, 0, 0, int(self.duration * self.fps) - vid.shape[0]]
        pad_a = lambda audio: [0, 0, int(self.duration * self.sample_rate) - audio.shape[0]]

        try:
            melspec_b = torch.stack([F.pad(melspec, pad=pad_m(melspec)) for melspec in melspec_b])
            spec_b = torch.stack([F.pad(spec, pad=pad_s(spec)) for spec in spec_b])
            vid_b = torch.stack([F.pad(vid, pad=pad_v(vid)) for vid in vid_b])
            num_v_frames_b = torch.stack(num_v_frames_b)
            audio_b = torch.stack([F.pad(audio, pad=pad_a(audio)) for audio in audio_b])
            num_a_frames_b = torch.stack(num_a_frames_b)
        except Exception as e:
            for audio in audio_b:
                print(audio.shape)
            print(e)
            exit(0)

        if len(batches[0] == 6):
            return melspec_b, spec_b, vid_b, num_v_frames_b, audio_b, num_a_frames_b
        else:
            extra = [batch[6:] for batch in batches]
            extra = default_collate(extra)
            return [melspec_b, spec_b, vid_b, num_v_frames_b, audio_b, num_a_frames_b] + extra


class LRWKMultiDataset(Dataset):

    def __init__(
        self,
        data_dir,
        mode,
        max_v_timesteps=155,
        window_size=40,
        sample_rate=16000,
        hop_length=160,
        filter_length=640,
        win_length=640,
        subject=None,
        augmentations=False,
        num_mel_bins=80,
        fast_validate=False,
        fps=25,
    ):
        assert mode in ['train', 'test', 'val']
        self.data_dir = data_dir
        self.mode = mode
        self.sample_window = True if mode == 'train' else False
        self.fast_validate = fast_validate
        # self.max_v_timesteps = window_size if self.sample_window else max_v_timesteps
        self.max_v_timesteps = max_v_timesteps
        self.max_a_samples = int(self.max_v_timesteps * sample_rate / fps)
        self.window_size = window_size
        self.augmentations = augmentations if mode == 'train' else False
        self.num_mel_bins = num_mel_bins
        self.file_paths = self.build_file_list(data_dir, mode, subject)
        self.f_min = 55.
        self.f_max = 7500.
        self.sample_rate = sample_rate
        self.fps = fps
        self.hop_length = hop_length
        self.filter_length = filter_length
        self.win_length = win_length
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 n_mel_channels=num_mel_bins,
                                 sampling_rate=sample_rate,
                                 mel_fmin=self.f_min,
                                 mel_fmax=self.f_max)
        if self.sample_window:
            print(f'max_v_timesteps:{self.window_size} max_a_samples:{int(self.window_size * sample_rate / fps)} ')
        else:
            print(f'max_v_timesteps:{self.max_v_timesteps} max_a_samples:{self.max_a_samples} ')

    def load_video(self, root, vid_st_ed):
        transform = transforms.Compose([
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
        ])
        file_path = os.path.join(root, vid_st_ed.split('_')[0])
        ss = int((float(vid_st_ed.split('_')[1]) - 0.2) * self.fps) + 1
        ed = round((float(vid_st_ed.split('_')[2]) - float(vid_st_ed.split('_')[1]) + 0.4) * self.fps) + ss
        vid = []
        if not os.path.exists(os.path.join(file_path, f'{ss}.jpg')):
            return torch.zeros([1, 112, 112, 3])
        for i in range(ss, ed, 1):
            image_path = os.path.join(file_path, str(i)+'.jpg')
            if os.path.exists(image_path):
                image = transform(Image.open(image_path))
                image = image.transpose(0, 1).transpose(1, 2)
            # print(image.shape)
            vid.append(image)
        vid = torch.stack(vid)
        # print(vid.shape)
        return vid
        # pass

    def build_file_list(self, datapath, mode, subject):
        file_list = []
        self.vid2aid = {}
        with open('/work104/cchen/data/audio-visual/LRW1000/LRW1000_Public/all_audio_video.txt') as fp:
            for line in fp.readlines():
                vid, aid, _, _, st, ed = line.strip('\n').split(',')
                self.vid2aid[f'{vid}_{float(st)}_{float(ed)}'] = aid
        with open(f'/work104/cchen/data/audio-visual/LRW1000/LRW1000_Public/filelist/{mode}.txt', 'r') as f:
            for line in f.readlines():
                vid, _, _, st, ed = line.strip('\n').split(',')
                file_list.append(f'{vid}_{float(st)}_{float(ed)}')
        return file_list

    def build_tensor(self, frames):
        if self.augmentations:
            augmentations1 = transforms.Compose([StatefulRandomHorizontalFlip(0.5)])
        else:
            augmentations1 = transforms.Compose([])
        # crop = [52, 84, 170, 202]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            # Crop(crop),
            transforms.Resize([112, 112]),
            augmentations1,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.4136, 0.1700),
        ])

        max_v_timesteps = self.window_size if self.sample_window else self.max_v_timesteps
        temporalVolume = torch.zeros(max_v_timesteps, 1, 112, 112)
        for i, frame in enumerate(frames):
            if max_v_timesteps <= i:
                break
            temporalVolume[i] = transform(frame)

        ### Random Erasing ###
        if self.augmentations:
            x_s, y_s = [random.randint(-10, 66) for _ in range(2)]  # starting point
            temporalVolume[:, :,
                           np.maximum(0, y_s):np.minimum(112, y_s + 56),
                           np.maximum(0, x_s):np.minimum(112, x_s + 56)] = 0.

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, T, H, W)
        return temporalVolume

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        info = {}
        vid = self.load_video('/work104/cchen/data/audio-visual/LRW1000/LRW1000_Public/lip_images/', file_path)
        # print(vid.shape)
        audio, info['audio_fps'] = librosa.load(
            os.path.join('/work104/cchen/data/audio-visual/LRW1000/LRW1000_Public/audio/', self.vid2aid[file_path]+'.wav'),
            sr=self.sample_rate,
        )
        audio = torch.FloatTensor(audio).unsqueeze(0)
        # print(vid.shape)
        # print(audio.shape)

        if not 'video_fps' in info:
            info['video_fps'] = self.fps
            info['audio_fps'] = self.sample_rate
            
        if vid.size(0) < 5 or audio.size(1) < 5:
            vid = torch.zeros([self.max_v_timesteps, 112, 112, 3])
            audio = torch.zeros([1, self.max_a_samples])
        elif vid.size(0) < self.max_v_timesteps or audio.size(1) < self.max_a_samples:
            p_vid = torch.zeros([self.max_v_timesteps, 112, 112, 3])
            p_aud = torch.zeros([1, self.max_a_samples])
            if vid.size(0) < self.max_v_timesteps:
                p_vid[:vid.size(0), :, :, :] = vid[:vid.size(0), :, :, :]
            else:
                p_vid[:self.max_v_timesteps, :, :, :] = vid[:self.max_v_timesteps, :, :, :]
            if audio.size(1) < self.max_a_samples:
                p_aud[:, :audio.size(1)] = audio
            else:
                p_aud[:, :self.max_a_samples] = audio[:, :self.max_a_samples]
            vid = p_vid
            audio = p_aud

        vid = vid[:self.max_v_timesteps, :, :, :]
        audio = audio[:, :self.max_a_samples]

        ## Audio ##
        if not torch.abs(audio).max() < 1e-5:
            aud = audio / torch.abs(audio).max() * 0.9
            aud = torch.FloatTensor(self.preemphasize(aud.squeeze(0))).unsqueeze(0)
            aud = torch.clamp(aud, min=-1, max=1)
        else:
            aud = torch.FloatTensor(self.preemphasize(audio.squeeze(0))).unsqueeze(0)
            aud = torch.clamp(aud, min=-1, max=1)
        melspec, spec = self.stft.mel_spectrogram(aud)

        ## Video ##
        vid = vid.permute(0, 3, 1, 2)  # T C H W

        if self.sample_window:
            try:
                vid, melspec, spec, audio = self.extract_window(vid, melspec, spec, audio, info)
            except Exception as e:
                print(file_path)
                exit(-1)

        num_v_frames = vid.size(0)
        vid = self.build_tensor(vid)

        melspec = self.normalize(melspec)

        max_v_timesteps = self.window_size if self.sample_window else self.max_v_timesteps
        num_a_frames = melspec.size(2)
        melspec = nn.ConstantPad2d((0,max_v_timesteps * 4 - num_a_frames, 0, 0), 0.0)(melspec)
        spec = nn.ConstantPad2d((0,max_v_timesteps * 4 - num_a_frames, 0, 0), 0.0)(spec)

        if not self.sample_window:
            audio = audio[:, :int(self.max_v_timesteps * 4 * self.hop_length)]
            audio = torch.cat([
                audio,
                torch.zeros([1, int(self.max_v_timesteps / info['video_fps'] * info['audio_fps'] - aud.size(1))])
            ], 1)

        # print(melspec.shape)
        # print(spec.shape)
        # print(vid.shape)
        # print(audio.shape)
        # exit()

        if self.mode == 'test':
            return melspec, spec, vid, num_v_frames, audio.squeeze(0), num_a_frames, self.vid2aid[file_path]
        else:
            return melspec, spec, vid, num_v_frames, audio.squeeze(0), num_a_frames

    def extract_window(self, vid, mel, spec, aud, info):
        # vid : T,C,H,W
        vid_2_aud = info['audio_fps'] / info['video_fps'] / self.hop_length

        st_fr = random.randint(0, vid.size(0) - self.window_size)
        vid = vid[st_fr:st_fr + self.window_size]

        st_mel_fr = int(st_fr * vid_2_aud)
        mel_window_size = int(self.window_size * vid_2_aud)

        mel = mel[:, :, st_mel_fr:st_mel_fr + mel_window_size]
        spec = spec[:, :, st_mel_fr:st_mel_fr + mel_window_size]

        aud = aud[:, st_mel_fr * self.hop_length:st_mel_fr * self.hop_length + mel_window_size * self.hop_length]
        aud = torch.cat(
            [aud, torch.zeros([1, int(self.window_size / info['video_fps'] * info['audio_fps'] - aud.size(1))])], 1)

        return vid, mel, spec, aud

    def inverse_mel(self, mel, stft):
        if len(mel.size()) < 4:
            mel = mel.unsqueeze(0)  # B,1,80,T

        mel = self.denormalize(mel)
        mel = stft.spectral_de_normalize(mel)
        mel = mel.transpose(2, 3).contiguous()  # B,80,T --> B,T,80
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.matmul(mel, stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(2, 3).squeeze(1)  # B,1,F,T
        spec_from_mel = spec_from_mel * spec_from_mel_scaling

        wav = griffin_lim(spec_from_mel, stft.stft_fn, 60).squeeze(1)  # B,L
        wav = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
        wavs = []
        for w in wav:
            w = self.deemphasize(w)
            wavs += [w]
        wavs = np.array(wavs)
        wavs = np.clip(wavs, -1, 1)
        return wavs

    def inverse_spec(self, spec, stft):
        if len(spec.size()) < 4:
            spec = spec.unsqueeze(0)  # B,1,321,T

        wav = griffin_lim(spec.squeeze(1), stft.stft_fn, 60).squeeze(1)  # B,L
        wav = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
        wavs = []
        for w in wav:
            w = self.deemphasize(w)
            wavs += [w]
        wavs = np.array(wavs)
        wavs = np.clip(wavs, -1, 1)
        return wavs

    def preemphasize(self, aud):
        aud = signal.lfilter([1, -0.97], [1], aud)
        return aud

    def deemphasize(self, aud):
        aud = signal.lfilter([1], [1, -0.97], aud)
        return aud

    def normalize(self, melspec):
        melspec = ((melspec - log1e5) / (-log1e5 / 2)) - 1  #0~2 --> -1~1
        return melspec

    def denormalize(self, melspec):
        melspec = ((melspec + 1) * (-log1e5 / 2)) + log1e5
        return melspec

    def audio_preprocessing(self, aud):
        fc = self.f_min
        w = fc / (self.sample_rate / 2)
        b, a = signal.butter(7, w, 'high')
        aud = aud.squeeze(0).numpy()
        aud = signal.filtfilt(b, a, aud)
        return torch.tensor(aud.copy()).unsqueeze(0)

    def plot_spectrogram_to_numpy(self, mels):
        fig, ax = plt.subplots(figsize=(15, 4))
        im = ax.imshow(np.squeeze(mels, 0), aspect="auto", origin="lower", interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        data = self.save_figure_to_numpy(fig)
        plt.close()
        return data

    def save_figure_to_numpy(self, fig):
        # save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        return data.transpose(2, 0, 1)


class TacotronSTFT(torch.nn.Module):

    def __init__(self,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mel_channels=80,
                 sampling_rate=22050,
                 mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
        )
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output, magnitudes
