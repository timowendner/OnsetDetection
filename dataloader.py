import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import os


class OnsetDataset(Dataset):
    def __init__(self, config, device: torch.device):
        files = glob.glob(os.path.join(config.data_path, '*.wav'))
        waveforms = []
        for path in files:
            # load and normalize the audio file
            waveform, sr = torchaudio.load(path)
            waveform = waveform * 0.98 / torch.max(waveform)

            # load the onsets file
            with open(path[:-4] + '.onsets.gt', 'r') as f:
                text = f.readlines()

            # transform the text into float values
            onsets = []
            for i in text:
                onsets.append(float(i.replace('\n', '')))
            waveforms.append(waveform, sr, onsets)

        if len(waveforms) == 0:
            raise AttributeError('Data-path seems to be empty')

        self.waveforms = waveforms
        self.device = device

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform, sr, onsets = self.waveforms[idx]

        waveform = waveform.to(self.device)

        # Apply gain
        waveform = waveform * (1 - np.random.normal(0, 0.15)**2)

        # create a different starting point and roll the data over
        waveform = torch.roll(waveform, np.random.randint(waveform.shape[0]))

        # create the diffusion
        max_timestamp = self.diffusion.steps
        timestamp = np.random.randint(1, max_timestamp)
        x_t, noise = self.diffusion(waveform, timestamp)

        x_t = x_t.to(self.device)
        noise = noise.to(self.device)

        return x_t, noise, timestamp, label
