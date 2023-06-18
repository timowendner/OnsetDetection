import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import os


class OnsetDataset(Dataset):
    def __init__(self, config, device: torch.device, data_path=None):
        self.device = device
        self.length = config.data_length
        self.sigma = config.data_targetSD

        files = config.data_path if data_path is None else data_path
        files = glob.glob(os.path.join(files, '*.wav'))
        waveforms = []
        for path in files:
            # load and normalize the audio file
            waveform, sr = torchaudio.load(path)
            waveform = waveform * 0.98 / torch.max(waveform)
            padding = 2**int(
                np.ceil(np.log2(waveform.shape[1]))) - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

            # load the onsets file
            with open(path[:-4] + '.onsets.gt', 'r') as f:
                text = f.readlines()

            # transform the text into float values
            onsets = []
            for i in text:
                onsets.append(float(i.replace('\n', '')) * sr)

            # create the target vector with the resulting probabilities.
            targets = torch.zeros_like(waveform)
            for onset in onsets:
                current = torch.arange(0, waveform.shape[1])
                current = 1 / (self.sigma * np.sqrt(2 * np.pi)) * \
                    np.exp(-0.5 * ((current - onset) / self.sigma)**2)
                targets = torch.maximum(targets, current)

            # normalize the targets
            if onsets:
                targets = targets / torch.max(targets)

            waveforms.append((waveform, sr, targets, path))
        r = torch.randn((1, 2**15))
        r = r * 0.98 / torch.max(r)
        waveforms.append((r, sr, torch.zeros_like(r), 'none'))

        if len(waveforms) == 0:
            raise AttributeError('Data-path seems to be empty')

        self.waveforms = waveforms

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform, sr, targets, path = self.waveforms[idx]

        if np.random.uniform() > 0.75:
            noise = torch.randn_like(waveform) / np.random.uniform(8, 100)
            waveform += noise

        if np.random.uniform() > 0.75:
            threshold = np.random.uniform(0.4, 1)
            ratio = np.random.uniform()
            waveform[waveform > threshold] = threshold + \
                (waveform[waveform > threshold] - threshold) * ratio
            waveform = waveform * 0.98 / torch.max(waveform)

        if np.random.uniform() > 0.75:
            gain = np.random.normal(1, np.sqrt(0.5))
            waveform *= gain
            waveform = torch.clip(waveform, max=1)

        return waveform.to(self.device), targets.to(self.device), path
