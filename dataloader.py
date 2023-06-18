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
                onsets.append(float(i.replace('\n', '')) * sr)

            # create the target vector with the resulting probabilities.
            targets = torch.zeros_like(waveform)
            k = config.data_targetSD
            for onset in onsets:
                onset = int(onset)
                value = torch.arange(2*k + 1, dtype=float) - k
                value = 1 - (torch.abs(value) / k)**2
                smaller = max(0, onset - k)
                bigger = min(targets.shape[1], onset + k + 1)
                r = bigger - smaller
                value = value[k - (onset - smaller): k + bigger - onset]
                value = torch.max(value, targets[:, smaller: bigger])
                targets[:, smaller: bigger] = value

            waveforms.append((waveform, sr, targets))

        # r = torch.randn_like(waveform)
        # r = r * 0.98 / torch.max(r)
        # waveforms.append((r, sr, torch.zeros_like(r)))

        if len(waveforms) == 0:
            raise AttributeError('Data-path seems to be empty')

        self.waveforms = waveforms
        self.device = device
        self.length = config.data_length
        self.sigma = config.data_targetSD

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform, sr, targets = self.waveforms[idx]

        # find a random index and start from this point
        index = np.random.randint(low=0, high=waveform.shape[1])

        # create a waveform of specified length, with indexing and zero padding
        waveform = torch.nn.functional.pad(
            waveform[:, index: index + self.length],
            (0, max(0, self.length + index - waveform.shape[1]))
        )
        targets = torch.nn.functional.pad(
            targets[:, index: index + self.length],
            (0, max(0, self.length + index - targets.shape[1]))
        )

        if np.random.uniform() > 0.75:
            noise = torch.randn_like(waveform) / np.random.uniform(8, 100)
            waveform += noise

        # if np.random.uniform() > 0.75:
        #     threshold = np.random.uniform(0.4, 1)
        #     ratio = np.random.uniform()
        #     waveform[waveform > threshold] = threshold + \
        #         (waveform[waveform > threshold] - threshold) * ratio
        #     waveform = waveform * 0.98 / torch.max(waveform)

        # if np.random.uniform() > 0.75:
        #     gain = np.random.normal(1, np.sqrt(0.5))
        #     waveform *= gain
        #     waveform = torch.clip(waveform, max=1)

        return waveform.to(self.device), targets.to(self.device)

    def getFull(self, idx):

        waveform, sr, targets = self.waveforms[idx]
        zeros = torch.zeros((1, self.length // 2))
        zeros2 = torch.zeros((1, self.length))
        waveform = torch.cat((zeros, waveform, zeros2), dim=1)
        targets = torch.cat((zeros, targets, zeros2), dim=1)

        waveform = waveform.to(self.device)
        targets = targets.to(self.device)
        waveform_list = []
        target_list = []
        for i in range(0, waveform.shape[1] - self.length,  self.length // 2):
            waveform_list.append(waveform[:, i:i+self.length])
            target_list.append(targets[:, i:i+self.length])

        return waveform_list, target_list


class OnsetFull(OnsetDataset):
    def __getitem__(self, idx):
        waveform, sr, onsets = self.waveforms[idx]
