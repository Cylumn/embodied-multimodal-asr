import os
from ast import literal_eval
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AVDataset(Dataset):
    """
    An audio-visual data class for spoken instructions
    from the preprocessed ALFRED dataset.
    """
    def __init__(self,
                 dir_data: str,
                 id_noise: str,
                 pad_token: int,
                 max_target_length: int,
                 load_noise: bool = False):
        self.id_noise = id_noise
        self.pad_token = pad_token
        self.max_target_length = max_target_length
        self.load_noise = load_noise

        self.dir_audio = os.path.join(dir_data, 'audio', id_noise)
        self.target = pd.read_csv(os.path.join(dir_data, 'target.csv'))
        self.vision = pd.read_csv(os.path.join(dir_data, f'clip.csv'))
        if self.load_noise:
            self.noise = pd.read_csv(os.path.join(dir_data, 'noise.csv'))
            self.noise = self.noise.loc[self.noise['id_noise'] == id_noise, 'idxs'].apply(literal_eval).values

        self.indices = self.target[['id_assignment', 'idx_instruction']]
        self.indices = self.indices.astype({"idx_instruction": int}).astype({"idx_instruction": str})
        self.vision = self.vision.iloc[:, 2:].values
        self.target_str = self.target['transcript_str'].values
        self.target = self.target['transcript_tokens'].apply(literal_eval).values

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.indices)

    def __getitem__(self, 
                    idx: int) -> Tuple[Tuple, Tuple]:
        """
        Returns the audio embedding, image embedding, 
        the target token sequence, and the target string.

        If load_noise = True, also return the indices where words were masked.
        """
        f = '_'.join(self.indices.iloc[idx]) + '.pt'
        audio = torch.load(os.path.join(self.dir_audio, f), map_location='cpu')
        vision = torch.tensor(self.vision[idx], dtype=torch.float)
        target = torch.tensor(
            np.pad(
                self.target[idx],
                (0, max(0, self.max_target_length - len(self.target[idx]))),
                mode='constant',
                constant_values=self.pad_token
            )
        )
        target_str = self.target_str[idx]

        if self.load_noise:
            if "clean" in self.id_noise:
                noise = torch.tensor([], dtype=torch.long)
            else:
                noise = torch.tensor(self.noise[idx], dtype=torch.long)

            return (audio, vision), (target, target_str, noise)
        else:
            return (audio, vision), (target, target_str)
