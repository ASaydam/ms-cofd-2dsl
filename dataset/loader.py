"""
PyTorch Dataset of the files written by dataset/build_dataset.py:

    from dataset.loader import MSCOFD

    data = MSCOFD('<dataset folder>', split='train', levels=['base_lu', '16x'], channels=['u_x', 'u_y', 'p'],
                  normalise=True, parameters=['blockage_ratio', 'rotation'])
    sample = data[0]   # {'base_lu': (3, 360, 120), '16x': (3, 1440, 480), 'parameters': (2,), 'case_name', 'index'}
    loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True, num_workers=4)

Row i of every level is the same case, so the levels of a sample are read by index. The HDF5 files are opened in
the process that reads them (each DataLoader worker opens its own handles). Without PyTorch the samples are numpy
arrays and the class is a plain sequence.

    split        a split name or a list of names (train, val, test, ood_blockage, ood_rotation, ood_position)
    levels       runs: 0.25x, base_fo, base_lu, 4x, 16x
    channels     names or indices of channels (dataset_info.json), all six by default
    normalise    (x - mean) / std per channel, with the train-split statistics stored in each file
    parameters   metadata.parquet columns returned as one float32 vector (needs pyarrow)
"""

import json
import os
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # numpy samples without PyTorch
    torch = None
    Dataset = object


class MSCOFD(Dataset):

    def __init__(self, root, split='train', levels=('base_lu', '16x'), channels=None, normalise=False, parameters=()):
        import h5py

        self.root = Path(root)
        self.info = json.loads((self.root / 'dataset_info.json').read_text())
        self.splits = [split] if isinstance(split, str) else list(split)
        self.levels = list(levels)
        for name in self.splits:
            if name not in self.info['splits']:
                raise ValueError(f"split {name} not in {list(self.info['splits'])}")
        for level in self.levels:
            if level not in self.info['runs']:
                raise ValueError(f"level {level} not in {list(self.info['runs'])}")
        names = self.info['channels']
        self.channels = [names.index(c) if isinstance(c, str) else int(c) for c in (channels or names)]

        # (split, row) of every sample, and the case names in that order (the same for every level)
        self.rows, self.case_names = [], []
        for name in self.splits:
            with h5py.File(self.path(self.levels[0], name), 'r') as h5:
                cases = [c.decode() if isinstance(c, bytes) else c for c in h5['case_name'][:]]
            self.rows += [(name, i) for i in range(len(cases))]
            self.case_names += cases

        self.normalise = normalise
        self.stats = {}
        for level in self.levels:
            with h5py.File(self.path(level, self.splits[0]), 'r') as h5:
                mean, std = np.asarray(h5.attrs['mean']), np.asarray(h5.attrs['std'])
            std = np.where(std > 0, std, 1.0)
            self.stats[level] = (mean[self.channels, None, None].astype(np.float32),
                                 std[self.channels, None, None].astype(np.float32))

        self.parameters = list(parameters)
        self.parameter_values = None
        if self.parameters:
            import pyarrow.parquet as pq

            table = pq.read_table(self.root / 'metadata.parquet', columns=['case_name'] + self.parameters).to_pydict()
            of_case = {case: [table[p][i] for p in self.parameters] for i, case in enumerate(table['case_name'])}
            self.parameter_values = np.array([of_case[case] for case in self.case_names], dtype=np.float32)

        self.files = {}
        self.pid = None

    def path(self, level, split):
        return self.root / level / f"{split}.h5"

    def file(self, level, split):
        """the open file of a level and split, opened again in a new process (h5py handles do not survive a fork)"""
        import h5py

        if self.pid != os.getpid():
            self.files, self.pid = {}, os.getpid()
        if (level, split) not in self.files:
            self.files[(level, split)] = h5py.File(self.path(level, split), 'r')
        return self.files[(level, split)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        split, row = self.rows[index]
        sample = {'case_name': self.case_names[index], 'index': index}
        for level in self.levels:
            fields = self.file(level, split)['fields'][row][self.channels]
            if self.normalise:
                mean, std = self.stats[level]
                fields = (fields - mean) / std
            sample[level] = torch.from_numpy(fields) if torch else fields
        if self.parameter_values is not None:
            values = self.parameter_values[index]
            sample['parameters'] = torch.from_numpy(values) if torch else values
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state['files'], state['pid'] = {}, None   # DataLoader workers open their own files
        return state
