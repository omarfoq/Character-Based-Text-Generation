import torch
from torch.utils.data import Dataset


class CharacheterDataset(Dataset):
    def __init__(self, text_as_int, seq_length):
        self.seq_length = seq_length
        self.text_as_int = text_as_int

    def __len__(self):
        return len(self.text_as_int) // (self.seq_length + 1)

    def __getitem__(self, idx):
        input_ = torch.tensor(self.text_as_int[idx: idx + self.seq_length])
        target = torch.tensor(self.text_as_int[idx + 1: idx + self.seq_length + 1])

        return input_, target