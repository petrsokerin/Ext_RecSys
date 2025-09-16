import torch
from torch.utils.data import Dataset

class TrainSASRecDataset(Dataset):
    def __init__(self, sequences, times, max_len=50):
        self.sequences = sequences
        self.times = times
        self.max_len = max_len


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        time = self.times[idx]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
            time = time[-self.max_len:]
        else:
            seq = [0] * (self.max_len - len(seq)) + seq
            time = [0] * (self.max_len - len(time)) + time
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        input_times = torch.tensor(time[:-1], dtype=torch.long)
        target = torch.tensor(seq[1:], dtype=torch.long)
        return (input_seq, input_times), target


class ValSASRecDataset(Dataset):
    def __init__(self, sequences, times, max_len=50):
        self.sequences = sequences
        self.times = times
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        time = self.times[idx]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
            time = time[-self.max_len:]
        else:
            seq = [0] * (self.max_len - len(seq)) + seq
            time = [0] * (self.max_len - len(time)) + time
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        input_times = torch.tensor(time[:-1], dtype=torch.long)
        target = torch.tensor(seq[-1], dtype=torch.long)

        return (input_seq, input_times), target