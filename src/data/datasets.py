import torch
from torch.utils.data import Dataset

class TrainSASRecDataset(Dataset):
    def __init__(self, sequences, times, max_len=50):
        self.sequences = sequences
        self.times = times
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)
    
    def pad_sequence(self, seq, time):
        padding_mask = torch.ones(self.max_len, dtype=torch.bool)
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
            time = time[-self.max_len:]
        else:
            n_pads = self.max_len - len(seq)
            seq = [0] * (n_pads) + seq
            time = [0] * (n_pads) + time
            padding_mask[:n_pads] = False

        return seq, time, padding_mask


    def __getitem__(self, idx):
        seq = self.sequences[idx]
        time = self.times[idx]

        seq, time, padding_mask = self.pad_sequence(seq, time)
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        input_times = torch.tensor(time[:-1], dtype=torch.long)
        train_padding_mask = padding_mask[:-1]

        target = torch.tensor(seq[1:], dtype=torch.long)
        label_padding_mask = padding_mask[1:]

        return (input_seq, input_times, train_padding_mask), (target, label_padding_mask)


class ValSASRecDataset(TrainSASRecDataset):

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        time = self.times[idx]

        seq, time, padding_mask = self.pad_sequence(seq, time)
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        input_times = torch.tensor(time[:-1], dtype=torch.long)
        train_padding_mask = padding_mask[:-1]

        target = torch.tensor(seq[-1], dtype=torch.long)
        label_padding_mask = padding_mask[-1]


        return (input_seq, input_times, train_padding_mask), (target, label_padding_mask)