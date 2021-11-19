from torch.utils import data


class JetDataset(data.Dataset):
    def __init__(self, data: list):
        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
