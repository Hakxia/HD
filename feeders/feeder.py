from torch.utils.data import Dataset
import numpy as np


class FeatureFeeder(Dataset):
    def __init__(self, path, split='train'):
        if split == 'train':
            x = np.load(path + '/train.npy')
            y = np.load(path + '/train_label.npy')
        elif split == 'val':
            x = np.load(path + '/ztest.npy')
            y = np.load(path + '/z_label.npy')
        else:
            raise ValueError(f"Unsupported split: {split}")

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim == 2:
            if x.shape[1] != 256:
                raise ValueError(f"Expected feature shape [N, 256], got {x.shape}.")
            x = x[:, None, :]
        elif x.ndim == 3:
            if x.shape[1:] != (1, 256):
                raise ValueError(f"Expected feature shape [N, 1, 256], got {x.shape}.")
        else:
            raise ValueError(f"Expected feature array with 2 or 3 dimensions, got {x.ndim}.")

        if len(x) != len(y):
            raise ValueError(f"Feature count {len(x)} does not match label count {len(y)}.")

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], int(self.y[index])

    @property
    def features(self) -> np.ndarray:
        return self.x
