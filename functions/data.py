import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



# === Class: Inversion Dataset
class InversionDataset(Dataset):
    def __init__(self, data_dir):
        # Read dataset
        with h5py.File(data_dir, 'r') as f:
            self.target = f['target'][:]
            self.obs = f['obs'][:]
            self.obs_label = f['obs_label'][:]

        # Dim transformation
        self.target = self.target[:, np.newaxis, :, :]
        # Tensor conversion
        self.target = self._to_tensor(self.target)
        self.obs = self._to_tensor(self.obs)
        self.obs_label = self._to_tensor(self.obs_label)

    def _to_tensor(self, arr):
        if isinstance(arr, np.ndarray):
            arr = torch.tensor(arr, dtype=torch.float32)
        return arr

    def get_size_params(self):
        return {
            "field_channels": self.target.shape[1],
            "field_size": self.target.shape[2],
            "obs_vector_len": self.obs.shape[2],
            "obs_element_num": self.obs.shape[1],
            "obs_label_size": self.obs_label.shape[2],
            "obs_label_channels": self.obs_label.shape[1],
            "data_length": self.target.shape[0],
        }

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):

        # return data
        return {
            "target": self.target[idx],
            "obs": self.obs[idx],
            "obs_label": self.obs_label[idx],
        }


# === Func: Dataloader
def dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4,):
    dataset = InversionDataset(data_dir)
    model_params = dataset.get_size_params()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, model_params