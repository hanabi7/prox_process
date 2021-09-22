import torch
from torch.utils.data import Dataset
import os


class ProxDataset(Dataset):
    def __init__(self, save_dir):
        super(ProxDataset, self).__init__()
        self.save_dir = save_dir
