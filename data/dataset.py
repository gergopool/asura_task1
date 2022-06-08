import torch
from torch import nn
import pandas as pd
import os
from PIL import Image

CLASSES = ['car', 'motorcycle', 'bus', 'train', 'truck']


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, root_folder: str, csv: str, trans: nn.Module):
        super().__init__()
        self.df = pd.read_csv(os.path.join(root_folder, csv))
        self.df.img = self.df.apply(lambda x: os.path.join(root_folder, x.y, x.img), axis=1)
        self.trans = trans

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.img).convert('RGB')
        img = self.trans(img)
        y = torch.tensor(CLASSES.index(row.y), dtype=torch.long)
        return img, y