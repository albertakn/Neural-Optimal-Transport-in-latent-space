import torch
import PIL
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FolderDataset(Dataset):
    def __init__(self, data_root, img_size):
        self.data_root = data_root
        self.img_size = img_size
        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(self.data_root)]

    def __len__(self):
        return len(self.image_paths)

    def _rescale(self, x, old_range, new_range, clamp=False):
        old_min, old_max = old_range
        new_min, new_max = new_range
        x -= old_min
        x *= (new_max - new_min) / (old_max - old_min)
        x += new_min
        if clamp:
            x = x.clamp(new_min, new_max)
        return x

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = image.convert("RGB") if not image.mode == "RGB" else image

        img = torch.tensor(np.array(image.resize((self.img_size, self.img_size))), dtype=torch.float32)
        img = self._rescale(img, (0, 255), (-1, 1))
        img = img.permute(2, 0, 1)
        noise = torch.randn(4, 64, 64)
        return img, noise
