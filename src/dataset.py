import random
import PIL.ImageOps
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None, invert=False, showimage=False, testing=False):
        self.image_folder = image_folder
        self.transform = transform
        self.invert = invert
        self.showimage = showimage
        self.testing = testing

    def __getitem__(self, index):
        sample_1 = random.choice(self.image_folder.imgs)
        same_class = random.randint(0, 1)
        # Image.open(sample_1)
        if same_class:
            while True:
                # keep looping till the same class image is found
                sample_2 = random.choice(self.image_folder.imgs)
                if sample_1[1] == sample_2[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found
                sample_2 = random.choice(self.image_folder.imgs)
                if sample_1[1] != sample_2[1]:
                    break
        counter44 = 1
        img_1 = Image.open(sample_1[0]).convert("L")
        img_2 = Image.open(sample_2[0]).convert("L")
        if self.showimage:
            counter44 = counter44 + 1
        if self.invert:
            img_1 = PIL.ImageOps.invert(img_1)
            img_2 = PIL.ImageOps.invert(img_2)
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        if self.testing:
            return img_1, img_2, torch.from_numpy(
                np.array([int(sample_1[1] != sample_2[1])], dtype=np.float32)), torch.from_numpy(
                np.array([sample_1[1]], dtype=np.float32)), torch.from_numpy(np.array([sample_2[1]], dtype=np.float32))
        else:
            return img_1, img_2, torch.from_numpy(np.array([int(sample_1[1] != sample_2[1])], dtype=np.float32))

    def __len__(self):
        return len(self.image_folder.imgs)
