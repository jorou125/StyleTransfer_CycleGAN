import torch
from PIL import Image
import os
import config
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, direction="AtoB", transforms=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transforms = transforms
        self.direction = direction

        self.images_A = os.listdir(root_A)
        self.images_B = os.listdir(root_B)
