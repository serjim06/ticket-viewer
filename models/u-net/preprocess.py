import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms as T
from PIL import ImageFilter
import random


class UNetDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None):
        """
        Args:
            images_dir (str): directorio de imágenes.
            masks_dir (str, optional): directorio de máscaras con el mismo nombre de fichero.
            transform (callable, optional): transform v2 aplicado conjuntamente a imagen y máscara.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(os.listdir(images_dir))

        if masks_dir is not None:
            missing = [n for n in self.images if not os.path.exists(os.path.join(masks_dir, n))]
            assert not missing, f"{len(missing)} imágenes sin máscara, p. ej. {missing[:3]}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img = Image.open(os.path.join(self.images_dir, name)).convert('L')
        img = tv_tensors.Image(F.pil_to_tensor(img))  # uint8 (1, H, W)

        if self.masks_dir is None:
            return self.transform(img) if self.transform else img

        mask = np.array(Image.open(os.path.join(self.masks_dir, name)))
        mask = tv_tensors.Mask(torch.from_numpy(mask))  # uint8 (H, W), valores 0/1/2

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask.long()


def pad_to_multiple(*inputs, multiple=32):
    """Rellena por abajo y por la derecha hasta un múltiplo de `multiple`.
    255 es blanco en la imagen e ignore_index en la máscara."""
    h, w = inputs[0].shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    out = tuple(v2.functional.pad(x, [0, 0, pad_w, pad_h], fill=255) for x in inputs)
    return out if len(out) > 1 else out[0]


def get_unet_loaders(train_path, train_mask_path, test_path, test_mask_path, split=0.8, batch_size=8, shuffle=True):
    FILL = {tv_tensors.Image: 255, tv_tensors.Mask: 255}
    MEAN, STD = [0.9137], [0.1841]  # placeholder: recalcúlalos sobre datasets/u-net/train

    train_tf = v2.Compose([
        v2.RandomAffine(degrees=3, scale=(0.8, 1.2), fill=FILL, interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomCrop(512, pad_if_needed=True, fill=FILL),
        v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.5),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.4)], p=0.5),
        v2.ToDtype(torch.float32, scale=True),   # solo afecta a la imagen
        v2.Normalize(MEAN, STD),                 # solo afecta a la imagen
    ])

    val_tf = v2.Compose([
        pad_to_multiple,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])


    train_dataset = UNetDataset(
            images_dir=train_path, 
            masks_dir=train_mask_path, 
            transform=train_tf
        )
        
    val_dataset = UNetDataset(
            images_dir=train_path, 
            masks_dir=train_mask_path, 
            transform=val_tf
         )
            
    test_dataset = UNetDataset(
            images_dir=test_path,
            masks_dir=test_mask_path,
            transform=val_tf
        )  
    
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(train_dataset), generator=g).tolist()
    
    
    train_size = int(len(idx)*split)
    
    train_dataset = Subset(train_dataset, idx[:train_size])
    val_dataset = Subset(val_dataset, idx[train_size:])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )    
    
    return train_loader, val_loader, test_loader