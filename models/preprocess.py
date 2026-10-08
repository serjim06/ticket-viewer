import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torchvision.transforms.functional as F
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

raw_chars = " #$%&'()*-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZlr"
BLANK_TOKEN = "<BLANK>"
vocab = [BLANK_TOKEN] + sorted(list(set(raw_chars)))
char2idx = {char: i for i, char in enumerate(vocab)}
idx2char = {i: char for i, char in enumerate(vocab)}

BLANK_IDX = 0


class PreprocessTransform:
    def __init__(self, alto_fijo=32, ancho_maximo=128):
        self.alto_fijo = alto_fijo
        self.ancho_maximo = ancho_maximo
        self.media = [0.91]
        self.std = [0.15]

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert('L')
        
        w, h = img.size
        factor_escala = self.alto_fijo / h
        nuevo_ancho = int(w * factor_escala)
        
        nuevo_ancho = min(nuevo_ancho, self.ancho_maximo)
        img = F.resize(img, (self.alto_fijo, nuevo_ancho))
        
        if nuevo_ancho < self.ancho_maximo:
            pad_derecha = self.ancho_maximo - nuevo_ancho
            img = F.pad(img, (0, 0, pad_derecha, 0), fill=255)
            
        tensor_img = F.to_tensor(img)
        tensor_img = F.normalize(tensor_img, mean=self.media, std=self.std)
        
        return tensor_img
    
class ImageDataset(Dataset):
    def __init__(self, dataset_dir, labels_csv_path, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        self.df = pd.read_csv(labels_csv_path)
        
        self.df = self.df.dropna(subset=['label', 'image_path'])
        self.df['label'] = self.df['label'].astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        ruta_img = os.path.join(row['image_path'])
        label_text = row['label']
        
        imagen = Image.open(ruta_img)
        
        if self.transform:
            imagen = self.transform(imagen)
            
        label_encoded = torch.tensor(
            [char2idx[c] for c in label_text if c in char2idx], 
            dtype=torch.long
        )
        
        return imagen, label_encoded, label_text
    
def ctc_collate_fn(batch):
    """
    Empaqueta un batch de muestras con secuencias de distinta longitud.
    """
    images, labels, labels_text = zip(*batch)
    
    images = torch.stack(images, dim=0)
    
    target_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    
    targets = torch.nn.utils.rnn.pad_sequence(
        labels, 
        batch_first=True, 
        padding_value=BLANK_IDX
    )
    
    return images, targets, target_lengths, labels_text    

    
def get_loader(dataset_dir, labels_csv_path, batch_size=64, shuffle=True):
    transformer = PreprocessTransform(alto_fijo=32, ancho_maximo=256)
    
    dataset = ImageDataset(
        dataset_dir=dataset_dir, 
        labels_csv_path=labels_csv_path, 
        transform=transformer
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=ctc_collate_fn
    )
    
    return loader
    