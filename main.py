import models
from models.model import CRNN
from models.preprocess import vocab, idx2char
from models.train import train_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = models.preprocess.get_loader(dataset_dir='dataset_crops/train', labels_csv_path='train.csv', batch_size=64, shuffle=True)
val_loader = models.preprocess.get_loader(dataset_dir='dataset_crops/test', labels_csv_path='test.csv', batch_size=64, shuffle=False)

num_classes = len(vocab)

model = CRNN(1, num_classes, 256)

print("Preprocesado completado. Iniciando entrenamiento...")

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    idx2char=idx2char,
    epochs=25,
    lr=1e-3,
    device=device
)