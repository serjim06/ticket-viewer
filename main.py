import argparse
import models
from models.crnn.crnn import CRNN
from models.preprocess import vocab, idx2char
from models.train import train_model
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el modelo CRNN para reconocimiento de texto en recortes de tickets.")

    parser.add_argument('--train-dir', type=str, default='dataset_crops/train', help='Directorio con las imágenes de entrenamiento (default: dataset_crops/train)')
    parser.add_argument('--test-dir', type=str, default='dataset_crops/test', help='Directorio con las imágenes de test (default: dataset_crops/test)')
    parser.add_argument('--train-csv', type=str, default='train.csv', help='CSV con las etiquetas de entrenamiento (default: train.csv)')
    parser.add_argument('--test-csv', type=str, default='test.csv', help='CSV con las etiquetas de test (default: test.csv)')
    parser.add_argument('--split', type=float, default=0.8, help="Train/val split")
    parser.add_argument('--batch-size', type=int, default=64, help='Tamaño de batch (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, help='Número de épocas de entrenamiento (default: 25)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Dispositivo a usar (default: cuda si está disponible, si no cpu)')
    parser.add_argument('--output', type=str, default='best_crnn_model.pth', help='Ruta donde guardar los mejores pesos del modelo (default: best_crnn_model.pth)')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = models.preprocess.get_loaders(train_dir=args.train_dir, test_dir=args.test_dir, 
                                                                         train_labels_path=args.train_csv, test_labels_path = args.test_csv,
                                                                         split=args.split, batch_size=args.batch_size, shuffle=True)
    

    num_classes = len(vocab)

    model = CRNN(1, num_classes, 256)

    print("Preprocesado completado. Iniciando entrenamiento...")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        idx2char=idx2char,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
