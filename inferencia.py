"""Command-line inference script for the ticket models: CRNN (OCR) or U-Net (text segmentation)."""

import argparse
import glob
import importlib
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
import torchvision.transforms.functional as F

from models.crnn.crnn import CRNN
from models.crnn.preprocess import PreprocessTransform
from models.crnn.train import greedy_decode

# El directorio "u-net" no es un identificador válido, así que se importa por nombre
unet_module = importlib.import_module('models.u-net.unet')
unet_preprocess = importlib.import_module('models.u-net.preprocess')

RAW_CHARS = " #$%&'()*-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZlr"
BLANK_TOKEN = "<BLANK>"
VOCAB = [BLANK_TOKEN] + sorted(set(RAW_CHARS))
IDX2CHAR = {i: char for i, char in enumerate(VOCAB)}
BLANK_IDX = 0

UNET_CLASSES = ["fondo", "relleno", "borde"]
# Fondo transparente, relleno verde y borde rojo para superponer la máscara a la imagen
MASK_CMAP = ListedColormap([(0, 0, 0, 0), (0, 0.8, 0, 0.5), (1, 0, 0, 0.7)])


def show_results(image_paths, predictions):
    """Display each image next to its predicted text in a grid.

    Args:
        image_paths: Paths to the source images, in prediction order.
        predictions: Predicted text for each image.
    """
    num_imgs = len(image_paths)

    if num_imgs == 1:
        img = Image.open(image_paths[0])
        plt.figure(figsize=(6, 3))
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicción: {predictions[0]}", fontsize=12, fontweight='bold', color='blue')
        plt.axis('off')
    else:
        cols = min(4, num_imgs)
        rows = math.ceil(num_imgs / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2 * rows))
        axes = axes.flatten() if num_imgs > 1 else [axes]

        for i in range(num_imgs):
            img = Image.open(image_paths[i])
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Pred: {predictions[i]}", fontsize=10, color='blue')
            axes[i].axis('off')

        for j in range(num_imgs, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def show_masks(image_paths, masks):
    """Display each image next to the same image with its predicted mask overlaid.

    Args:
        image_paths: Paths to the source images, in prediction order.
        masks: Predicted class mask (H, W) for each image.
    """
    num_imgs = len(image_paths)
    fig, axes = plt.subplots(num_imgs, 2, figsize=(10, 6 * num_imgs), squeeze=False)

    for i, (path, mask) in enumerate(zip(image_paths, masks)):
        img = Image.open(path).convert('L')
        axes[i][0].imshow(img, cmap='gray')
        axes[i][0].set_title(os.path.basename(path), fontsize=10)
        axes[i][1].imshow(img, cmap='gray')
        axes[i][1].imshow(mask, cmap=MASK_CMAP, vmin=0, vmax=len(UNET_CLASSES) - 1, interpolation='nearest')
        axes[i][1].set_title("Máscara (verde: relleno, rojo: borde)", fontsize=10, color='blue')
        axes[i][0].axis('off')
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()


def predict(model, image_paths, device):
    """Run the CRNN model on a batch of images and decode the predictions.

    Args:
        model: A loaded CRNN model.
        image_paths: Paths to the images to predict.
        device: Torch device to run inference on.

    Returns:
        A list of decoded text predictions, in the same order as `image_paths`.
    """
    transform = PreprocessTransform(alto_fijo=32, ancho_maximo=256)
    tensors = [transform(Image.open(path)) for path in image_paths]
    batch_tensor = torch.stack(tensors, dim=0).to(device)

    model.eval()
    with torch.no_grad():
        log_probs = model(batch_tensor)
        predictions = greedy_decode(log_probs, IDX2CHAR, blank_idx=BLANK_IDX)

    return predictions


def predict_masks(model, image_paths, device):
    """Run the U-Net model on each image and return its predicted class mask.

    Images have different sizes, so they are processed one at a time with the same
    preprocessing used for validation, and the padding is cropped from the output.

    Args:
        model: A loaded U-Net model.
        image_paths: Paths to the images to segment.
        device: Torch device to run inference on.

    Returns:
        A list of (H, W) numpy arrays with values 0 (fondo), 1 (relleno) and 2 (borde).
    """
    transform = v2.Compose([
        unet_preprocess.pad_to_multiple,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(unet_preprocess.MEAN, unet_preprocess.STD),
    ])

    model.eval()
    masks = []
    with torch.no_grad():
        for path in image_paths:
            img = tv_tensors.Image(F.pil_to_tensor(Image.open(path).convert('L')))
            h, w = img.shape[-2:]
            x = transform(img).unsqueeze(0).to(device)
            mask = torch.argmax(model(x), dim=1)[0, :h, :w]
            masks.append(mask.cpu().numpy())

    return masks


def resolve_image_paths(inputs):
    """Expand a list of paths and glob patterns into a flat list of image files.

    Args:
        inputs: Raw strings from the command line, each either a direct
            path or a glob pattern (e.g. "data/*.png").

    Returns:
        A list of existing image paths.
    """
    image_paths = []
    for item in inputs:
        matched = glob.glob(item)
        if matched:
            image_paths.extend(matched)
        elif os.path.exists(item):
            image_paths.append(item)

    return image_paths


def run_crnn(args, image_paths, device):
    model = CRNN(in_channels=1, output_classes=len(VOCAB), width=256)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    predictions = predict(model, image_paths, device)

    print("\n--- Resultados ---")
    for img_path, pred in zip(image_paths, predictions):
        print(f"[{os.path.basename(img_path)}] -> {pred}")

    show_results(image_paths, predictions)


def run_unet(args, image_paths, device):
    model = unet_module.UNet(1, args.channels, len(UNET_CLASSES))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    masks = predict_masks(model, image_paths, device)

    print("\n--- Resultados ---")
    for img_path, mask in zip(image_paths, masks):
        shares = ", ".join(f"{name}: {np.mean(mask == c):.1%}" for c, name in enumerate(UNET_CLASSES))
        print(f"[{os.path.basename(img_path)}] -> {shares}")

    show_masks(image_paths, masks)


RUNNERS = {
    'crnn': run_crnn,
    'unet': run_unet,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia con uno de los modelos del proyecto: CRNN (reconocimiento de texto) o U-Net (segmentación de texto en tickets).")
    subparsers = parser.add_subparsers(dest='model', required=True, help='Modelo a utilizar')

    crnn = subparsers.add_parser('crnn', help='Reconoce el texto de recortes de tickets con el CRNN.')
    crnn.add_argument('--weights', type=str, default='best_crnn_model.pth', help='Ruta al archivo .pth con los pesos del modelo (default: best_crnn_model.pth)')

    unet = subparsers.add_parser('unet', help='Segmenta el texto de tickets completos con la U-Net.')
    unet.add_argument('--weights', type=str, default='best_unet_model.pth', help='Ruta al archivo .pth con los pesos del modelo (default: best_unet_model.pth)')
    unet.add_argument('--channels', type=int, default=32, help='Canales de la primera capa de la U-Net; debe coincidir con el entrenamiento (default: 32)')

    for sub in (crnn, unet):
        sub.add_argument('--input', nargs='+', required=True, help="Ruta a una imagen, varias imágenes, o un patrón glob (ej. 'data/*.png')")

    return parser.parse_args()


def main():
    args = parse_args()

    image_paths = resolve_image_paths(args.input)
    if not image_paths:
        print("Error: No se encontraron imágenes válidas con la entrada proporcionada.")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    RUNNERS[args.model](args, image_paths, device)


if __name__ == "__main__":
    main()
