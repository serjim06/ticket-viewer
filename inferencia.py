"""Command-line inference script for the CRNN ticket OCR model."""

import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from models.model import CRNN
from models.preprocess import PreprocessTransform
from models.train import greedy_decode

RAW_CHARS = " #$%&'()*-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZlr"
BLANK_TOKEN = "<BLANK>"
VOCAB = [BLANK_TOKEN] + sorted(set(RAW_CHARS))
IDX2CHAR = {i: char for i, char in enumerate(VOCAB)}
BLANK_IDX = 0


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


def main():
    parser = argparse.ArgumentParser(description="Script de Inferencia para CRNN con CTC Loss")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Ruta a una imagen, varias imágenes, o un patrón glob (ej. 'data/*.png')"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best_crnn_model.pth",
        help="Ruta al archivo .pth con los pesos del modelo"
    )

    args = parser.parse_args()

    image_paths = resolve_image_paths(args.input)
    if not image_paths:
        print("Error: No se encontraron imágenes válidas con la entrada proporcionada.")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CRNN(in_channels=1, output_classes=len(VOCAB), width=256)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    predictions = predict(model, image_paths, device)

    print("\n--- Resultados ---")
    for img_path, pred in zip(image_paths, predictions):
        print(f"[{os.path.basename(img_path)}] -> {pred}")

    show_results(image_paths, predictions)


if __name__ == "__main__":
    main()
