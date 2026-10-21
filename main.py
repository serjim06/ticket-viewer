import argparse
import importlib
import os
from models.results import save_results, default_results_dir
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena uno de los modelos del proyecto: CRNN (reconocimiento de texto) o U-Net (segmentación de texto en tickets).")
    subparsers = parser.add_subparsers(dest='model', required=True, help='Modelo a entrenar')

    crnn = subparsers.add_parser('crnn', help='Entrena el CRNN para reconocimiento de texto en recortes de tickets.')
    crnn.add_argument('--train-dir', type=str, default='datasets/dataset_crops/train', help='Directorio con las imágenes de entrenamiento (default: datasets/dataset_crops/train)')
    crnn.add_argument('--test-dir', type=str, default='datasets/dataset_crops/test', help='Directorio con las imágenes de test (default: datasets/dataset_crops/test)')
    crnn.add_argument('--train-csv', type=str, default='datasets/dataset_crops/train.csv', help='CSV con las etiquetas de entrenamiento (default: datasets/dataset_crops/train.csv)')
    crnn.add_argument('--test-csv', type=str, default='datasets/dataset_crops/test.csv', help='CSV con las etiquetas de test (default: datasets/dataset_crops/test.csv)')
    crnn.add_argument('--split', type=float, default=0.8, help="Train/val split")
    crnn.add_argument('--batch-size', type=int, default=64, help='Tamaño de batch (default: 64)')
    crnn.add_argument('--epochs', type=int, default=25, help='Número de épocas de entrenamiento (default: 25)')
    crnn.add_argument('--patience', type=int, default=5, help='Épocas sin mejorar val_cer antes de parar el entrenamiento; 0 lo desactiva (default: 5)')
    crnn.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    crnn.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Dispositivo a usar (default: cuda si está disponible, si no cpu)')
    crnn.add_argument('--init-weights', type=str, default=None, help='Pesos (*.pth) con los que inicializar el modelo antes de entrenar (default: inicialización aleatoria)')
    crnn.add_argument('--output', type=str, default='best_crnn_model.pth', help='Ruta donde guardar los mejores pesos del modelo (default: best_crnn_model.pth)')
    crnn.add_argument('--results-dir', type=str, default=default_results_dir('crnn'), help='Directorio donde guardar los resultados del entrenamiento (default: results/crnn)')

    unet = subparsers.add_parser('unet', help='Entrena la U-Net para segmentación de texto en tickets completos.')
    unet.add_argument('--train-dir', type=str, default='datasets/u-net/train', help='Directorio con las imágenes de entrenamiento (default: datasets/u-net/train)')
    unet.add_argument('--train-mask-dir', type=str, default='datasets/u-net/train_masks', help='Directorio con las máscaras de entrenamiento (default: datasets/u-net/train_masks)')
    unet.add_argument('--test-dir', type=str, default='datasets/u-net/test', help='Directorio con las imágenes de test (default: datasets/u-net/test)')
    unet.add_argument('--test-mask-dir', type=str, default='datasets/u-net/test_masks', help='Directorio con las máscaras de test; si no existe se omite la evaluación en test (default: datasets/u-net/test_masks)')
    unet.add_argument('--split', type=float, default=0.8, help="Train/val split")
    unet.add_argument('--batch-size', type=int, default=8, help='Tamaño de batch (default: 8)')
    unet.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento (default: 100)')
    unet.add_argument('--patience', type=int, default=10, help='Épocas sin mejorar val_dice antes de parar el entrenamiento; 0 lo desactiva (default: 10)')
    unet.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    unet.add_argument('--channels', type=int, default=32, help='Canales de la primera capa de la U-Net (default: 32)')
    unet.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Dispositivo a usar (default: cuda si está disponible, si no cpu)')
    unet.add_argument('--init-weights', type=str, default=None, help='Pesos (*.pth) con los que inicializar el modelo antes de entrenar (default: inicialización aleatoria)')
    unet.add_argument('--output', type=str, default='best_unet_model.pth', help='Ruta donde guardar los mejores pesos del modelo (default: best_unet_model.pth)')
    unet.add_argument('--results-dir', type=str, default=default_results_dir('unet'), help='Directorio donde guardar los resultados del entrenamiento (default: results/unet)')

    return parser.parse_args()


def load_init_weights(model, args, device):
    """
    Carga en el modelo los pesos de --init-weights, si se han indicado, para continuar entrenando desde ellos.
    """
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location=device, weights_only=True))
        print(f"Modelo inicializado con los pesos de {args.init_weights}")
        if os.path.abspath(args.init_weights) == os.path.abspath(args.output):
            # El mejor valor de la métrica empieza de cero, así que la primera época siempre guarda checkpoint
            print(f"Aviso: --init-weights coincide con --output; {args.output} se sobrescribirá al terminar la primera época.")


def train_crnn(args, device):
    from models.crnn.crnn import CRNN
    from models.crnn.preprocess import vocab, idx2char, get_loaders_crnn
    from models.crnn.train import train_model, evaluate_model

    train_loader, val_loader, test_loader = get_loaders_crnn(train_dir=args.train_dir, test_dir=args.test_dir,
                                                             train_labels_path=args.train_csv, test_labels_path=args.test_csv,
                                                             split=args.split, batch_size=args.batch_size, shuffle=True)

    num_classes = len(vocab)

    model = CRNN(1, num_classes, 256)
    load_init_weights(model, args, device)

    print(f"Preprocesado completado. Iniciando entrenamiento en {device}...")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        idx2char=idx2char,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_path=args.output,
        patience=args.patience
    )

    if not history:
        # Terminado desde la pausa antes de completar una época: no hay checkpoint de esta ejecución
        print("Entrenamiento terminado sin ninguna época completa: se omite la evaluación en test.")
        return history, None, 'val_cer', 'min'

    print("Entrenamiento completado. Evaluando el mejor checkpoint en test...")

    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        idx2char=idx2char,
        weights_path=args.output,
        device=device
    )

    return history, test_metrics, 'val_cer', 'min'


def train_unet(args, device):
    # El directorio "u-net" no es un identificador válido, así que se importa por nombre
    unet_module = importlib.import_module('models.u-net.unet')
    unet_preprocess = importlib.import_module('models.u-net.preprocess')
    unet_train = importlib.import_module('models.u-net.train')

    has_test_masks = os.path.isdir(args.test_mask_dir)

    train_loader, val_loader, test_loader = unet_preprocess.get_unet_loaders(
        train_path=args.train_dir, train_mask_path=args.train_mask_dir,
        test_path=args.test_dir, test_mask_path=args.test_mask_dir if has_test_masks else None,
        split=args.split, batch_size=args.batch_size, shuffle=True)

    model = unet_module.UNet(1, args.channels, 3)
    load_init_weights(model, args, device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)

    print(f"Preprocesado completado. Iniciando entrenamiento en {device}...")

    history = unet_train.train(
        model=model,
        optimiser=optimiser,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scheduler=scheduler,
        epochs=args.epochs,
        output_path=args.output,
        patience=args.patience
    )

    test_metrics = None
    if not history:
        # Terminado desde la pausa antes de completar una época: no hay checkpoint de esta ejecución
        print("Entrenamiento terminado sin ninguna época completa: se omite la evaluación en test.")
    elif has_test_masks:
        print("Entrenamiento completado. Evaluando el mejor checkpoint en test...")
        model.load_state_dict(torch.load(args.output, map_location=device))
        test_loss, test_acc, test_dice, test_iou, test_dice_c = unet_train.accuracy(model, test_loader, device=device)
        test_metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_dice": test_dice,
            "test_dice_fill": test_dice_c[1],
            "test_dice_border": test_dice_c[2],
            "test_iou": test_iou,
        }
        print(f"Test | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Dice: {test_dice:.4f} | IoU: {test_iou:.4f}")
    else:
        print(f"Entrenamiento completado. No existe {args.test_mask_dir}: se omite la evaluación en test.")

    return history, test_metrics, 'val_dice', 'max'


TRAINERS = {
    'crnn': train_crnn,
    'unet': train_unet,
}


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history, test_metrics, best_metric, best_mode = TRAINERS[args.model](args, device)

    config = {k: v for k, v in vars(args).items() if k not in ('results_dir', 'model')}
    config['device'] = str(device)
    results_path = save_results(args.model, config, history, results_dir=args.results_dir, best_metric=best_metric,
                                best_mode=best_mode, test_metrics=test_metrics)
    print(f"Resultados guardados en {results_path}")


if __name__ == '__main__':
    main()
