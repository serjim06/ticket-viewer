import json
import os
from datetime import datetime


def default_results_dir(model_name):
    """
    Directorio por defecto donde se guardan los resultados de un modelo: ./results/<model_name>.
    """
    return os.path.join("results", model_name)


def save_results(model_name, config, history, results_dir=None, best_metric="val_loss"):
    """
    Guarda la configuración y el historial de un entrenamiento en un JSON con marca de tiempo.
    La mejor época es la que minimiza best_metric (debe coincidir con el criterio del checkpoint).
    Devuelve la ruta del fichero generado.
    """
    results_dir = results_dir or default_results_dir(model_name)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(results_dir, f"{model_name}_{timestamp}.json")

    best = min(history, key=lambda h: h[best_metric]) if history else None

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "timestamp": timestamp,
            "config": config,
            "best_metric": best_metric,
            "best_epoch": best,
            "history": history,
        }, f, indent=2, ensure_ascii=False)

    return path
