from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
from models.pause import TrainingPause

IGNORE_INDEX = 255

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy(model, loader, num_classes=3, device=device):
    """
    Calcula pérdida, accuracy, Dice e IoU sobre un loader, ignorando los píxeles de padding.
    Dice e IoU se calculan por clase acumulando sobre todo el loader, y se devuelve la media
    de las clases de texto (1 = relleno, 2 = borde), sin el fondo.
    También devuelve el Dice de cada clase (lista indexada por clase) para seguir el borde por separado.
    """
    model = model.to(device)
    model.eval()

    cost = 0.
    correct = 0
    total = 0
    inter = torch.zeros(num_classes, device=device)
    pred_sum = torch.zeros(num_classes, device=device)
    true_sum = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            cost += F.cross_entropy(scores, y, ignore_index=IGNORE_INDEX).item()

            preds = torch.argmax(scores, dim=1)
            valid = y != IGNORE_INDEX
            p, t = preds[valid], y[valid]

            correct += (p == t).sum().item()
            total += t.numel()

            for c in range(num_classes):
                pc, tc = p == c, t == c
                inter[c] += (pc & tc).sum()
                pred_sum[c] += pc.sum()
                true_sum[c] += tc.sum()

    dice_c = 2 * inter / (pred_sum + true_sum + 1e-8)
    iou_c = inter / (pred_sum + true_sum - inter + 1e-8)

    dice = dice_c[1:].mean().item()
    iou = iou_c[1:].mean().item()

    return cost / len(loader), correct / total, dice, iou, dice_c.tolist()

def train(model, optimiser, train_loader, val_loader, device, scheduler=None, epochs=100,
          output_path="best_unet_model.pth", patience=10):
    """
    Entrena la U-Net guardando el checkpoint con mayor Dice de validación.
    Si el Dice no mejora en `patience` épocas seguidas, se detiene. patience=0 lo desactiva.
    Con Ctrl+C se pausa liberando la GPU; desde la pausa se puede terminar, descartando la época en curso.
    Devuelve el histórico por época.
    """
    model = model.to(device)

    best_dice = -1.0
    epochs_without_improvement = 0
    history = []
    pauser = TrainingPause(model, optimiser, device)
    stop = False

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct_num = 0
        train_total = 0
        train_cost_acum = 0.

        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            cost = F.cross_entropy(input=scores, target=y, ignore_index=IGNORE_INDEX)

            optimiser.zero_grad()
            cost.backward()
            optimiser.step()

            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            valid = y != IGNORE_INDEX
            train_predictions = torch.argmax(scores, dim=1)
            train_correct_num += (train_predictions[valid] == y[valid]).sum().item()
            train_total += valid.sum().item()
            train_cost_acum += cost.item()

            if pauser.requested:
                # Se sueltan los tensores del batch para que la pausa pueda liberar su memoria en la GPU
                del x, y, scores, cost, valid, train_predictions
                stop = pauser.pause()
                if stop:
                    break

        if stop:
            print(f"Entrenamiento terminado manualmente en la época {epoch}, que se descarta. Mejor Dice: {best_dice:.4f}")
            break

        train_cost = train_cost_acum / len(train_loader)
        train_acc = train_correct_num / train_total

        val_cost, val_acc, dice, iou, dice_c = accuracy(model, val_loader, device=device)
        val_cost, val_acc, dice, iou = float(val_cost), float(val_acc), float(dice), float(iou)
        dice_fill, dice_border = dice_c[1], dice_c[2]

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_cost)

        if dice > best_dice:
            best_dice = dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_path)
            saved_str = " -> ¡Modelo Guardado!"
        else:
            epochs_without_improvement += 1
            saved_str = ""

        history.append({
            "epoch": epoch,
            "train_loss": train_cost,
            "train_acc": train_acc,
            "val_loss": val_cost,
            "val_acc": val_acc,
            "val_dice": dice,
            "val_dice_fill": dice_fill,
            "val_dice_border": dice_border,
            "val_iou": iou,
            "lr": optimiser.param_groups[0]['lr'],
        })

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_cost:.4f} | Val Loss: {val_cost:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Dice: {dice:.4f} "
              f"(relleno {dice_fill:.4f}, borde {dice_border:.4f}) | IoU: {iou:.4f}{saved_str}")

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping: Dice sin mejorar en {patience} épocas. Mejor Dice: {best_dice:.4f}")
            break

    pauser.close()
    return history
