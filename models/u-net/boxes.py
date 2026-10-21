"""Post-processing of U-Net segmentation masks into word bounding boxes.

The U-Net predicts three classes per pixel: background (0), box fill (1) and
box border (2). Borders separate neighbouring words, so keeping only the fill
class leaves every word as an isolated region, and each region becomes one
box. Predicted boxes are evaluated against the ground truth by one-to-one
IoU matching, giving box-level precision, recall and F1.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

FILL_CLASS = 1

# Los bordes se pintan con grosor 2 en models/u-net/setup.py y dejan el relleno
# 2 px por dentro de la caja real en cada lado.
BORDER_PAD = 2


def mask_to_boxes(mask, min_area=20, pad=BORDER_PAD):
    """Extract word bounding boxes from a class-index segmentation mask.

    Connected regions of the fill class are found with 4-connectivity, so two
    words touching only diagonally at a corner stay separate. Regions smaller
    than ``min_area`` pixels are dropped as noise, and every remaining box is
    grown by ``pad`` pixels on each side to undo the border painted inside
    the original box.

    Args:
        mask (np.ndarray): ``(H, W)`` mask with class indices (0, 1, 2), such
            as the ``argmax`` of the U-Net output with the padding cropped.
        min_area (int, optional): Minimum number of fill pixels for a region
            to be kept. Defaults to 20.
        pad (int, optional): Pixels added to each side of every box. Defaults
            to ``BORDER_PAD``.

    Returns:
        np.ndarray: ``(N, 4)`` int32 array of boxes in
        ``(x_min, y_min, x_max, y_max)`` pixel coordinates (inclusive, as in
        ``*_boxes.json``), clipped to the mask limits.
    """
    fill = (np.asarray(mask) == FILL_CLASS).astype(np.uint8)
    h, w = fill.shape

    _, _, stats, _ = cv2.connectedComponentsWithStats(fill, connectivity=4)

    boxes = []
    for x, y, bw, bh, area in stats[1:]:  # la etiqueta 0 es el fondo
        if area < min_area:
            continue
        boxes.append([
            max(x - pad, 0),
            max(y - pad, 0),
            min(x + bw - 1 + pad, w - 1),
            min(y + bh - 1 + pad, h - 1),
        ])

    return np.array(boxes, dtype=np.int32).reshape(-1, 4)


def box_intersection(boxes_a, boxes_b):
    """Compute the intersection area between every pair of boxes.

    Coordinates are inclusive pixel indices, so a box spans
    ``x_max - x_min + 1`` pixels in width.

    Args:
        boxes_a (np.ndarray): ``(N, 4)`` boxes in
            ``(x_min, y_min, x_max, y_max)`` coordinates.
        boxes_b (np.ndarray): ``(M, 4)`` boxes in the same format.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The ``(N, M)``
        intersection areas, and the ``(N,)`` and ``(M,)`` areas of each set.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0 + 1, 0, None) * np.clip(y1 - y0 + 1, 0, None)

    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    return inter, area_a, area_b


def box_iou(boxes_a, boxes_b):
    """Compute the IoU between every pair of boxes from two sets.

    Args:
        boxes_a (np.ndarray): ``(N, 4)`` boxes in
            ``(x_min, y_min, x_max, y_max)`` coordinates.
        boxes_b (np.ndarray): ``(M, 4)`` boxes in the same format.

    Returns:
        np.ndarray: ``(N, M)`` float matrix where entry ``(i, j)`` is the IoU
        of ``boxes_a[i]`` and ``boxes_b[j]``.
    """
    inter, area_a, area_b = box_intersection(boxes_a, boxes_b)
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-9)


def match_boxes(pred, gt, iou_threshold=0.5):
    """Match predicted boxes to ground-truth boxes one to one.

    The assignment maximises the total IoU (Hungarian algorithm), so every
    box is used at most once, and a matched pair counts as a true positive
    only if its IoU reaches ``iou_threshold``.

    Args:
        pred (np.ndarray): ``(N, 4)`` predicted boxes.
        gt (np.ndarray): ``(M, 4)`` ground-truth boxes.
        iou_threshold (float, optional): Minimum IoU for a match to count as
            a true positive. Defaults to 0.5.

    Returns:
        list[tuple[int, int, float]]: ``(pred_index, gt_index, iou)`` for
        every true positive.
    """
    iou = box_iou(pred, gt)
    if iou.size == 0:
        return []

    rows, cols = linear_sum_assignment(-iou)

    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]


def box_metrics(tp, n_pred, n_gt):
    """Compute box-level precision, recall and F1 from accumulated counts.

    Counts should be summed over the whole evaluation set before calling
    this, rather than averaging per-image scores, so receipts with many
    words weigh more.

    Args:
        tp (int): Number of true positives.
        n_pred (int): Number of predicted boxes.
        n_gt (int): Number of ground-truth boxes.

    Returns:
        dict: ``precision``, ``recall`` and ``f1``, each 0 when undefined.
    """
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def count_merges_splits(pred, gt, coverage=0.5):
    """Count merged and split words between predicted and ground-truth boxes.

    A box *covers* another when more than ``coverage`` of the other box's
    area lies inside it.

        - Merge: a predicted box covers two or more ground-truth boxes, so
          several words come out as one box (the borders failed to separate
          them).
        - Split: a ground-truth box covers two or more predicted boxes, so
          one word comes out in pieces (a spurious border or a hole in the
          fill).

    Args:
        pred (np.ndarray): ``(N, 4)`` predicted boxes.
        gt (np.ndarray): ``(M, 4)`` ground-truth boxes.
        coverage (float, optional): Fraction of a box's area that must lie
            inside another to be covered by it. Defaults to 0.5.

    Returns:
        dict: ``merges`` (predicted boxes covering 2+ words), ``merged_gt``
        (words inside those boxes), ``splits`` (words covering 2+ predicted
        boxes) and ``split_pred`` (predicted boxes inside those words).
    """
    inter, area_pred, area_gt = box_intersection(pred, gt)

    gt_in_pred = inter / np.maximum(area_gt[None, :], 1e-9) > coverage    # (N, M)
    pred_in_gt = inter / np.maximum(area_pred[:, None], 1e-9) > coverage  # (N, M)

    gt_per_pred = gt_in_pred.sum(axis=1)
    pred_per_gt = pred_in_gt.sum(axis=0)

    return {
        "merges": int((gt_per_pred >= 2).sum()),
        "merged_gt": int(gt_per_pred[gt_per_pred >= 2].sum()),
        "splits": int((pred_per_gt >= 2).sum()),
        "split_pred": int(pred_per_gt[pred_per_gt >= 2].sum()),
    }


def evaluate_boxes(pairs, iou_threshold=0.5, coverage=0.5):
    """Evaluate predicted boxes against the ground truth over a whole set.

    True positives, box counts, merges and splits are summed over every
    image before computing precision, recall and F1.

    Args:
        pairs (Iterable[tuple[np.ndarray, np.ndarray]]): ``(pred, gt)`` boxes
            for every image.
        iou_threshold (float, optional): Minimum IoU for a true positive.
            Defaults to 0.5.
        coverage (float, optional): Coverage used to count merges and
            splits. Defaults to 0.5.

    Returns:
        dict: ``precision``, ``recall``, ``f1``, ``mean_iou`` (over the true
        positives), the totals ``tp``, ``n_pred`` and ``n_gt``, and the
        merge and split counts from ``count_merges_splits``.
    """
    totals = {"tp": 0, "n_pred": 0, "n_gt": 0, "merges": 0, "merged_gt": 0, "splits": 0, "split_pred": 0}
    ious = []

    for pred, gt in pairs:
        pred = np.asarray(pred).reshape(-1, 4)
        gt = np.asarray(gt).reshape(-1, 4)

        matches = match_boxes(pred, gt, iou_threshold)
        ious.extend(iou for _, _, iou in matches)

        totals["tp"] += len(matches)
        totals["n_pred"] += len(pred)
        totals["n_gt"] += len(gt)
        for k, v in count_merges_splits(pred, gt, coverage).items():
            totals[k] += v

    return {
        **box_metrics(totals["tp"], totals["n_pred"], totals["n_gt"]),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        **totals,
    }


def main():
    """Evaluate ``mask_to_boxes`` on a folder of masks.

    Run on the ground-truth masks to check the post-processing itself, which
    gives the upper bound any model can reach with it::

        python models/u-net/boxes.py --masks datasets/u-net/test_masks --boxes datasets/u-net/test_boxes.json
    """
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evalúa por cajas las máscaras de una carpeta frente a las cajas reales.")
    parser.add_argument('--masks', type=str, default='datasets/u-net/test_masks', help='Carpeta con las máscaras de clases (default: datasets/u-net/test_masks)')
    parser.add_argument('--boxes', type=str, default='datasets/u-net/test_boxes.json', help='JSON con las cajas reales (default: datasets/u-net/test_boxes.json)')
    parser.add_argument('--min-area', type=int, default=20, help='Área mínima de una región de relleno (default: 20)')
    parser.add_argument('--pad', type=int, default=BORDER_PAD, help=f'Píxeles añadidos a cada lado de la caja (default: {BORDER_PAD})')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU mínimo para un acierto (default: 0.5)')
    args = parser.parse_args()

    gt_boxes = json.loads(Path(args.boxes).read_text())

    def pairs():
        for name, gt in gt_boxes.items():
            mask = cv2.imread(str(Path(args.masks) / name), cv2.IMREAD_UNCHANGED)
            yield mask_to_boxes(mask, args.min_area, args.pad), gt

    metrics = evaluate_boxes(pairs(), iou_threshold=args.iou)
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}" if isinstance(v, float) else f"{k:>10}: {v}")


if __name__ == "__main__":
    main()
