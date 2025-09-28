def iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)

    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)

    union = gt_area + det_area - intersection + 1e-6
    return intersection / union
