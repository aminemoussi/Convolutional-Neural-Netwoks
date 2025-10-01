import torch


def iou(gt, det):
    r"""
    IOU between two sets of boxes
    :param boxes1: (Tensor of shape N x 4)
    :param boxes2: (Tensor of shape M x 4)
    :return: IOU matrix of shape N x M
    """
    # det_x1, det_y1, det_x2, det_y2 = det
    # gt_x1, gt_y1, gt_x2, gt_y2 = gt
    #
    # x_left = max(det_x1, gt_x1)
    # y_top = max(det_y1, gt_y1)
    #
    # x_right = min(det_x2, gt_x2)
    # y_bottom = min(det_y2, gt_y2)
    #
    # if x_right < x_left or y_bottom < y_top:
    #     return 0.0
    #
    # intersection = (x_right - x_left) * (y_bottom - y_top)
    # gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    # det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    #
    # union = gt_area + det_area - intersection + 1e-6

    # Areas
    det_area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    # x1, y1, x2, y2 for intersection area
    x_left = torch.max(gt[:, None, 0], det[:, 0])
    y_top = torch.max(gt[:, None, 1], det[:, 1])
    x_right = torch.min(gt[:, None, 2], det[:, 2])
    y_bottom = torch.min(gt[:, None, 3], det[:, 3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(
        min=0
    )
    union_area = gt_area[:, None] + det_area - intersection_area

    iou = intersection_area / union_area
    return iou
