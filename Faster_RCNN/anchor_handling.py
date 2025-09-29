import math

import torch


def apply_regression_pred_to_anchors_or_proposals(
    box_transform_pred, anchors_or_proposals
):
    r"""
    Given the transformation parameter predictions for all
    input anchors or proposals, transform them accordingly
    to generate predicted proposals or predicted boxes
    :param box_transform_pred: (num_anchors_or_proposals, num_classes, 4)
    :param anchors_or_proposals: (num_anchors_or_proposals, 4)
    :return pred_boxes: (num_anchors_or_proposals, num_classes, 4)
    x_proposal = x + dx * w
    y_proposal = y + dy * h
    w_proposal = w * exp(dw)
    h_proposal = h * exp(dh)
    """

    # Convert from (x1, y1, x2, y2) to (center_x, center_y, width, height)
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]  # x2-x1
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]  # y2 - y1
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h

    # extracting adjustments
    dx = box_transform_pred[..., 0]  # center x adj
    dy = box_transform_pred[..., 1]  # center y adj
    dw = box_transform_pred[..., 2]  # width adj
    dh = box_transform_pred[..., 3]  # height adj

    # prevent very large scaling (63x)
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    # new centers
    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    # Anchor: w=100, center_x=150, predicted dx=0.2
    # pred_center_x = 0.2 * 100 + 150 = 20 + 150 = 170

    # new scale
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]
    # Anchor: w=100, predicted dw=0.2
    # pred_w = exp(0.2) * 100 ≈ 1.221 * 100 = 122.1

    # back to corner format
    pred_x1 = pred_center_x - 0.5 * pred_w
    pred_x2 = pred_center_x + 0.5 * pred_w
    pred_y1 = pred_center_y - 0.5 * pred_h
    pred_y2 = pred_center_y + 0.5 * pred_h

    pred_box = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=2)
    # (anchors, classes, 4)

    return pred_box


def clamp_boxes_to_image_boundaries(boxes, image_shape):
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]

    h, w = image_shape[-2:]

    boxes_x1 = boxes_x1.clamp(min=0, max=w)
    boxes_x2 = boxes_x2.clamp(min=0, max=w)
    boxes_y1 = boxes_y1.clamp(min=0, max=h)
    boxes_y2 = boxes_y2.clamp(min=0, max=h)

    boxes = torch.cat(
        (
            boxes_x1[..., None],
            boxes_y1[..., None],
            boxes_x2[..., None],
            boxes_y2[..., None],
        ),
        dim=-1,
    )
    return boxes
