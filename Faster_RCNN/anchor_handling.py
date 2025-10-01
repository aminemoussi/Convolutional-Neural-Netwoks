import math

import torch
from torch import torch_version


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


def boxes_to_transformation_targets(gt_boxes, anchors_or_proposals):
    r"""
    Given all anchor boxes or proposals in image and their respective
    ground truth assignments, we use the x1,y1,x2,y2 coordinates of them
    to get tx,ty,tw,th transformation targets for all anchor boxes or proposals
    :param ground_truth_boxes: (anchors_or_proposals_in_image, 4)
        Ground truth box assignments for the anchors/proposals
    :param anchors_or_proposals: (anchors_or_proposals_in_image, 4) Anchors/Proposal boxes
    :return: regression_targets: (anchors_or_proposals_in_image, 4) transformation targets tx,ty,tw,th
        for all anchors/proposal boxes
    """

    # center_x, center_y and height, width for anchors_or_proposals
    w_anchors = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h_anchors = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w_anchors
    center_y = anchors_or_proposals[:, 1] + 0.5 * h_anchors

    # same for gt boxes
    w_gt = gt_boxes[:, 2] - gt_boxes[:, 0]
    h_gt = gt_boxes[:, 3] - gt_boxes[:, 1]
    center_x_gt = gt_boxes[:, 0] + 0.5 * w_gt
    center_y_gt = gt_boxes[:, 1] + 0.5 * h_gt

    # now getting the transformation that should be calculated by 1x1 conv
    # layer

    dx = (center_x_gt - center_x) / w_anchors
    dy = (center_y_gt - center_y) / h_anchors
    dw = torch.log(w_gt / w_anchors)
    dh = torch.log(h_gt / h_anchors)

    regression_params = torch.stack((dx, dy, dw, dh), dim=1)

    return regression_params


def sample_positive_negative(labels, pos_count, total_count):
    r"""
        Before sampling:
        Total anchors: 16,000
        Positive anchors: 150 (0.9%)
        Negative anchors: 15,800 (98.8%)
        Ignored anchors: 50 (0.3%)

    After sampling:
        Sampled positives: 128 (50% of batch)
        Sampled negatives: 128 (50% of batch)
        Total for training: 256 anchors

    This balanced batch allows the RPN to learn both:
        What objects look like (from positive anchors)
        What background looks like (from negative anchors)
    """

    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]

    num_pos = min(positive.numel(), pos_count)
    num_neg = total_count - num_pos
    num_neg = min(negative.numel(), num_neg)

    pos_indc_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    neg_indc_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_indc = positive[pos_indc_perm]
    neg_indc = negative[neg_indc_perm]

    # we return boolean masks of the selected samples
    pos_mask = torch.zeros_like(labels, dtype=torch.bool)
    neg_mask = torch.zeros_like(labels, dtype=torch.bool)

    pos_mask[pos_indc] = True
    neg_mask[neg_indc] = True

    return neg_mask, pos_mask
