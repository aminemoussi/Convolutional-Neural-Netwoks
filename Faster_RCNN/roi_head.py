import math

import anchor_handling
import core
import torch
import torch.nn as nn
import torchvision
from torch._C import dtype
from torch.cuda import _compile_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RPN on ", device)


class ROIHead(nn.Module):
    def __init__(self, num_classes=21, in_channels=512):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.pool_size = 7
        self.fc_inner_dim = 1024

        self.fc6 = nn.Linear(
            in_channels * self.pool_size * self.pool_size, self.fc_inner_dim
        )
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)

        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)

    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        r"""
        Given a set of proposals and ground truth boxes and their respective labels.
        Use IOU to assign these proposals to some gt box or background
        :param proposals: (number_of_proposals, 4)
        :param gt_boxes: (number_of_gt_boxes, 4)
        :param gt_labels: (number_of_gt_boxes)
        :return:
            labels: (number_of_proposals)
            matched_gt_boxes: (number_of_proposals, 4)
        """
        # Get IOU Matrix between gt boxes and proposals
        iou_matrix = get_iou(gt_boxes, proposals)
        # For each gt box proposal find best matching gt box
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        background_proposals = (best_match_iou < self.iou_threshold) & (
            best_match_iou >= self.low_bg_iou
        )
        ignored_proposals = best_match_iou < self.low_bg_iou

        # Update best match of low IOU proposals to -1
        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2

        # Get best marching gt boxes for ALL proposals
        # Even background proposals would have a gt box assigned to it
        # Label will be used to ignore them later
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]

        # Get class label for all proposals according to matching gt boxes
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)

        # Update background proposals to be of label 0(background)
        labels[background_proposals] = 0

        # Set all to be ignored anchor labels as -1(will be ignored)
        labels[ignored_proposals] = -1

        return labels, matched_gt_boxes_for_proposals

    def forward(self, feat, proposals, img_shape, target):
        if self.training and target is not None:
            gt_boxes = target["bboxes"][0]
            gt_lables = target["labels"][0]

            # assign_target_to_proposal
            labels, matched_gt_boxes_per_proposal = self.assign_target_to_proposals(
                proposals, gt_boxes, gt_lables
            )
