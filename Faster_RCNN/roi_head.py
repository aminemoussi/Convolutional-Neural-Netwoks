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

    def assign_target_to_proposal(self, proposals, gt_boxes, gt_lables):
        iou_matrix = core.iou(gt_boxes, proposals)
        # best match per proposal
        iou_best_match, best_gt_index = iou_matrix.max(dim=0)

        below_low_thrshold = iou_best_match < 0.5
        best_gt_index[below_low_thrshold] = -1

        matched_gt_boxes_per_proposal = gt_boxes[best_gt_index.clamp(min=0)]

        labels = gt_lables[best_gt_index.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)

        background_proposal = best_gt_index == -1
        labels[background_proposal] = 0
        return labels, matched_gt_boxes_per_proposal

    def forward(self, feat, proposals, img_shape, target):
        if self.training and target is not None:
            gt_boxes = target["bboxes"][0]
            gt_lables = target["labels"][0]

            # assign_target_to_proposal
            labels, matched_gt_boxes_per_proposal = self.assign_target_to_proposal(
                proposals, gt_boxes, gt_lables
            )
