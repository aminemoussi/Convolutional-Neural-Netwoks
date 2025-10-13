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
    def __init__(self, model_config, num_classes=21, in_channels=512):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.pool_size = 7
        self.fc_inner_dim = 1024

        self.roi_batch_size = model_config["roi_batch_size"]
        self.roi_pos_count = int(model_config["roi_pos_fraction"] * self.roi_batch_size)
        self.iou_threshold = model_config["roi_iou_threshold"]
        self.low_bg_iou = model_config["roi_low_bg_iou"]
        self.nms_threshold = model_config["roi_nms_threshold"]
        self.topK_detections = model_config["roi_topk_detections"]
        self.low_score_threshold = model_config["roi_score_threshold"]

        self.fc6 = nn.Linear(
            in_channels * self.pool_size * self.pool_size, self.fc_inner_dim
        )
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)

        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)

        torch.nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_reg_layer.bias, 0)

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
        iou_matrix = core.iou(gt_boxes, proposals)
        # For each proposal find best matching gt box
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        background_proposals = (best_match_iou < self.iou_threshold) & (
            best_match_iou >= self.low_bg_iou
        )
        ignored_proposals = best_match_iou < self.low_bg_iou

        # Update best match of low IOU proposals to -1
        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2

        # Get best matching gt boxes for ALL proposals
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
            proposals = torch.cat([proposals, target["bboxes"][0]], dim=0)

            gt_boxes = target["bboxes"][0]
            gt_lables = target["labels"][0]

            # assign_target_to_proposal
            labels, matched_gt_boxes_per_proposal = self.assign_target_to_proposals(
                proposals, gt_boxes, gt_lables
            )

            # pos + neg sampling
            pos_samples_indx_mask, neg_samples_indx_mask = (
                anchor_handling.sample_positive_negative(
                    labels,
                    pos_count=self.roi_pos_count,
                    total_count=self.roi_batch_size,
                )
            )
            sampled_indxs = torch.where(pos_samples_indx_mask | neg_samples_indx_mask)[
                0
            ]

            # keep only sampled proposals and their labels
            proposals = proposals[sampled_indxs]
            labels = labels[sampled_indxs]
            matched_gt_boxes_per_proposal = matched_gt_boxes_per_proposal[sampled_indxs]

            # get how much the boxes are far from the gt_boxes
            # returns (proposals, 4)
            regression_targets = anchor_handling.boxes_to_transformation_targets(
                matched_gt_boxes_per_proposal, proposals
            )

        # Now ROI POOLING outside the training condition
        # to get the respective ft map section of each proposal
        # and turn preparet for classification + regression

        # first get the down scalling factor from img to ft_map
        # for vgg16 its 1/16 = 0.0625
        # spatial_scale = 0.0625
        size = feat.shape[-2:]  # Feature map size (e.g., 50x50)
        possible_scales = []
        for s1, s2 in zip(size, img_shape):  # img_shape is original image size
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        spatial_scale = possible_scales[0]

        # pooling proposals to the same size
        proposal_roi_pool_feats = torchvision.ops.roi_pool(
            feat, [proposals], output_size=self.pool_size, spatial_scale=spatial_scale
        )
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)

        # now run proposals through the fully connected layers
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
        cls_score = self.cls_layer(box_fc_7)  # 128x21 (21 classes)
        box_trans_pred = self.bbox_reg_layer(box_fc_7)  # 128x84 (4 trans per class 21)
        # reshape box_trans_pred to be 128x21x4
        num_boxes, num_classes = cls_score.shape
        box_trans_pred = box_trans_pred.reshape(num_boxes, num_classes, 4)

        # now getting the loss of classification + localization
        frcnn_output = {}
        if self.training and target is not None:
            classification_loss = torch.nn.functional.cross_entropy(cls_score, labels)

            # extracting only foreground proposals
            fg_proposals_indxs = torch.where(labels > 0)[0]
            fg_class_labels = labels[fg_proposals_indxs]
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_trans_pred[fg_proposals_indxs, fg_class_labels],
                regression_targets[fg_proposals_indxs],
                beta=1 / 9,
                reduction="sum",
            )

            localization_loss = localization_loss / labels.numel()
            frcnn_output["frcnn_classification_loss"] = classification_loss
            frcnn_output["frcnn_localization_loss"] = localization_loss
            return frcnn_output
