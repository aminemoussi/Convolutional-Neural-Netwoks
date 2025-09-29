import math

import anchor_handling
import torch
import torch.nn as nn
import torchvision
from torch.cuda import _compile_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RPN on ", device)


# def apply_regression_pred_to_anchors_or_proposals(
#     box_transform_pred, anchors_or_proposals
# ):
#     r"""
#     Given the transformation parameter predictions for all
#     input anchors or proposals, transform them accordingly
#     to generate predicted proposals or predicted boxes
#     :param box_transform_pred: (num_anchors_or_proposals, num_classes, 4)
#     :param anchors_or_proposals: (num_anchors_or_proposals, 4)
#     :return pred_boxes: (num_anchors_or_proposals, num_classes, 4)
#     x_proposal = x + dx * w
#     y_proposal = y + dy * h
#     w_proposal = w * exp(dw)
#     h_proposal = h * exp(dh)
#     """
#
#     # Convert from (x1, y1, x2, y2) to (center_x, center_y, width, height)
#     w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]  # x2-x1
#     h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]  # y2 - y1
#     center_x = anchors_or_proposals[:, 0] + 0.5 * w
#     center_y = anchors_or_proposals[:, 1] + 0.5 * h
#
#     # extracting adjustments
#     dx = box_transform_pred[..., 0]  # center x adj
#     dy = box_transform_pred[..., 1]  # center y adj
#     dw = box_transform_pred[..., 2]  # width adj
#     dh = box_transform_pred[..., 3]  # height adj
#
#     # prevent very large scaling (63x)
#     dw = torch.clamp(dw, max=math.log(1000.0 / 16))
#     dh = torch.clamp(dh, max=math.log(1000.0 / 16))
#
#     # new centers
#     pred_center_x = dx * w[:, None] + center_x[:, None]
#     pred_center_y = dy * h[:, None] + center_y[:, None]
#     # Anchor: w=100, center_x=150, predicted dx=0.2
#     # pred_center_x = 0.2 * 100 + 150 = 20 + 150 = 170
#
#     # new scale
#     pred_w = torch.exp(dw) * w[:, None]
#     pred_h = torch.exp(dh) * h[:, None]
#     # Anchor: w=100, predicted dw=0.2
#     # pred_w = exp(0.2) * 100 ≈ 1.221 * 100 = 122.1
#
#     # back to corner format
#     pred_x1 = pred_center_x - 0.5 * pred_w
#     pred_x2 = pred_center_x + 0.5 * pred_w
#     pred_y1 = pred_center_y - 0.5 * pred_h
#     pred_y2 = pred_center_y + 0.5 * pred_h
#
#     pred_box = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=2)
#     # (anchors, classes, 4)
#
#     return pred_box


# def clamp_boxes_to_image_boundaries(boxes, image_shape):
#     boxes_x1 = boxes[..., 0]
#     boxes_y1 = boxes[..., 1]
#     boxes_x2 = boxes[..., 2]
#     boxes_y2 = boxes[..., 3]
#
#     h, w = image_shape[-2:]
#
#     boxes_x1 = boxes_x1.clamp(min=0, max=w)
#     boxes_x2 = boxes_x2.clamp(min=0, max=w)
#     boxes_y1 = boxes_y1.clamp(min=0, max=h)
#     boxes_y2 = boxes_y2.clamp(min=0, max=h)
#
#     boxes = torch.cat(
#         (
#             boxes_x1[..., None],
#             boxes_y1[..., None],
#             boxes_x2[..., None],
#             boxes_y2[..., None],
#         ),
#         dim=-1,
#     )
#     return boxes


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super(RegionProposalNetwork, self).__init__()
        # anchor_boxes defenition
        self.scales = [128, 256, 512]  # areas of an anchor box
        self.aspect_ratios = [0.5, 1, 2]
        self.num_anchors = len(self.scales) * len(
            self.aspect_ratios
        )  # each FT map cell is going to have 9 anchor boxes

        # 1st 3x3 conv layer, this goes through ReLU also
        self.rpn_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        # 2nd layer 1x1 for classification, output channels = n° anchors (K)
        self.cls_layer = nn.Conv2d(
            in_channels, self.num_anchors, kernel_size=1, stride=1
        )

        # 3rd layer for bbox adjustment, output = 4K
        self.bbox_reg_layer = nn.Conv2d(
            in_channels, self.num_anchors * 4, kernel_size=1, stride=1
        )

    def generate_anchors(self, image, feat):
        # getting dimentions
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]

        # calculating stride
        # stride = 16 means 1 ft map point represents 16x16 from img
        stride_h = torch.tensor(
            image_h // grid_h, dtype=torch.int64, device=feat.device
        )
        stride_w = torch.tensor(
            image_w // grid_w, dtype=torch.int64, device=feat.device
        )

        # scales + aspect_ratios as tensors
        scales = torch.tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.tensor(
            self.aspect_ratios, dtype=feat.dtype, device=feat.device
        )

        # now finding the zero centered anchors
        # we have h/w = aspect_ratio, and h*w = 1
        # start with height and width ratios of a 1 area box
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        # now scale up for different areas to get possible coordonates
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)  # 9 values
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        # defining the coordonates of the anchors
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()  # 9x4 anchor boxes

        # place 9 template boxes at every one of the 50x37 grid
        # shifts in x and y axes
        shifts_x = (
            torch.arange(0, grid_w, dtype=torch.int64, device=feat.device) * stride_w
        )
        shifts_y = (
            torch.arange(0, grid_h, dtype=torch.int64, device=feat.device) * stride_h
        )

        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

        # For each of the 1850 grid locations, take each of the 9 base anchors and add the location's shift to it
        anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
        anchors = anchors.reshape(-1, 4)  # merge first 2 dims

        return anchors

    def filter_proposals(self, proposals, cls_scores, image_shape):
        # select top k=10000 proposals
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(10000)
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]

        # clamp to image boundaries
        proposals = anchor_handling.clamp_boxes_to_image_boundaries(
            proposals, image_shape
        )

        # perform NMS 0.7 IoU threshold
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, 0.7)
        # keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        # keep_mask[keep_indices] = True
        # keep_indices = torch.where(keep_mask)[0]
        # post_nms_keep_indeces = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]

        # keeping only top 2000
        proposals = proposals[keep_indices[:2000]]
        cls_scores = cls_scores[keep_indices[:2000]]
        # proposals = proposals[post_nms_keep_indeces[:2000]]
        # cls_scores = cls_scores[post_nms_keep_indeces[:2000]]

        return proposals, cls_scores

    # original rgb img + its ft map + target (a dict of grount truths + labels)
    def forward(self, image, feat, target):
        # go through conv layers
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)  # objectness scores
        box_transform_pred = self.bbox_reg_layer(rpn_feat)  # anchor box adjustments

        # generate anchor boxes
        anchors = self.generate_anchors(image, feat)  # thousands across the img

        # cls_scores reshape
        number_of_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        # (Batch_Size, num_anchors, H_feat, W_feat)
        cls_scores = cls_scores.reshape(-1, 1)
        # (Batch_Size × H_feat × W_feat × num_anchors, 1)
        # so that each anchor gets objectness score

        # box_transform reshape
        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1],
        )
        # (Batch_Size, num_anchors × 4, H_feat, W_feat)
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)
        # (Batch_Size × H_feat × W_feat × num_anchors, 4)
        # each anchor gets 4 values for box adjustments

        # transforming generated anchorboxes
        proposals = anchor_handling.apply_regression_pred_to_anchors_or_proposals(
            box_transform_pred.detach().reshape(-1, 1, 4), anchors
        )
        proposals = proposals.reshape(proposals.size(0), 4)

        # filtering proposals
        proposals, cls_scores = self.filter_proposals(
            proposals, cls_scores.detach(), image.shape
        )

        rpn_out = {"proposals": proposals, "scores": cls_scores}

        if not self.training or target is None:
            return rpn_out
