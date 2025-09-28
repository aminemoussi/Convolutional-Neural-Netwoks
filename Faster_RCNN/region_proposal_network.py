import math

import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RPN on ", device)


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

    # original rgb img + its ft map + target (a dict of grount truths + labels)
    def forward(self, image, feat, target):
        # go through conv layers
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)  # either foreground or background
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        # generate anchor boxes
        anchors = self.generate_anchors(image, feat)
