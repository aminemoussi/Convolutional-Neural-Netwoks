import math
from os import pread

import torch
import torch.nn as nn
import torchvision

from . import anchor_handling, core, region_proposal_network, roi_head


class faster_rcnn(nn.Module):
    def __init__(self, model_config, num_classes=21):
        super(faster_rcnn, self).__init__()
        self.model_config = model_config
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # vgg16 except last classification layer
        self.backbone = vgg16.features[:-1]
        self.rpn = region_proposal_network.RegionProposalNetwork(in_channels=512)
        self.roi_head = roi_head.ROIHead(
            model_config, num_classes=num_classes, in_channels=512
        )

        # freeze firs 10 layers
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config["min_im_size"]
        self.max_size = model_config["max_im_size"]

    def normalize_resize_image_and_boxes(self, image, bboxes):
        dtype, device = image.dtype, image.device

        # normalization (image - mean)/std
        # center around 0
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]

        # resizing to 1000x600
        # Make smallest side = 600px, largest side ≤ 1000px
        # Original: 400x800 (height x width)
        # - min_size = 400, max_size = 800
        # - Scale for min: 600/400 = 1.5
        # - Scale for max: 1000/800 = 1.25
        # - Chosen scale: min(1.5, 1.25) = 1.25
        # Result: 500x1000 (400*1.25, 800*1.25)
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(
            float(self.min_size) / min_size, float(self.max_size) / max_size
        )
        scale_factor = scale.item()

        # resize to a specific scale using bilinear interpolation
        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )

        # resize bboxes
        if bboxes is not None:
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]

            ratio_height, ratio_width = ratios

            # Scale all box coordinates
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width  # Scale X coordinates
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height  # Scale Y coordinates
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        return image, bboxes

    def forward(self, image, target=None):
        old_shape = image.shape[-2:]
        if self.training:
            image, bboxes = self.normalize_resize_image_and_boxes(
                image, target["bboxes"]
            )
            target["bboxes"] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)

        # calling backbone
        feat = self.backbone(image)

        # rpn + roi_head
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output["proposals"]

        # roi head + get final bboxes
        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)
        # in inference return bboxes back to their original sizes
        if not self.training:
            frcnn_output["boxes"] = core.transform_boxes_to_original_size(
                frcnn_output["boxes"], image.shape[-2:], old_shape
            )

        return rpn_output, frcnn_output
