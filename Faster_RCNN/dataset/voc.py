import glob
import os
import random
import xml.etree.ElementTree as ET

import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

"""
The voc dataloader does the following:
    Knows where all the images and labels are stored
    Understands how to read annotation files
    Prepares data in the exact format the model needs
    Can even do simple data augmentation (flipping)

Data format:
    Images: .jpg files
    Annotations: .xml files with object information
Example: 2007_000027.xml contains:
  - Image size: 500x375
  - Objects: person [195, 102, 294, 375], horse [108, 162, 274, 348]
"""


def load_images_and_anns(im_dir, ann_dir, label2idx):
    '''
    # XML content:
    """
    <annotation>
        <size><width>500</width><height>375</height></size>
        <object>
            <name>person</name>
            <bndbox>
                <xmin>195</xmin><ymin>102</ymin><xmax>294</xmax><ymax>375</ymax>
            </bndbox>
        </object>
    </annotation>
    """

    # Becomes Python dictionary:
    im_info = {
        'img_id': '2007_000027',
        'filename': '/path/to/2007_000027.jpg',
        'width': 500, 'height': 375,
        'detections': [
            {'label': 15, 'bbox': [194, 101, 293, 374]}  # person class
        ]
    }
    '''

    im_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, "*.xml"))):
        im_info = {}
        im_info["img_id"] = os.path.basename(ann_file).split(".xml")[0]
        im_info["filename"] = os.path.join(im_dir, "{}.jpg".format(im_info["img_id"]))
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        im_info["width"] = width
        im_info["height"] = height
        detections = []

        for obj in ann_info.findall("object"):
            det = {}
            label = label2idx[obj.find("name").text]
            bbox_info = obj.find("bndbox")
            bbox = [
                int(float(bbox_info.find("xmin").text)) - 1,
                int(float(bbox_info.find("ymin").text)) - 1,
                int(float(bbox_info.find("xmax").text)) - 1,
                int(float(bbox_info.find("ymax").text)) - 1,
            ]
            det["label"] = label
            det["bbox"] = bbox
            detections.append(det)
        im_info["detections"] = detections
        im_infos.append(im_info)
    print("Total {} images found".format(len(im_infos)))
    return im_infos


class VOCDataset(Dataset):
    def __init__(self, split, im_dir, ann_dir):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        classes = [
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor",
        ]
        classes = sorted(classes)
        classes = ["background"] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_dir, self.label2idx)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info["filename"])
        to_flip = False
        if self.split == "train" and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        targets["bboxes"] = torch.as_tensor(
            [detection["bbox"] for detection in im_info["detections"]]
        )
        targets["labels"] = torch.as_tensor(
            [detection["label"] for detection in im_info["detections"]]
        )
        if to_flip:
            for idx, box in enumerate(targets["bboxes"]):
                x1, y1, x2, y2 = box
                w = x2 - x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets["bboxes"][idx] = torch.as_tensor([x1, y1, x2, y2])
        return im_tensor, targets, im_info["filename"]
