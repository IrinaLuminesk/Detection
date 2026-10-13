#Phiên bản 2 của DatasetLoader, giải quyết các vấn đề như đổi format của bounding box

# Hiện tại có 6 format được dùng phổ biến
# 1. COCO — [x,y,width,height]
# x, y: top-left corner.
# width, height: box dimensions.
# Used in COCO dataset annotations.
# Example: [100, 200, 50, 80]

# 2.XYXY — [x_min, y_min, x_max, y_max]
# Top-left and bottom-right corners.
# Common in torchvision object detection.
# Example: [100, 200, 150, 280]

# 3.YOLO — [x_center, y_center, width, height]
# Center coordinates and dimensions.
# Usually normalized to the range 0–1 relative to image size.
# Example: [0.25, 0.40, 0.10, 0.16]

# 4.CXCYWH — [x_center,y_center,width,height]
# Center coordinates and dimensions.
# Common in models such as DETR.
# May use pixels or normalized coordinates, depending on the implementation.

# 5. XYXY normalized — [x_min, y_min, x_max, y_max]
# Same coordinate convention as XYXY.
# Coordinates are divided by image width and height, giving values usually between 0 and 1.
# Often used in normalized annotation pipelines.

# 6. Pascal VOC — [xmin, ymin, xmax, ymax]
# Corner-coordinate format commonly found in Pascal VOC XML annotations.
# Similar to XYXY, but annotation conventions can differ, especially regarding inclusive pixel coordinates.
# Example: [100, 200, 150, 280]

from utils.BBox_Utils import convert_bbox
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

class ObjectRecognitionDataset(Dataset):
    def __init__(self, label_root, image_root, original_format, convert_format):
        self.label_root = Path(label_root) #đường dẫn tới label và bounding_box
        self.image_root = Path(image_root) #đường dẫn tới ảnh
        self.original_format = original_format
        self.convert_format = convert_format

        self.data = self.get_data()
    
    def get_dimensions(self, image_path):
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    
    def get_label_and_bbox(self, label_path, image_path):
        labels = []
        bboxes = []
        img_size = self.get_dimensions(image_path)
        with open(label_path, "r") as f:
            for line in f:
                row = line.strip().split()
                if len(row) != 0:
                    labels.append(int(row[0]))
                    bbox = [float(i) for i in row[1:]]
                    bbox = convert_bbox(bbox, 
                                        from_format=self.original_format, 
                                        to_format=self.convert_format,
                                        image_size=img_size)
                    bboxes.append(bbox)
        #Trường hợp không có object
        if len(bboxes) == 0:
            bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        return labels, bboxes
    
    def get_data(self):
        accepted_format = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".heic"]
        images = sorted([img for img in self.image_root.rglob("*") if (img.suffix.lower() in accepted_format and img.is_file())])

        accepted_images = [img.stem for img in images]
        accepted_labels = sorted([label for label in self.label_root.rglob("*") if (label.suffix.lower() == ".txt" and label.stem in accepted_images and label.is_file())])
        all_labels, all_bboxes = [], []
        
        for l_path, image_path in zip(accepted_labels, images):
            labels, bboxes = self.get_label_and_bbox(l_path, image_path) #Đây là label và bbox của từng ảnh
            all_labels.append(labels)
            all_bboxes.append(bboxes)
        return list(zip(images, all_labels, all_bboxes))
    
    
    def __getitem__(self, index):
        image, all_labels, all_bboxes = self.data[index]
        
        image = Image.open(image).convert("RGB")
        image = F.to_tensor(image)
        
        target = {
            "boxes": torch.as_tensor(all_bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(all_labels, dtype=torch.int64),
        }

        return image, target
    
    def __len__(self):
        return len(self.data)
    
class DatasetLoader():
    def __init__(self, label_root, image_root, batch_size, original_format, convert_format) -> None:
        self.label_root = label_root
        self.image_root = image_root
        self.batch_size = batch_size
        self.original_format = original_format
        self.convert_format = convert_format
    def collate_fn(self, batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    def dataset_loader(self, type: str):
        type = type.lower() 
        dataset = ObjectRecognitionDataset(
            label_root=self.label_root,
            image_root=self.image_root,
            original_format=self.original_format,
            convert_format=self.convert_format
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if type == "train" else False,
            num_workers=2 if type == "train" else 0,
            pin_memory=True if type == "train" else False,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=self.collate_fn,
        )
        return loader
        

