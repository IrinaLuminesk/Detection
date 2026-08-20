from pathlib import Path
from PIL import Image

import torch
from torchvision.tv_tensors import BoundingBoxes
from torchvision.ops import box_convert
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

class ObjectRecognitionDataset(Dataset):
    def __init__(self, label_root, image_root, std, mean, img_size, data_type, transform = True):
        self.label_root = Path(label_root)
        self.image_root = Path(image_root)
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.data_type = data_type
        self.transform = transform


        self.samples = self.Get_Data()
        self.data_transform = self.train_transform() if self.data_type == "train" else self.test_transform()
    def align_img_label_arrays(self, imgs, labels, fill=-1):
        #Assign -1 cho các ảnh không có label
        lookup = {guid: val for val, guid in labels}
        return [lookup.get(guid, fill) for _, guid, _ in imgs]
    def get_image_size(self, path):
        with Image.open(path) as img:
            width, height = img.size
            return (height, width)
    def get_label_and_bbox(self, path):
        labels = []
        bboxes = []
        with open(path, "r") as f:
            for line in f:
                row = line.strip().split()
                labels.append(int(row[0]))
                bboxes.append([float(i) for i in row[1:]])
        return labels, bboxes
    def Get_Data(self):
        images = [(img, img.stem, self.get_image_size(img)) for img in self.image_root.rglob("*") if img.is_file()]
        labels_n_bboxes = [(l_n_bb, l_n_bb.stem) for l_n_bb in self.label_root.rglob("*") if l_n_bb.is_file()]

        #Align label và image, nếu label = -1 nghĩa là ảnh đó thiếu label
        labels_n_bboxes = self.align_img_label_arrays(images, labels_n_bboxes)
        images = [(img, img_size) for img, _, img_size in images]

        samples = []
        for (img, img_size), l_n_bb in zip(images, labels_n_bboxes):
            if l_n_bb != -1:
                labels, bboxes = self.get_label_and_bbox(l_n_bb)

                bboxes = torch.tensor(bboxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
                # Chỉ lấy các bbox có tọa độ
                if len(bboxes) > 0 and len(labels) > 0:
                    #Đổi format do coco đang sd xywh
                    bboxes =  box_convert(bboxes, in_fmt="xywh", out_fmt="xyxy")
                    #Thêm hàm này để khi sử dụng transform, bbox sẽ được thay đổi theo
                    bboxes = BoundingBoxes(
                        bboxes,
                        format="XYXY",
                        canvas_size=img_size
                    )
                target = {
                    "boxes": bboxes,
                    "labels": labels,
                }
                #Đây là một tuples chứa đường dẫn ảnh, label và bbox
                samples.append((img, target))
        return samples

    def train_transform(self):
        if self.transform:
            return v2.Compose([
                v2.Resize(self.img_size),
                v2.RandomChoice([
                    v2.RandomHorizontalFlip(p=1),
                    v2.RandomVerticalFlip(p=1),
                    v2.RandomRotation(degrees=(-180, 180)),
                    # v2.ColorJitter(brightness=(1,2), contrast=(1,2)),
                    v2.Lambda(lambda x: x),
                    ]),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=self.mean,
                        std=self.std
                    )
                ])
        return v2.Compose([
            v2.Resize(self.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])
    def test_transform(self):
        return v2.Compose([
            v2.Resize(self.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img  = tv_tensors.Image(img)
        img, target = self.data_transform(img, target)
        return img, target
    
class DatasetLoader():
    def __init__(self, label_path, img_path, std, mean, img_size, batch_size, transform = True) -> None:
        self.img_path = img_path
        self.label_path = label_path
        self.std = std
        self.mean = mean
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transform
    #Cần thêm cái này do các target không đồng nhất về kích thước
    def collate_fn(self, batch):
        images, targets = zip(*batch)
        return torch.stack(images), list(targets)
    def dataset_loader(self, type):
        if type == "train":
            train_dataset = ObjectRecognitionDataset(
                label_root=self.label_path,
                image_root=self.img_path,
                std=self.std,
                mean=self.mean,
                img_size=self.img_size,
                data_type=type,
                transform = self.transform
            )
            # print("Total train image: {0}, train mask: {1}".format(len(train_dataset), len(train_dataset)))
            loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,          # START HERE
                pin_memory=True,
                persistent_workers=False, #Chỉnh cái này thành False để tránh hết Ram
                prefetch_factor=2,
                collate_fn=self.collate_fn
            )
        else:
            test_dataset = ObjectRecognitionDataset(
                label_root=self.label_path,
                image_root=self.img_path,
                std=self.std,
                mean=self.mean,
                img_size=self.img_size,
                data_type=type,
                transform = self.transform
            )
            # print("Total test image: {0}, train mask: {1}".format(len(test_dataset), len(test_dataset)))
            loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,          # START HERE
                pin_memory=False,
                persistent_workers=False, #Chỉnh cái này thành False để tránh hết Ram,
                collate_fn=self.collate_fn
            )
        return loader