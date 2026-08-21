import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
class Model(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.num_classes = num_classes + 1
        self.model_type = model_type
        self.model = self.build_model()
    def build_model(self):
        match self.model_type:
            case 1:
                model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
                
                in_features = model.roi_heads.box_predictor.cls_score.in_features

                #cls_score là số class
                #bbox_pred là số lượng tọa độ, vì có 4 nên class * 4
                model.roi_heads.box_predictor = FastRCNNPredictor(
                    in_features,
                    self.num_classes
                )
                print("Training Faster R-CNN using Resnet50 backbone")
                return model
    def forward(self, x):
        return self.model(x)