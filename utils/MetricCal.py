import torch
from torchmetrics.detection import MeanAveragePrecision


class MetricCal():
    def __init__(self, num_classes,device, training_type="train", iou_threshold=0.5, confidence_threshold=0.25) -> None:
        self.num_classes = num_classes + 1 #Do bao gồm background
        self.iou_threshold=iou_threshold
        self.confidence_threshold=confidence_threshold
        self.device = device
        self.training_type = training_type.lower()
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[
                0.50, 0.55, 0.60, 0.65, 0.70,
                0.75, 0.80, 0.85, 0.90, 0.95,
            ],
        ).to(device)

        self.map50_metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
                iou_thresholds=[0.50],
        ).to(device)
        self.reset()
    def reset(self):
        if self.training_type == "test":
            self.tp_per_class = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
            self.fp_per_class = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
            self.fn_per_class = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
            self.map50_metric.reset()
            self.map_metric.reset()
        else:
            #Ouput trong trường hợp này là dict_keys(['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'])
            # self.total_loss = torch.zeros(1, device=self.device)
            self.total_loss_classifier = torch.zeros(1, device=self.device)
            self.total_loss_box_reg = torch.zeros(1, device=self.device)
            self.total_loss_objectness = torch.zeros(1, device=self.device)
            self.total_loss_rpn_box_reg = torch.zeros(1, device=self.device)
            self.total = torch.zeros(1, device=self.device)
    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area

    def update_train(self, loss, batch_size):
        self.total_loss_classifier += (
            loss["loss_classifier"].detach() * batch_size
        )
        self.total_loss_box_reg += (
            loss["loss_box_reg"].detach() * batch_size
        )
        self.total_loss_objectness += (
            loss["loss_objectness"].detach() * batch_size
        )
        self.total_loss_rpn_box_reg += (
            loss["loss_rpn_box_reg"].detach() * batch_size
        )
        
        self.total += batch_size

    def update_test(self, outputs, targets):
        self.map_metric.update(outputs, targets)
        self.map50_metric.update(outputs, targets)
        # pred và grt là dict
        for pred, grt in zip(outputs, targets): 
            keep_pred = pred["scores"] >= self.confidence_threshold
            
            pred_scores = pred["scores"][keep_pred]
            pred_labels = pred["labels"][keep_pred]
            pred_boxes = pred["boxes"][keep_pred]

            # Sort scores in descending order
            sorted_idx = torch.argsort(pred_scores, descending=True)

            # Apply the same sorting order to all
            pred_scores = pred_scores[sorted_idx]
            pred_labels = pred_labels[sorted_idx]
            pred_boxes = pred_boxes[sorted_idx]

            grt_labels = grt["labels"]
            grt_boxes = grt["boxes"]
            
            matched_gt = set()
            
            for pred_l, pred_b in zip(pred_labels, pred_boxes):
                best_iou = 0.0
                best_idx = None
                for i, (gt_class, gt_box) in enumerate(zip(grt_labels, grt_boxes)):
                    if i in matched_gt:
                        continue

                    if pred_l != gt_class:
                        continue
                    
                    iou = self.compute_iou(pred_b, gt_box)

                    if iou >= self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_idx is not None:
                    self.tp_per_class[pred_l] += 1
                    matched_gt.add(best_idx)
                else:
                    self.fp_per_class[pred_l] += 1
            for i, (gt_class, gt_box) in enumerate(zip(grt_labels, grt_boxes)):
                if i not in matched_gt:
                    self.fn_per_class[gt_class] += 1

    @property
    def avg_loss_classifier(self):
        return (self.total_loss_classifier / self.total).item() if self.total > 0 else 0.0
    @property
    def avg_loss_box_reg(self):
        return (self.total_loss_box_reg / self.total).item() if self.total > 0 else 0.0
    @property
    def avg_loss_objectness(self):
        return (self.total_loss_objectness / self.total).item() if self.total > 0 else 0.0
    @property
    def avg_loss_rpn_box_reg(self):
        return (self.total_loss_rpn_box_reg / self.total).item() if self.total > 0 else 0.0
    @property
    def avg_total_loss(self):
        return (
            self.avg_loss_classifier + self.avg_loss_box_reg + self.avg_loss_objectness + self.avg_loss_rpn_box_reg
        )
    @property
    def precision(self):
        """Per-class precision"""
        denom = self.tp_per_class[1:] + self.fp_per_class[1:]
        prec = torch.where(denom > 0, self.tp_per_class[1:] / denom, torch.zeros_like(denom))
        return prec
    @property
    def recall(self):
        """Per-class recall"""
        denom = self.tp_per_class[1:] + self.fn_per_class[1:]
        rec = torch.where(denom > 0, self.tp_per_class[1:] / denom, torch.zeros_like(denom))
        return rec
    @property
    def f1_score(self):
        """Per-class F1-score"""
        prec = self.precision
        rec = self.recall
        denom = prec + rec
        f1 = torch.where(denom > 0, 2 * prec * rec / denom, torch.zeros_like(denom))
        return f1
    @property
    def precision_macro(self):
        return self.precision.mean().item()
    @property
    def recall_macro(self):
        return self.recall.mean().item()
    @property
    def f1_macro(self):
        return self.f1_score.mean().item()
    
    @property
    def map_50(self):
        """Mean Average Precision at IoU = 0.50."""
        return self.map50_metric.compute()["map"].item()

    @property
    def map_50_95(self):
        """Mean Average Precision at IoU = 0.50:0.95."""
        return self.map_metric.compute()["map"].item()
# detections = [
#     [10, 20, 50, 60, 0, 0.95],
#     [60, 20, 100, 60, 1, 0.82],
#     [110, 20, 150, 60, 0, 0.45]
# ]

# # [x1, y1, x2, y2, class_id]
# annotations = [
#     [12, 22, 48, 58, 0],
#     [62, 22, 98, 58, 1]
# ]
# from collections import defaultdict

# def compute_metrics(
#     detections,
#     annotations,
#     iou_threshold=0.5,
#     confidence_threshold=0.0
# ):
#     tp = defaultdict(int)
#     fp = defaultdict(int)
#     fn = defaultdict(int)

#     for det, ann in zip(detections, annotations):
#         # Filter predictions by confidence and sort highest first
#         det = [
#             d for d in det
#             if float(d[5]) >= confidence_threshold
#         ]
#         det = sorted(det, key=lambda d: float(d[5]), reverse=True)

#         matched_gt = set()

#         for d in det:
#             pred_box = d[:4]
#             pred_class = int(d[4])

#             best_iou = 0.0
#             best_idx = None

#             for i, a in enumerate(ann):
#                 if i in matched_gt:
#                     continue

#                 gt_box = a[:4]
#                 gt_class = int(a[4])

#                 # Class must match
#                 if pred_class != gt_class:
#                     continue

#                 iou = compute_iou(pred_box, gt_box)

#                 if iou >= iou_threshold and iou > best_iou:
#                     best_iou = iou
#                     best_idx = i

#             if best_idx is not None:
#                 tp[pred_class] += 1
#                 matched_gt.add(best_idx)
#             else:
#                 fp[pred_class] += 1

#         # Unmatched ground-truth objects are false negatives
#         for i, a in enumerate(ann):
#             if i not in matched_gt:
#                 fn[int(a[4])] += 1

#     classes = sorted(set(tp) | set(fp) | set(fn))
#     per_class = {}

#     for cls in classes:
#         precision = (
#             tp[cls] / (tp[cls] + fp[cls])
#             if tp[cls] + fp[cls] > 0 else 0.0
#         )
#         recall = (
#             tp[cls] / (tp[cls] + fn[cls])
#             if tp[cls] + fn[cls] > 0 else 0.0
#         )
#         f1 = (
#             2 * precision * recall / (precision + recall)
#             if precision + recall > 0 else 0.0
#         )

#         per_class[cls] = {
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1,
#             "TP": tp[cls],
#             "FP": fp[cls],
#             "FN": fn[cls],
#         }

#     if per_class:
#         precision_macro = sum(
#             m["precision"] for m in per_class.values()
#         ) / len(per_class)
#         recall_macro = sum(
#             m["recall"] for m in per_class.values()
#         ) / len(per_class)
#         f1_macro = sum(
#             m["f1_score"] for m in per_class.values()
#         ) / len(per_class)
#     else:
#         precision_macro = recall_macro = f1_macro = 0.0

#     return per_class, precision_macro, recall_macro, f1_macro
# How to use it
# python
# per_class, precision_macro, recall_macro, f1_macro = (
#     compute_metrics(
#         detections,
#         annotations,
#         iou_threshold=0.5,
#         confidence_threshold=0.5
#     )
# )