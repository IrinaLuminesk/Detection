import torch
from torchvision.ops import box_iou
class MetricCal():
    def __init__(self, num_classes,device, training_type="train", iou_threshold=0.5, confidence_threshold=0.25) -> None:
        self.num_classes = num_classes
        self.iou_threshold=iou_threshold
        self.confidence_threshold=confidence_threshold
        self.device = device
        self.training_type = training_type.lower()
        self.reset()
    def reset(self):
        self.num_images = 0
        if self.training_type == "train":
            #Ouput trong trường hợp này là dict_keys(['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'])
            self.total_cls_loss = torch.zeros(1, device=self.device)
            self.loss_classifier = torch.zeros(1, device=self.device)
            self.loss_box_reg = torch.zeros(1, device=self.device)
            self.loss_objectness = torch.zeros(1, device=self.device)
            self.loss_rpn_box_reg = torch.zeros(1, device=self.device)
        else:        
            self.tp_per_class = torch.zeros(
                self.num_classes + 1, device=self.device
            )
            self.fp_per_class = torch.zeros(
                self.num_classes + 1, device=self.device
            )
            self.fn_per_class = torch.zeros(
                self.num_classes + 1, device=self.device
            )
    @torch.no_grad()
    def update_test(self, outputs, targets):
        if len(outputs) != len(targets):
            raise ValueError(
                "Number of prediction images must equal number of target images.")
        for pred, target in zip(outputs, targets):
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            scores = pred["scores"]

            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            self.num_images += 1

            # Apply the confidence threshold.
            keep = scores >= self.confidence_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            scores = scores[keep]

            for cls in range(self.num_classes):
                
                pred_idx = torch.where(pred_labels == cls)[0]
                gt_idx = torch.where(gt_labels == cls)[0]

                p_boxes = pred_boxes[pred_idx]
                p_scores = scores[pred_idx]
                g_boxes = gt_boxes[gt_idx]

                num_pred = len(p_boxes)
                num_gt = len(g_boxes)

                if num_pred == 0:
                    # Every ground-truth object was missed.
                    self.fn_per_class[cls] += num_gt
                    continue

                if num_gt == 0:
                    # Every prediction is a false positive.
                    self.fp_per_class[cls] += num_pred
                    continue

                # IoU matrix: [number of predictions, number of GTs]
                ious = box_iou(p_boxes, g_boxes)

                # Process predictions from highest to lowest confidence.
                order = torch.argsort(p_scores, descending=True)
                matched_gt = torch.zeros(
                    num_gt, dtype=torch.bool, device=self.device
                )

                tp = 0
                fp = 0

                for pred_i in order:
                    pred_ious = ious[pred_i].clone()

                    # A GT box can only be matched once.
                    pred_ious[matched_gt] = -1

                    best_iou, best_gt = pred_ious.max(dim=0)

                    if best_iou >= self.iou_threshold:
                        matched_gt[best_gt] = True
                        tp += 1
                    else:
                        fp += 1

                fn = num_gt - int(matched_gt.sum().item())

                self.tp_per_class[cls] += tp
                self.fp_per_class[cls] += fp
                self.fn_per_class[cls] += fn
            
    @torch.no_grad()
    def update_train(self, cls_loss, saliency_loss1, saliency_loss2, outputs, targets, has_masks, type="soft"):
        a = 1



    @property
    def precision(self):
        denom = self.tp_per_class + self.fp_per_class
        return torch.where(
            denom > 0,
            self.tp_per_class / denom.clamp(min=1),
            torch.zeros_like(denom),
        )

    @property
    def recall(self):
        denom = self.tp_per_class + self.fn_per_class
        return torch.where(
            denom > 0,
            self.tp_per_class / denom.clamp(min=1),
            torch.zeros_like(denom),
        )

    @property
    def f1_score(self):
        p = self.precision
        r = self.recall
        denom = p + r

        return torch.where(
            denom > 0,
            2 * p * r / denom.clamp(min=1e-12),
            torch.zeros_like(denom),
        )

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
    def precision_micro(self):
        tp = self.tp_per_class.sum()
        fp = self.fp_per_class.sum()
        return (tp / (tp + fp).clamp(min=1)).item()

    @property
    def recall_micro(self):
        tp = self.tp_per_class.sum()
        fn = self.fn_per_class.sum()
        return (tp / (tp + fn).clamp(min=1)).item()

    @property
    def f1_micro(self):
        p = self.precision_micro
        r = self.recall_micro
        return 2 * p * r / (p + r) if p + r > 0 else 0.0