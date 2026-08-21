import torch
class MetricCal():
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.reset()
    def reset(self):
        self.total_cls_loss = 0.0
        self.total_bce_loss = 0.0
        self.total_dice_loss = 0.0
        self.total_overall_loss = 0.0
        self.correct = 0
        self.total = 0
        self.total_bce_dice = 0

        self.tp_per_class = torch.zeros(self.num_classes)
        self.fp_per_class = torch.zeros(self.num_classes)
        self.fn_per_class = torch.zeros(self.num_classes)

    def update_test(self, loss, outputs, targets, type="soft"):
        #Dùng để tính classification loss
        batch_size = targets.size(0)
        self.total_cls_loss += loss.item() * batch_size
        

        
        if type == "soft":
            pred_class = outputs.argmax(dim=1)
            true_class = targets.argmax(dim=1)
        else:
            _, pred_class = outputs.max(1)
            true_class = targets
        self.correct += (pred_class == true_class).sum().item()
        self.total += batch_size

        
        pred = pred_class.detach().cpu()
        true = true_class.detach().cpu()
        
        # Confusion matrix update
        cm = torch.bincount(
            self.num_classes * true + pred,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.tp_per_class += cm.diag().float()
        self.fp_per_class += (cm.sum(dim=0) - cm.diag()).float()
        self.fn_per_class += (cm.sum(dim=1) - cm.diag()).float()

    def update_train(self, cls_loss, bce_loss, dice_loss, outputs, targets, has_masks, type="soft"):
        #Dùng để tính classification loss
        batch_size = targets.size(0)
        self.total_cls_loss += cls_loss.item() * batch_size
        self.total += batch_size
        
        #Dùng để tính bce loss và dice loss
        valid_count = has_masks.sum().item()
        self.total_bce_loss  += bce_loss.item() * valid_count
        self.total_dice_loss += dice_loss.item() * valid_count
        self.total_bce_dice += valid_count

        # self.total_overall_loss += overall_loss.item()

        
        if type == "soft":
            pred_class = outputs.argmax(dim=1)
            true_class = targets.argmax(dim=1)
        else:
            _, pred_class = outputs.max(1)
            true_class = targets
        self.correct += (pred_class == true_class).sum().item()

        
        pred = pred_class.detach().cpu()
        true = true_class.detach().cpu()
        
        # Confusion matrix update
        cm = torch.bincount(
            self.num_classes * true + pred,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.tp_per_class += cm.diag().float()
        self.fp_per_class += (cm.sum(dim=0) - cm.diag()).float()
        self.fn_per_class += (cm.sum(dim=1) - cm.diag()).float()



    @property
    def avg_cls_loss(self):
        """Average loss over all accumulated batches."""
        return self.total_cls_loss / self.total if self.total > 0 else 0.0

    @property
    def avg_bce_loss(self):
        return self.total_bce_loss / self.total_bce_dice if self.total_bce_dice > 0 else 0.0

    @property
    def avg_dice_loss(self):
        return self.total_dice_loss / self.total_bce_dice if self.total_bce_dice > 0 else 0.0

    def overall_loss(self, alpha, beta, gamma):
        return (
            alpha * self.avg_cls_loss
            + beta * self.avg_bce_loss
            + gamma * self.avg_dice_loss
        )

    @property
    def avg_accuracy(self):
        """Accuracy (%) over all accumulated batches."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def precision(self):
        """Per-class precision"""
        denom = self.tp_per_class + self.fp_per_class
        prec = torch.where(denom > 0, self.tp_per_class / denom, torch.zeros_like(denom))
        return prec

    @property
    def recall(self):
        """Per-class recall"""
        denom = self.tp_per_class + self.fn_per_class
        rec = torch.where(denom > 0, self.tp_per_class / denom, torch.zeros_like(denom))
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