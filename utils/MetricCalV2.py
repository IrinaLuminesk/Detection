import torch
class MetricCalV2():
    def __init__(self, num_classes, device) -> None:
        self.num_classes = num_classes
        self.device = device
        self.reset()
    def reset(self):
        self.total_cls_loss = torch.zeros(1, device=self.device)
        self.total_bce_loss = torch.zeros(1, device=self.device)
        self.total_dice_loss = torch.zeros(1, device=self.device)
        self.total_tv_loss = torch.zeros(1, device=self.device)
        self.total_overall_loss = torch.zeros(1, device=self.device)
        
        self.correct = torch.zeros(1, device=self.device)
        self.total = torch.zeros(1, device=self.device)
        self.total_bce_dice = torch.zeros(1, device=self.device)

        self.cm = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device
        )
        self.tp_per_class = torch.zeros(self.num_classes, device=self.device)
        self.fp_per_class = torch.zeros(self.num_classes, device=self.device)
        self.fn_per_class = torch.zeros(self.num_classes, device=self.device)

    @torch.no_grad()
    def update_test(self, loss, outputs, targets, type="soft"):
        #Dùng để tính classification loss
        batch_size = targets.size(0)
        
        self.total_cls_loss += loss.detach() * batch_size
        self.total += batch_size

        
        if type == "soft":
            pred_class = outputs.argmax(dim=1)
            true_class = targets.argmax(dim=1)
        else:
            _, pred_class = outputs.max(1)
            true_class = targets
        
        self.correct += (pred_class == true_class).sum()

        
        # pred = pred_class.detach().cpu()
        # true = true_class.detach().cpu()
        
        # Confusion matrix update
        cm = torch.bincount(
            self.num_classes * true_class + pred_class,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.cm += cm

        self.tp_per_class += cm.diag()
        self.fp_per_class += cm.sum(dim=0) - cm.diag()
        self.fn_per_class += cm.sum(dim=1) - cm.diag()

    @torch.no_grad()
    def update_train(self, cls_loss, bce_loss, dice_loss, outputs, targets, has_masks, type="soft"):
        #Dùng để tính classification loss
        batch_size = targets.size(0)


        self.total_cls_loss += cls_loss.detach() * batch_size
        self.total += batch_size
        
        #Dùng để tính bce loss và dice loss
        valid_count = has_masks.sum()
        self.total_bce_loss  += bce_loss.detach() * valid_count
        self.total_dice_loss += dice_loss.detach() * valid_count
        self.total_bce_dice += valid_count

        # self.total_overall_loss += overall_loss.item()

        
        if type == "soft":
            pred_class = outputs.argmax(dim=1)
            true_class = targets.argmax(dim=1)
        else:
            _, pred_class = outputs.max(1)
            true_class = targets
        
        self.correct += (pred_class == true_class).sum()

        
        # pred = pred_class.detach().cpu()
        # true = true_class.detach().cpu()
        
        # Confusion matrix update
        cm = torch.bincount(
            self.num_classes * true_class + pred_class,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.cm += cm

        self.tp_per_class += cm.diag()
        self.fp_per_class += cm.sum(dim=0) - cm.diag()
        self.fn_per_class += cm.sum(dim=1) - cm.diag()



    @property
    def avg_cls_loss(self):
        """Average loss over all accumulated batches."""
        return (self.total_cls_loss / self.total).item() if self.total > 0 else 0.0

    @property
    def avg_bce_loss(self):
        return (self.total_bce_loss / self.total_bce_dice).item() if self.total_bce_dice > 0 else 0.0

    @property
    def avg_dice_loss(self):
        return (self.total_dice_loss / self.total_bce_dice).item() if self.total_bce_dice > 0 else 0.0

    def overall_loss(self, alpha, beta, gamma):
        return (
            alpha * self.avg_cls_loss
            + beta * self.avg_bce_loss
            + gamma * self.avg_dice_loss
        )

    @property
    def avg_accuracy(self):
        """Accuracy (%) over all accumulated batches."""
        return (self.correct / self.total).item() if self.total > 0 else 0.0

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
    
    # Matthews Correlation Coefficient(MCC)
    @property
    def MCC(self):
        eps = 1e-12
        cm = self.cm.detach().float()

        t_sum = cm.sum(dim=1)      # true counts per class
        p_sum = cm.sum(dim=0)      # predicted counts per class

        n_correct = torch.trace(cm)
        n_samples = cm.sum()

        cov_ytyp = n_correct * n_samples - torch.dot(t_sum, p_sum)

        cov_ypyp = n_samples**2 - torch.dot(p_sum, p_sum)
        cov_ytyt = n_samples**2 - torch.dot(t_sum, t_sum)

        denom = torch.sqrt(cov_ytyt * cov_ypyp).clamp_min(eps)

        return (cov_ytyp / denom).item()

    # Fowlkes–Mallows Index
    @property
    def FMI(self):
        eps = 1e-12

        tp = self.tp_per_class.detach().float()
        fp = self.fp_per_class.detach().float()
        fn = self.fn_per_class.detach().float()

        denom = torch.sqrt((tp + fp) * (tp + fn)).clamp_min(eps)

        fmi_per_class = tp / denom

        return fmi_per_class.mean().item()

    @property
    def cohen_kappa(self):
        eps = 1e-12
        cm = self.cm.float()

        n = cm.sum()

        po = torch.trace(cm) / (n + eps)

        true_sum = cm.sum(dim=1)
        pred_sum = cm.sum(dim=0)

        pe = torch.dot(true_sum, pred_sum) / ((n * n) + eps)

        return ((po - pe) / (1 - pe + eps)).item()