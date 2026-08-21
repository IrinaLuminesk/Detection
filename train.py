#Dùng để phân biệt AI và Real sử dụng baseline

import argparse
import os
import random
from typing import Sequence
import numpy as np
from tqdm import tqdm


from utils.MetricCalV3 import MetricCalV3
from learning_rate_helper.learning_rate import PiecewiseScheduler
from model_builder.baseline import Model
from dataset_helper.DatasetLoader import DatasetLoader
from utils.Utilities import Get_Max_Acc, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric3, YAML_Reader, get_mean_std

import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="A simple argparse example")
    
    # Add arguments
    parser.add_argument(
    "--cfg",
    type=str,
    default="config/Detection_config.yaml",
    help="Config file used to train the model (default: config/default_config.yaml)"
    )
    args = parser.parse_args()
    config = YAML_Reader(args.cfg)
    return config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Không cần dùn
def train(epoch: int, end_epoch: int, model, loader, optimizer, device, num_classes):
    model.train()
    # metrics = MetricCalV2(num_classes=num_classes, device=device)
    for inputs, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):

        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        #Ouput trong trường hợp này là dict_keys(['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'])
        outputs = model(inputs, targets)
        loss = sum(outputs.values()) #outputs là dict
        
        loss.backward()
        optimizer.step()

        # metrics.update_test(loss=loss,
        #                     outputs=outputs,
        #                     targets= targets,
        #                      type="soft" if batchWiseAug != None else "hard")
    # return metrics

def validate(epoch, end_epoch, model, loader, device, num_classes):
    model.eval()
    # metrics = MetricCalV2(num_classes=num_classes, device=device)
    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # metrics.update_test(loss=loss, outputs=outputs, targets=targets, type="hard")
    # return metrics


def main():
    config = parse_args()


    #Data parameters
    root_path = config["DATASET"]["ROOT_FOLDER"]
    train_image_path = config["DATASET"]["TRAIN_IMAGE_FOLDER"]
    train_label_path = config["DATASET"]["TRAIN_LABEL_FOLDER"]
    test_image_path = config["DATASET"]["TEST_IMAGE_FOLDER"]
    test_label_path = config["DATASET"]["TEST_LABEL_FOLDER"]
    # CLASSES = sorted([i for i in os.listdir(root_path)])
    mean: Sequence[float] = config["TRAIN"]["DATA"]["MEAN"]
    std: Sequence[float] = config["TRAIN"]["DATA"]["STD"]
    batch_size = config["TRAIN"]["DATA"]["BATCH_SIZE"]
    

    #Training parameters
    img_size = config["TRAIN"]["DATA"]["IMAGE_SIZE"]
    enabled_transform = config["TRAIN"]["TRANSFORM"]
    begin_epoch = config["TRAIN"]["TRAIN_PARA"]["BEGIN_EPOCH"] 
    end_epoch = config["TRAIN"]["TRAIN_PARA"]["END_EPOCH"]
    resume = config["TRAIN"]["TRAIN_PARA"]["RESUME"]
    early_stopping = int(config["TRAIN"]["TRAIN_PARA"]["EARLY_STOPPING"])
    patience = config["TRAIN"]["TRAIN_PARA"]["PATIENCE"]
    epochs_no_improve = 0
    model_type = int(config["TRAIN"]["TRAIN_PARA"]["MODEL_TYPE"])



    #Learning_rate
    Learning_rate_para = config["TRAIN"]["LEARNING_RATE"]["PieceWise"]

    #Optional
    save_checkpoint = config["TRAIN"]["OPTIONAL"]["SAVE_CHECKPOINT"]
    save_best = config["TRAIN"]["OPTIONAL"]["SAVE_BEST"]
    save_metrics = config["TRAIN"]["OPTIONAL"]["SAVE_METRICS"]
    checkpoint_path = config["TRAIN"]["OPTIONAL"]["CHECKPOINT_PATH"]
    best_path = config["TRAIN"]["OPTIONAL"]["BEST_PATH"]
    metrics_path = config["TRAIN"]["OPTIONAL"]["METRICS_PATH"]
    
    set_seed()
    
    # if mean is None or std is None:
    #     print("Calculating mean and std")
    #     mean, std = get_mean_std(train_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = DatasetLoader(
        img_path=train_image_path,
        label_path=train_label_path,
        std=std,
        mean=mean,
        img_size=img_size,
        batch_size=batch_size,
        transform = enabled_transform)
    test_data = DatasetLoader(
        img_path=test_image_path,
        label_path=test_label_path,
        std=std,
        mean=mean,
        img_size=img_size,
        batch_size=batch_size)

    training_loader = train_data.dataset_loader("train")
    testing_loader = test_data.dataset_loader("test")

    CLASSES = list(train_data.unique_labels | test_data.unique_labels)
    model = Model(
        num_classes=len(CLASSES), 
        model_type=model_type
    ).to(device)


    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate_para["MAX_LR"], weight_decay=1e-2)

    lr_schedule = PiecewiseScheduler(
        start_lr=Learning_rate_para["START_LR"],
        max_lr=Learning_rate_para["MAX_LR"],
        min_lr=Learning_rate_para["MIN_LR"],
        rampup_epochs=Learning_rate_para["RAMPUP_EPOCHS"],
        sustain_epochs=Learning_rate_para["SUSTAIN_EPOCHS"],
        exp_decay=Learning_rate_para["EXP_DECAY"]
    )
    print("Training using PiecewiseScheduler")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    best_acc = 0

    # if resume == True:
    #     begin_epoch = Loading_Checkpoint(path=checkpoint_path,
    #                                      model=model,
    #                                      optimizer=optimizer,
    #                                      scheduler=scheduler,
    #                                      device=device)
    #     best_acc = Get_Max_Acc(metrics_path)
    for epoch in range(begin_epoch, end_epoch):
        train(epoch, 
                                end_epoch, 
                                model=model, 
                                loader=training_loader,  
                                optimizer=optimizer, 
                                device=device,
                                num_classes=len(CLASSES))
        scheduler.step()
        print()
        validate(epoch, end_epoch, model, loader=testing_loader, device=device, num_classes=len(CLASSES))
        print()

        # if save_checkpoint == True:
        #     Saving_Checkpoint(epoch=epoch, 
        #                     model=model, 
        #                     optimizer=optimizer, 
        #                     scheduler=scheduler,
        #                     last_epoch=epoch, 
        #                     path=checkpoint_path)

        # print("Epoch [{0}/{1}]: Training loss: {2}, Training Acc: {3}%".
        #     format(epoch, end_epoch, train_loss, round(train_acc * 100.0, 2)))
        # print("Epoch [{0}/{1}]: Validation loss: {2}, Validation Acc: {3}%".
        #     format(epoch, end_epoch, val_loss, round(val_acc * 100.0, 2)))
        # if val_acc > best_acc:
        #     if save_best == True:
        #         print("Validation accuracy increase from {0}% to {1}% at epoch {2}. Saving best result".
        #             format(round(best_acc * 100.0, 2), round(val_acc * 100.0, 2),  epoch))
        #         Saving_Best(model, best_path)
        #     else:
        #         print("Validation accuracy increase from {0}% to {1}% at epoch {2}".
        #             format(round(best_acc * 100.0, 2), round(val_acc * 100.0, 2),  epoch))
        #     best_acc = val_acc
        #     epochs_no_improve = 0  # reset patience
        # else:
        #     epochs_no_improve += 1
        # if save_metrics:
        #     metric_row = {
        #         "train_cls_loss": train_metrics.avg_cls_loss,
        #         "train_focal_loss": train_metrics.avg_focal_loss,
        #         "train_tversky_loss": train_metrics.avg_tversky_loss,
        #         # "train_tv_loss": train_metrics.avg_tv_loss,
        #         "train_overall_loss": train_loss,
        #         "train_acc": train_acc,
        #         "train_precision": train_metrics.precision_macro,
        #         "train_recall": train_metrics.recall_macro,
        #         "train_f1": train_metrics.f1_macro, 
        #         "train_MCC": train_metrics.MCC,
        #         "train_FMI": train_metrics.FMI,
        #         "train_Cohen_Kappa": train_metrics.cohen_kappa,
        #         "val_loss": val_loss,
        #         "val_acc": val_acc,
        #         "val_precision": val_metrics.precision_macro,
        #         "val_recall": val_metrics.recall_macro,
        #         "val_f1": val_metrics.f1_macro, 
        #         "val_MCC": val_metrics.MCC,
        #         "val_FMI": val_metrics.FMI,
        #         "val_Cohen_Kappa": val_metrics.cohen_kappa,
        #     }
        #     Saving_Metric3(epoch=epoch, metric_row=metric_row, path=metrics_path)
        # if epochs_no_improve >= patience and early_stopping == True:
        #     print("Early stopping triggered at epoch {0}".format(epoch))
        #     break
        print()

    
if __name__ == '__main__':
    main()
