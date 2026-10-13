import argparse
import random
from typing import Sequence
import numpy as np
from tqdm import tqdm


from utils.MetricCal import MetricCal
from learning_rate_helper.learning_rate import PiecewiseScheduler
from model_builder.baseline import Model
from dataset_helper.DatasetLoaderV2 import DatasetLoader
from utils.Utilities import Get_Max_Monitor_Metric, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric3, YAML_Reader, get_mean_std

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


def train(epoch: int, end_epoch: int, model, loader, optimizer, device, num_classes):
    model.train()
    metrics = MetricCal(num_classes=num_classes, device=device, training_type="train", confidence_threshold=0.8)
    for inputs, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):

        inputs = [image.to(device, non_blocking=True) for image in inputs]
        targets = [
            {
                key: value.to(device, non_blocking=True)
                for key, value in target.items()
            } for target in targets
        ]
        optimizer.zero_grad()
        #Ouput trong trường hợp này là dict_keys(['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'])
        outputs = model(inputs, targets)
        loss = sum(outputs.values()) #outputs là dict
        
        loss.backward()
        optimizer.step()

        metrics.update_train(loss=loss,batch_size=len(targets))
    return metrics

def validate(epoch, end_epoch, model, loader, device, num_classes):
    model.eval()
    metrics = MetricCal(num_classes=num_classes, device=device, training_type="test", confidence_threshold=0.8)
    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs = [image.to(device, non_blocking=True) for image in inputs]
            targets = [
                {
                    key: value.to(device, non_blocking=True)
                    for key, value in target.items()
                } for target in targets
            ]
            #output của faster R-CNN khi validate là boxes, labels và scores
            outputs = model(inputs)
            metrics.update_test(outputs=outputs, targets=targets)
    return metrics

def main():
    config = parse_args()

    #Data parameters
    train_image_path = config["DATASET"]["TRAIN_IMAGE_FOLDER"]
    train_label_path = config["DATASET"]["TRAIN_LABEL_FOLDER"]
    test_image_path = config["DATASET"]["TEST_IMAGE_FOLDER"]
    test_label_path = config["DATASET"]["TEST_LABEL_FOLDER"]
    batch_size = config["TRAIN"]["DATA"]["BATCH_SIZE"]
    num_classes = config["DATASET"]["NUM_CLASSES"]
    original_format = config["DATASET"]["ORIGINAL_FORMAT"]
    convert_format = config["DATASET"]["CONVERT_FORMAT"]

    #Training parameters
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = DatasetLoader(
        label_root=train_image_path,
        image_root=train_label_path,
        batch_size=batch_size,
        original_format=original_format,
        convert_format=convert_format
    )
    test_data = DatasetLoader(
        label_root=test_image_path,
        image_root=test_label_path,
        batch_size=batch_size,
        original_format=original_format,
        convert_format=convert_format
    )

    training_loader = train_data.dataset_loader("train")
    testing_loader = test_data.dataset_loader("test")

    model = Model(
        num_classes=num_classes, 
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
    best_map_50_95  = 0

    if resume == True:
        begin_epoch = Loading_Checkpoint(path=checkpoint_path,
                                         model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         device=device)
        best_map_50_95 = Get_Max_Monitor_Metric(metrics_path, "train_map_50_95")
    for epoch in range(begin_epoch, end_epoch):
        train_metrics = train(epoch, 
            end_epoch, 
            model=model, 
            loader=training_loader,  
            optimizer=optimizer, 
            device=device,
            num_classes=num_classes)
        train_loss = train_metrics.avg_total_loss
        scheduler.step()
        print()
        val_metrics = validate(epoch, end_epoch, model, loader=testing_loader, device=device, num_classes=num_classes)
        val_map_50_95 = val_metrics.map_50_95
        print()

        if save_checkpoint == True:
            Saving_Checkpoint(epoch=epoch, 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler,
                            last_epoch=epoch, 
                            path=checkpoint_path)

        print("Epoch [{0}/{1}]: Training loss: {2}".
            format(epoch, end_epoch, train_loss))
        print("Epoch [{0}/{1}]: Validation mAP@0.5:0.95: {2}%".
            format(epoch, end_epoch, round(val_map_50_95 * 100.0, 2)))
        if val_map_50_95 > best_map_50_95:
            if save_best == True:
                print("Validation mAP@0.5:0.95 increase from {0}% to {1}% at epoch {2}. Saving best result".
                    format(round(best_map_50_95 * 100.0, 2), round(val_map_50_95 * 100.0, 2),  epoch))
                Saving_Best(model, best_path)
            else:
                print("Validation mAP@0.5:0.95 increase from {0}% to {1}% at epoch {2}".
                    format(round(best_map_50_95 * 100.0, 2), round(val_map_50_95 * 100.0, 2),  epoch))
            best_map_50_95 = val_map_50_95
            epochs_no_improve = 0  # reset patience
        else:
            epochs_no_improve += 1
        if save_metrics:
            metric_row = {
                "train_loss_classifier": train_metrics.avg_loss_classifier,
                "train_loss_box_reg": train_metrics.avg_loss_box_reg,
                "train_loss_objectness": train_metrics.avg_loss_objectness,
                "train_loss_rpn_box_reg": train_metrics.avg_loss_rpn_box_reg,
                "train_loss": train_metrics.avg_total_loss,
                "val_precision": val_metrics.precision_macro,
                "val_recall": val_metrics.recall_macro,
                "val_f1": val_metrics.f1_macro,
                "val_map_50": val_metrics.map_50,
                "val_map_50_95": val_metrics.map_50_95 
            }
            Saving_Metric3(epoch=epoch, metric_row=metric_row, path=metrics_path)
        if epochs_no_improve >= patience and early_stopping == True:
            print("Early stopping triggered at epoch {0}".format(epoch))
            break
        print()

if __name__ == '__main__':
    main()