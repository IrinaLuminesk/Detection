from concurrent.futures import ThreadPoolExecutor
import cv2
import torch
from tqdm import tqdm
import yaml
import numpy as np
import os
import pandas as pd
import pickle

import torch.nn as nn

def Create_Folder(path):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

def YAML_Reader(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def read_img(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError("Image not found")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img.mean(axis=(0, 1)), img.std(axis=(0, 1)), 1
    except:
        return np.zeros(3), np.zeros(3), 0

def get_mean_std(path, max_workers=4):
    image_paths = []
    for class_folder in os.listdir(path):
        class_path = os.path.join(path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                image_paths.append(os.path.join(class_path, img_file))

    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mean, std, count in tqdm(executor.map(read_img, image_paths), total=len(image_paths), desc="Computing mean/std"):
            total_mean += mean
            total_std += std
            total_count += count

    if total_count == 0:
        raise ValueError("No valid images found.")

    mean = total_mean / total_count
    std = total_std / total_count
    print("Mean: {0}, STD: {1}".format(mean, std))
    return mean.tolist(), std.tolist()

def Saving_Checkpoint(epoch, model, optimizer, scheduler, last_epoch, path):
    Create_Folder(path=path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        "last_epoch": last_epoch
    }, path)

def Saving_Best(model, path):
    Create_Folder(path=path)
    torch.save(model.state_dict(), path)

def Saving_Metric(epoch, train_acc, train_loss, top1_val_acc, top5_val_acc, val_loss, path):
    Create_Folder(path=path)
    if os.path.exists(path):
        metrics_df = pd.read_csv(path)
    else:
        metrics_df = pd.DataFrame({
            'epoch': pd.Series(dtype='int'),
            'train_loss': pd.Series(dtype='float'),
            'train_acc': pd.Series(dtype='float'),
            'val_loss': pd.Series(dtype='float'),
            'top1_val_acc': pd.Series(dtype='float'),
            'top5_val_acc': pd.Series(dtype='float'),
            'lr': pd.Series(dtype='float')
        })
    new_row = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'top1_val_acc': top1_val_acc,
        'top5_val_acc': top5_val_acc,
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
    metrics_df.to_csv(path, index=False)

def Saving_Metric2(epoch, 
                   train_acc,
                   train_precision,
                   train_recall,
                   train_f1,
                   train_MCC,
                   train_FMI,
                   train_Cohen_Kappa, 
                   train_cls_loss,
                   train_focal_loss,
                   train_tversky_loss,
                   train_tv_loss,
                   train_overall_loss, 
                   val_acc,
                   val_precision,
                   val_recall,
                   val_f1,
                   val_MCC,
                   val_FMI,
                   val_Cohen_Kappa, 
                   val_loss,
                   path):
    Create_Folder(path=path)
    if os.path.exists(path):
        metrics_df = pd.read_csv(path)
    else:
        metrics_df = pd.DataFrame({
            'epoch': pd.Series(dtype='int'),
            'train_cls_loss': pd.Series(dtype='float'),
            'train_focal_loss': pd.Series(dtype='float'),
            'train_tversky_loss': pd.Series(dtype='float'),
            'train_tv_loss': pd.Series(dtype='float'),
            'train_overall_loss': pd.Series(dtype='float'),
            'train_acc': pd.Series(dtype='float'),
            'train_precision': pd.Series(dtype='float'),
            'train_recall': pd.Series(dtype='float'),
            'train_f1': pd.Series(dtype='float'),
            'train_MCC': pd.Series(dtype='float'),
            'train_FMI': pd.Series(dtype='float'),
            'train_Cohen_Kappa': pd.Series(dtype='float'),
            'val_loss': pd.Series(dtype='float'),
            'val_acc': pd.Series(dtype='float'),
            'val_precision': pd.Series(dtype='float'),
            'val_recall': pd.Series(dtype='float'),
            'val_f1': pd.Series(dtype='float'),
            'val_MCC': pd.Series(dtype='float'),
            'val_FMI': pd.Series(dtype='float'),
            'val_Cohen_Kappa': pd.Series(dtype='float'),
            'lr': pd.Series(dtype='float')
        })
    new_row = {
        'epoch': epoch,
        'train_cls_loss': train_cls_loss,
        'train_focal_loss': train_focal_loss,
        'train_tversky_loss': train_tversky_loss,
        'train_tv_loss': train_tv_loss,
        'train_overall_loss': train_overall_loss,
        'train_acc': train_acc,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'train_MCC': train_MCC,
        'train_FMI': train_FMI,
        'train_Cohen_Kappa': train_Cohen_Kappa,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'val_MCC': val_MCC,
        'val_FMI': val_FMI,
        'val_Cohen_Kappa': val_Cohen_Kappa,
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
    metrics_df.to_csv(path, index=False)

# metric_row là một dict, ** sẽ unpack nó thành key: value
def Saving_Metric3(epoch, metric_row, path):
    Create_Folder(path=path)
    
    row = {'epoch': epoch, **metric_row}

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(path, index=False)


def Loading_Checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('last_epoch', 0) + 1
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch

def Get_Max_Acc(path):
    df = pd.read_csv(path)

    best_acc = df['val_acc'].max()

    return best_acc 

def get_num_workers(transform_heavy=True):
    # 1. Detect CPU cores
    cpu_cores = os.cpu_count() or 1

    # 2. Detect GPU count
    gpu_count = torch.cuda.device_count()

    # 3. Select scaling factor
    # Heavy transforms: ElasticTransform, RandomAffine, Perspective, etc.
    if transform_heavy:
        scale = 1.5
    else:
        scale = 1.0

    # 4. Handle GPU or CPU-only case
    if gpu_count > 0:
        # Workers per GPU
        workers = int((cpu_cores / gpu_count) * scale)
    else:
        # CPU-only system, use more workers since CPU handles everything
        workers = int(cpu_cores * 0.75)

    # 5. Clamp values
    workers = max(2, workers)        # never less than 2
    workers = min(workers, cpu_cores - 1)  # never use ALL CPU cores

    return workers

def replace_relu_helper(model):
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, name, nn.SiLU(inplace=True))
        else:
            replace_relu_helper(child)

def serialize_object(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def unserialize_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)