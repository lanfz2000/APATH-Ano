import argparse
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import time
from sklearn import metrics
import numpy as np
from dataset.alb_dataset import Tumor_dataset, Tumor_dataset_val, get_loader
import pandas as pd
import random
from open_clip import create_model_from_pretrained, get_tokenizer

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_files(data_root):
    new_file = []
    img_names = os.listdir(data_root+'images')
    for img_name in img_names:
        image_root = data_root+'images/'+img_name
        label_root = data_root+'labels/'+img_name
        label_img = np.array(Image.open(label_root))
        if label_img.max() > 0:
            label = 1
        else:
            label = 0
        new_sample = {'img': image_root, 'label': label}
        new_file.append(new_sample)
    return new_file

def get_arguments():
    parser = argparse.ArgumentParser(
        description="xxxx Pytorch implementation")
    parser.add_argument("--num_class", type=int, default=2, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int)
    parser.add_argument("--batch_size", type=int, default=256, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    return parser.parse_args()


if __name__ == "__main__":
    seed_torch(42)
    args = get_arguments()
    torch.cuda.set_device(args.gpu[0])

    # dataset
    train_data_root = '/home/ubuntu/data/X/datasets/RINGS/train-100-patch/'
    val_data_root = '/home/ubuntu/data/X/datasets/RINGS/val-patch/'
    test_data_root = '/home/ubuntu/data/X/datasets/RINGS/test-patch/'
    train_files = get_files(train_data_root)
    val_files = get_files(val_data_root)
    test_files = get_files(test_data_root)
    val_files, test_files = val_files+test_files, val_files+test_files
    
    np.random.shuffle(train_files)
    print(len(train_files))
    train_dataset = Tumor_dataset_val(args, files=train_files)
    test_dataset = Tumor_dataset_val(args, files=test_files)
    train_loader = get_loader(args, train_dataset)
    test_loader = get_loader(args, test_dataset)
    # get plip model   
    model = CLIPModel.from_pretrained("/home/ubuntu/data/X/codes/CLIP-main/plip")
    processor = CLIPProcessor.from_pretrained("/home/ubuntu/data/X/codes/CLIP-main/plip")

    model = model.cuda()
    model.eval()

    t1 = time.time()

    text_prompt = ["An H&E image of benign tissue", "An H&E image of malignant tissue", \
                   'An H&E image of normal epithelium']
    inputs = processor(text=text_prompt, return_tensors="pt", padding=True)
    
    pred, gt = np.zeros((args.num_class,)), np.zeros((args.num_class,))
    pred_all, gt_all = torch.zeros((1, )), torch.zeros((1, ))
    names = []
    with torch.no_grad():
        for counter, sample in enumerate(train_loader):
            x_batch = sample['img'].cuda()
            y_batch = sample['cls_label'].cuda()
            batch_names = sample['img_name']

            inputs['pixel_values'] = x_batch

            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            outputs = model.forward(**inputs)

            # this is the image-text similarity score
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            hard_probs = torch.argmax(probs, dim=1)

            # high-confidence index
            index1 = torch.where(hard_probs!=2)[0]
            probs = probs[index1]
            y_batch = y_batch[index1]
            batch_names = [batch_names[i] for i in index1.cpu().numpy()]

            logits_hard = torch.argmax(probs, dim=1)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
            gt_all = torch.cat((gt_all, y_batch.cpu()), dim=0)
            names += batch_names

            for i in range(logits_hard.shape[0]):
                gt[y_batch[i].item()] += 1
                if logits_hard[i] == y_batch[i]:
                    pred[logits_hard[i].item()] += 1
            break

    pred_all, gt_all = pred_all[1:], gt_all[1:]
    y_true, y_pred = gt_all.numpy().astype(np.uint8), pred_all.numpy().astype(np.uint8)
    
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    auc = metrics.roc_auc_score(y_true, y_pred)
    print(len(y_true), len(names), pred, gt)
    print(f"Test Accuracy: {test_accuracy.item()}, f1:{f1}, precision:{p}, recall:{r}, auc:{auc}")

    # write pandas
    data_df = pd.DataFrame()
    data_df['image_path'] = names
    data_df['pseudo_label'] = y_pred
    data_df['true_label'] = y_true
    data_df.to_csv(f'pseudo_data/ODP_labels.csv', index=False)
    t2 = time.time()
