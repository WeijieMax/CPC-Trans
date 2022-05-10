import os
import sys
import json
import pickle
import random
import numpy as np

import torch
from tqdm import tqdm


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def read_split_data(root: str):
    print('start data reading...')
    polyp_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    polyp_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(polyp_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []
    images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in polyp_class:
        cla_path = os.path.join(root, cla)
        images_path.extend([os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported])
        images_label.extend([class_indices[cla]]*len(images_path))
        every_class_num.append(len(images_path))
    print("{} images were found in the dataset.".format(sum(every_class_num)))

    return images_path, images_label


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def train_one_epoch(model, optimizer, data_loader, device, epoch, norm_coff=0.3):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function_aux = torch.nn.CosineEmbeddingLoss()
    total_loss = torch.zeros(1).to(device)
    accu_loss_CLS1 = torch.zeros(1).to(device)  
    accu_loss_CLS2 = torch.zeros(1).to(device)
    aux_loss_CGA = torch.zeros(1).to(device)
    aux_loss_SAM = torch.zeros(1).to(device)
    accu_num1 = torch.zeros(1).to(device)   
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, mininterval=2)
    for step, data in enumerate(data_loader):
        images1, images2, labels = data
        sample_num += images1.shape[0]

        pred1, pred2, pred1_cls, pred2_cls, attn_token11, attn_token22 = model(images1.to(device), images2.to(device))
        
        pred1_classes = torch.max(pred1, dim=1)[1]
        accu_num1 += torch.eq(pred1_classes, labels.to(device)).sum()

        loss_CLS1 = loss_function(pred1, labels.to(device))
        loss_CLS2 = loss_function(pred2, labels.to(device))
        loss_CGA = loss_function_aux(pred1_cls, pred2_cls, torch.ones(1).to(device))
        loss_SAM = loss_function_aux(attn_token11, attn_token22, torch.ones(1).to(device))*norm_coff
        loss = loss_CLS1+loss_CLS2+loss_CGA+loss_SAM

        loss.backward()
        total_loss += loss.detach()
        accu_loss_CLS1 += loss_CLS1.detach()
        accu_loss_CLS2 += loss_CLS2.detach()
        aux_loss_CGA += loss_CGA.detach()
        aux_loss_SAM += loss_SAM.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, loss_CLS1: {:.3f}, loss_CLS2: {:.3f}, loss_CGA: {:.3f}, loss_SAM: {:.3f}, acc: {:.3f}".format(
            epoch,
            total_loss.item() / (step + 1),
            accu_loss_CLS1.item() / (step + 1),
            accu_loss_CLS2.item() / (step + 1),
            aux_loss_CGA.item() / (step + 1),
            aux_loss_SAM.item() / (step + 1),
            accu_num1.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return total_loss.item() / (step + 1), accu_loss_CLS1.item() / (step + 1), accu_loss_CLS2.item() / (step + 1), aux_loss_CGA.item() / (step + 1), aux_loss_SAM.item() / (step + 1), accu_num1.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    accu_num1 = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, mininterval=2)
    for step, data in enumerate(data_loader):
        images1, images2, labels = data
        sample_num += images1.shape[0]

        pred = model(images1.to(device), images2.to(device))
        pred1_classes = torch.max(pred[0], dim=1)[1]
        accu_num1 += torch.eq(pred1_classes, labels.to(device)).sum()

        data_loader.desc = "[valid epoch {}] acc: {:.3f}".format(epoch, accu_num1.item() / sample_num)

    return accu_num1.item() / sample_num


