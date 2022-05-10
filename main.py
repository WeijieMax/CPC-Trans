import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model.cpc_dataset import CPCDataset_Multimodal
from model.cpc_transformer import cpc_transformer_small as create_model
from utils.utils import read_split_data, train_one_epoch, evaluate, set_seed


def main(args):
    set_seed(seed=args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suffix = '-scratch' if args.weights == '' else '-pretrain'

    if os.path.exists("./logs/weights-"+str(args.fold)+suffix) is False:
        os.makedirs("./logs/weights-"+str(args.fold)+suffix)

    tb_writer = SummaryWriter('./logs')

    train_data_root1 = args.data_path+'/White_light/train_fold_'+str(args.fold)
    val_data_root1 = args.data_path+'/White_light/val_fold_'+str(args.fold)
    train_data_root2 = args.data_path+'/NBI/train_fold_'+str(args.fold)
    val_data_root2 = args.data_path+'/NBI/val_fold_'+str(args.fold)

    train_images_path1, train_images_label1 = read_split_data(train_data_root1)
    val_images_path1, val_images_label1 = read_split_data(val_data_root1)
    train_images_path2, train_images_label2 = read_split_data(train_data_root2)
    val_images_path2, val_images_label2 = read_split_data(val_data_root2)

    assert train_images_label1 == train_images_label2
    assert val_images_label1 == val_images_label2

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    train_dataset = CPCDataset_Multimodal(images_path1=train_images_path1,
                                       images_path2=train_images_path2,
                                       images_class=train_images_label1,
                                       transform=data_transform["train"])

    val_dataset = CPCDataset_Multimodal(images_path1=val_images_path1,
                                     images_path2=val_images_path2,
                                     images_class=val_images_label1,
                                     transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    print('args.num_classes is ', args.num_classes)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            try:
                del weights_dict[k]
            except:
                print('the model has no key named ', k)
        print(model.load_state_dict(weights_dict, strict=False))

    if 'pretrain' not in suffix:
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    else:
        mlp_params_id = list(map(id, model.head.parameters()))
        base_params = filter(lambda p: id(p) not in mlp_params_id, model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.lr}, 
            {'params': model.head.parameters(), 'lr': args.lr}], momentum=0.9, weight_decay=5E-5)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    is_best1 = 0
    best_epoch1 = -1

    for epoch in range(args.epochs):
        if epoch == args.early_stop_epoch:
            break
        # train
        train_total_loss, train_loss1, train_loss2, train_loss3, train_loss4, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch)
        scheduler.step()
        # validate
        val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

        tags = ['train_total_loss', 'train_loss1', 'train_loss2', 'train_loss3', 'train_loss4', 'train_acc', 'val_acc', "learning_rate"]
        tb_writer.add_scalar(tags[0], train_total_loss, epoch)
        tb_writer.add_scalar(tags[1], train_loss1, epoch)
        tb_writer.add_scalar(tags[2], train_loss2, epoch)
        tb_writer.add_scalar(tags[3], train_loss3, epoch)
        tb_writer.add_scalar(tags[4], train_loss4, epoch)
        tb_writer.add_scalar(tags[5], train_acc, epoch)
        tb_writer.add_scalar(tags[6], val_acc, epoch)
        tb_writer.add_scalar(tags[7], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > is_best1 and val_acc < train_acc:
            is_best1 = val_acc
            best_epoch1 = epoch
            print('epoch {} is currently best for modality 1'.format(epoch))
            # torch.save(model.state_dict(), "./weights-"+str(args.fold)+suffix+"/model_best1.pth")
            # print('save done.')

    print('\nmodality wl: best epoch is {} and best metric result is {}'.format(best_epoch1, is_best1))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--early_stop_epoch', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default='', # F:\Research\Data\CPC
                        help='dataset path')
    parser.add_argument('--weights', type=str, default='',  # ./vit_small_patch16_224_in21k.pth
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
