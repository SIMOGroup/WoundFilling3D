from dataloader import plydataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from pathlib import Path
import torch.nn.functional as F
import datetime
import logging
from utils import test_semseg
from TSGCNet import TSGCNet
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# class SoftDiceLossV1(nn.Module):
#     '''
#     soft-dice loss, useful in binary segmentation
#     '''
#     def __init__(self,
#                  p=1,
#                  smooth=1):
#         super(SoftDiceLossV1, self).__init__()
#         self.p = p
#         self.smooth = smooth
#
#     def forward(self, logits, labels):
#         '''
#         inputs:
#             logits: tensor of shape (N, H, W, ...)
#             label: tensor of shape(N, H, W, ...)
#         output:
#             loss: tensor of shape(1, )
#         '''
#         probs = torch.sigmoid(logits)
#         numer = (probs * labels).sum()
#         denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
#         loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
#         return loss



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    """-------------------------- parameters --------------------------------------"""
    batch_size = 4
    k = 16

    """--------------------------- create Folder ----------------------------------"""
    experiment_dir = Path('./experiment_fixed_nqd/')
    experiment_dir.mkdir(exist_ok=True)
    current_time = str(datetime.datetime.now().strftime('%m-%d_%H-%M'))
    file_dir = Path(str(experiment_dir) + '/paper')
    file_dir.mkdir(exist_ok=True)
    log_dir, checkpoints = file_dir.joinpath('logs/'), file_dir.joinpath('checkpoints')
    log_dir.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)

    formatter = logging.Formatter('%(name)s - %(message)s')
    logger = logging.getLogger("all")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_dir) + '/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    writer = SummaryWriter(file_dir.joinpath('tensorboard'))

    """-------------------------------- Dataloader --------------------------------"""
    train_dataset = plydataset("data/train")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_dataset = plydataset("data/test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    """--------------------------- Build Network and optimizer----------------------"""
    model = TSGCNet(in_channels=12, output_channels=2, k=k)
#    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.cuda()
    optimizer = torch.optim.Adam(
    model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    """------------------------------------- train --------------------------------"""
    logger.info("------------------train------------------")
    best_mIoU = 0
    LEARNING_RATE_CLIP = 1e-5
    his_loss = []
    his_smotth = []
    class_weights = torch.ones(15).cuda()
    loss_each_epoch = []
    loss_val_epoch = []
    miou_epoch = []

    # criteria = SoftDiceLossV1()
    # criteria.cuda()

    for epoch in range(0, 50):
        print('='*100)
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        optimizer.param_groups[0]['lr'] = lr
        for i, (hihi, points_face, label_face, label_face_onehot, name, haha) in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            coordinate = points_face.transpose(2,1)
            coordinate, label_face = Variable(coordinate.float()), Variable(label_face.long())
            label_face_onehot = Variable(label_face_onehot)
            coordinate, label_face, label_face_onehot = coordinate.cuda(), label_face.cuda(), label_face_onehot.cuda()
            optimizer.zero_grad()
            pred = model(coordinate)

            label_face = label_face.view(-1, 1)[:, 0]
            pred = pred.contiguous().view(-1, 2)
            loss = F.nll_loss(pred, label_face)

            # loss = criteria(pred.contiguous().view(-1, 2), label_face.view(-1, 1))

            loss.backward()
            optimizer.step()
            his_loss.append(loss.cpu().data.numpy())

        print('Learning rate: %f' % (lr))
        print("loss: %f" % (np.mean(his_loss)))

        loss_each_epoch.append(np.mean(his_loss))
        writer.add_scalar("loss", np.mean(his_loss), epoch)
        metrics, mIoU, cat_iou, loss_val = test_semseg(model, test_loader, num_classes=2)
        miou_epoch.append(mIoU)
        loss_val_epoch.append(loss_val)

        print("Epoch: %d, accuracy= %f, mIoU= %f loss= %f , loss_val = %f" % (epoch, metrics['accuracy'], mIoU, np.mean(his_loss), loss_val))
        logger.info("Epoch: %d, accuracy= %f, mIoU= %f loss= %f , loss_val = %f" % (epoch, metrics['accuracy'], mIoU, np.mean(his_loss), loss_val))
        writer.add_scalar("accuracy", metrics['accuracy'], epoch)

        if (mIoU > best_mIoU):
            best_mIoU = mIoU
            print("best mIoU: ", mIoU)
            print(cat_iou)
            torch.save(model.state_dict(), '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_mIoU))
            best_pth = '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_mIoU)
            logger.info(cat_iou)
        his_loss.clear()
        writer.close()

    lst = {'loss_each_epoch': loss_each_epoch, 'loss_val_epoch': loss_val_epoch, 'miou_epoch': miou_epoch}
    df = pd.DataFrame(lst)
    df.to_csv('results_loss.csv', index=False)










