import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import os
import time
import pickle
from dataset.Criteo import CriteoDatasetTorch
from torch.utils.data import DataLoader
from model.DeepFM.DeepFM_PyTorch import DeepFM
from sklearn.metrics import roc_auc_score


def train(args, model, device, optimizer, loss_func, scheduler, train_loader, val_loader):
    writer = SummaryWriter()
    for epoch in range(args.epochs):
        print('---------------------------------------------------------------------')
        print('epoch is %d and the lr is %f' % (epoch, optimizer.param_groups[0]['lr']))
        start_time = time.time()
        model.train()
        gt = []
        pred = []
        train_loss = 0.0
        train_total = 0
    # train
        for index, (dense_feat, sparse_feat, label) in enumerate(train_loader):
            dense_feat = dense_feat.to(device, dtype=torch.float32)
            sparse_feat = sparse_feat.to(device, dtype=torch.long)
            label = label.to(device, dtype=torch.float32)
            output = model(dense_feat, sparse_feat).squeeze()
            optimizer.zero_grad()
            loss = loss_func(output, label)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
            optimizer.step()
            print('label:', sum(label > 0.5))
            print('out:', sum(output > 0.5))

            train_loss += loss.item()
            train_total += 1

            gt.extend(list(label.cpu().detach().numpy()))
            pred.extend(list(output.cpu().detach().numpy()))

        auc = roc_auc_score(np.array(gt), np.array(pred))
        log_loss = float(train_loss / train_total)
        scheduler.step()
        print('epoch %d cost %3f sec' % (epoch, time.time() - start_time))
        print('train AUC: %.9f   train loss: %.9f' % (auc, log_loss))
        writer.add_scalar('train_log_loss', log_loss, epoch)
        writer.add_scalar('train_auc', auc, epoch)
        model_save_path = os.path.join(args.model_save_dir, 'train_epoch{}.pkl'.format(epoch))
        print('Save model to {}'.format(model_save_path))
        torch.save(model, model_save_path)

        gt.clear()
        pred.clear()

    # val
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_total = 0
            for index, (dense_feat, sparse_feat, label) in enumerate(val_loader):
                dense_feat = dense_feat.to(device, dtype=torch.float32)
                sparse_feat = sparse_feat.to(device, dtype=torch.long)
                label = label.to(device, dtype=torch.float32)
                label = label.squeeze()
                output = model(dense_feat, sparse_feat)
                output = output.squeeze()
                loss = loss_func(output, label, reduction='sum')
                val_loss += loss.item()
                val_total += label.size(0)
                gt.extend(list(label.cpu().detach().numpy()))
                pred.extend(list(output.cpu().detach().numpy()))

            auc = roc_auc_score(np.array(gt), np.array(pred))
            log_loss = float(val_loss / val_total)
            print('val AUC: %.9f    val log loss: %.9f' % (auc, log_loss))
            writer.add_scalar('val_log_loss', log_loss, epoch)
            writer.add_scalar('val_auc', auc, epoch)


def main(params):
    root_path = '/home/xlm/project/recommendation/dataset/criteo/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=root_path, help='root path')
    parser.add_argument('--model_save_dir', type=str, default='/data/xlm/recommendation/FM/checkpoints_fm/')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer we are using')
    parser.add_argument('--loss', type=str, default='BCE', help='loss_func we are using')
    parser.add_argument('--embed_dim', type=int, default=8, help='embedding dim of sparse features')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids we are using')
    args = parser.parse_args(params)

    # set gpu we are using
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # get dataset
    with open(os.path.join(args.root_path, 'dense.info'), 'rb') as f:
        dense_feature_columns = pickle.load(f)
    with open(os.path.join(args.root_path, 'sparse.info'), 'rb') as f:
        sparse_feature_columns = pickle.load(f)

    train_data = CriteoDatasetTorch(args.root_path, filename='train.lmdb')
    val_data = CriteoDatasetTorch(args.root_path, filename='val.lmdb')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=100)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=100)
    print('success load dataset')

    # cuda device
    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = DeepFM(dense_feature_columns, sparse_feature_columns, embed_dim=8, dnn_layers=(256, 256),
                   init_std=0.01, device=device)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).to(device)
            print('multi cuda:{}'.format(torch.cuda.device_count()))
        else:
            model = model.to(device)
            print('single cuda:{}'.format(torch.cuda.device_count()))

    # pytorch使用weight_decay添加L2正则化（对bias 和 bn不进行l2正则）
    weight_decay_list = (param for name, param in model.named_parameters()
                         if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters()
                     if name[-4:] == 'bias' or "bn" in name)

    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.0}]

    # optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=1e-3)
    else:
        print('not support optimizer!')
        return None

    # loss func
    loss_func = torch.nn.functional.binary_cross_entropy

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train(args, model, device, optimizer, loss_func, scheduler, train_loader, val_loader)


if __name__ == '__main__':
    params = [
        '--epochs',      '200',
        '--batch_size',  '4096',
        '--optimizer',   'Adam',
        '--loss',        'BCE',
        '--use_gpu',     'True',
        '--gpu_ids',     '1, 2, 3, 4',
        '--lr',          '0.001'
    ]

    main(params)