import argparse
import torch.optim as opt
from model.plnet import PLNet, MultiBoxLoss
from dataset.datasets import PascalVOCDataset
from torch.utils.data import DataLoader
from utils import *
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'

def initialize():
    # 加载数据
    train_dataset = PascalVOCDataset(data_folder='/home/haogao/works/PLN/vocdata/jsonfiles/', split='train',
                                     keep_difficult=args.keep_difficult)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn, num_workers=args.workers, drop_last=True)
    # test_dataset = PascalVOCDataset(data_folder='/home/zhehaowang/ghao/PLN/vocdata/jsonfiles', split='test',
    #                                 keep_difficult=args.keep_difficult)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
    #                          collate_fn=test_dataset.collate_fn, drop_last=True)

    # 实例化model
    model = PLNet(args)
    model = model.to(device)
    optimizer = opt.RMSprop(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum)
    # 定义损失函数
    criterion = MultiBoxLoss(args)
    criterion = criterion.to(device)
    # 训练网络
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)

        # 保存模型
        save_checkpoint(epoch, model)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i, (imgs, boxs, labels, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        lt_label, lb_label, rt_label, rb_label = make_label(boxs, labels, args.S, args.B)
        lt, rt, lb, rb = model(imgs)

        loss = criterion(lt, rt, lb, rb, lt_label, rt_label, lb_label, rb_label)
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))
        batch_time.update(time.time() - start)

        start = time.time()
        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))
    del lt_label, lb_label, rt_label, rb_label, lt, rt, lb, rb, imgs, boxs, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="key parameters of PLN")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.00004)
    parser.add_argument('--datasets', type=str, default='voc2007', choices=['voc2007'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--S', type=int, default=14)
    parser.add_argument('--B', type=int, default=2)
    parser.add_argument('--keep_difficult', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--n_classes',type=int,default=20+1)
    parser.add_argument('--loss_weights',type=list,default=[1,1,1])
    args = parser.parse_args()
    initialize()
