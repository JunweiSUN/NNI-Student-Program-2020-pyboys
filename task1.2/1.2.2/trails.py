import argparse
import functools
import logging
import os
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# import torchvision.models as models
import torchvision.transforms as transforms

from utils import AverageMeterGroup, accuracy, prepare_logger, reset_seed
import nni

logger = logging.getLogger('hpo')

from models import *

def data_preprocess(args):
    def cutout_fn(img, length):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

    augmentation = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    cutout = [functools.partial(cutout_fn, length=int(args.cutout))] if args.cutout else []
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    transform_train = transforms.Compose(augmentation + normalize + cutout)
    transform_test = transforms.Compose(normalize)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader


def train(model, loader, criterion, optimizer, scheduler, args, epoch, device):
    logger.info('Current learning rate: %.6f', optimizer.param_groups[0]['lr'])
    model.train()
    meters = AverageMeterGroup()

    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        meters.update({'acc': accuracy(logits, targets), 'loss': loss.item()})

        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info('Epoch [%d/%d] Step [%d/%d]  %s', epoch, args.epochs, step + 1, len(loader), meters)
        scheduler.step()
    return meters.acc.avg, meters.loss.avg


def test(model, loader, criterion, args, epoch, device):
    model.eval()
    meters = AverageMeterGroup()
    correct = loss = total = 0.
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            bs = targets.size(0)
            logits = model(inputs)
            loss += criterion(logits, targets).item() * bs
            correct += accuracy(logits, targets) * bs
            total += bs
    logger.info('Eval Epoch [%d/%d] Loss = %.6f Acc = %.6f',
                epoch, args.epochs, loss / total, correct / total)
    return correct / total, loss / total


def main(args):
    reset_seed(args.seed)
    prepare_logger(args)

    logger.info("These are the hyper-parameters you want to tune:\n%s", pprint.pformat(vars(args)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = data_preprocess(args)
    # model = models.__dict__[args.model]()
    if args.model == 'vgg16':
        model = VGG('VGG16')
    elif args.model == 'vgg19':
        model = VGG('VGG19')
    elif args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'resnet34':
        model = ResNet34()
    elif args.model == 'senet18':
        model = SENet18()
    elif args.model == 'densenet121':
        model = densenet_cifar()
    # if args['model'] == 'resnet50':
    #     model = models.resnet50(pretrained=True)
    # if args['model'] == 'googlenet':
    #     model = GoogLeNet()
    # if args['model'] == 'mobilenet':
    #     model = MobileNet()
    # if args['model'] == 'dpn92':
    #     model = DPN92()
    # if args['model'] == 'shufflenetg2':
    #     model = ShuffleNetG2()
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.initial_lr)
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.initial_lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
    elif args.optimizer == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.initial_lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # if args.optimizer == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    # else:
    #     if args.optimizer == 'sgd':
    #         optimizer_cls = optim.SGD
    #     elif args.optimizer == 'rmsprop':
    #         optimizer_cls = optim.RMSprop
    #     optimizer = optimizer_cls(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.ending_lr)

    best_top1 = 0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, scheduler, args, epoch, device)
        top1, _ = test(model, test_loader, criterion, args, epoch, device)

        if top1 > best_top1:
            best_top1 = top1
        nni.report_intermediate_result(top1)
    logger.info("Final accuracy is: %.6f", top1)
    nni.report_final_result(best_top1)


if __name__ == '__main__':
    available_models = ['resnet18', 'resnet50', 'vgg16', 'vgg16_bn', 'densenet121', 'squeezenet1_1',
                        'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'mnasnet1_0']
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--initial_lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--ending_lr', default=0, type=float, help='ending learning rate')
    parser.add_argument('--cutout', default=0, type=int, help='cutout length in data augmentation')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer type', choices=['sgd', 'rmsprop', 'adam'])
    parser.add_argument('--momentum', default=0.9, type=int, help='optimizer momentum (ignored in adam)')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers to preprocess data')
    parser.add_argument('--model', default='resnet18', choices=available_models, help='the model to use')
    parser.add_argument('--grad_clip', default=0., type=float, help='gradient clip (use 0 to disable)')
    parser.add_argument('--log_frequency', default=20, type=int, help='number of mini-batches between logging')
    parser.add_argument('--seed', default=42, type=int, help='global initial seed')
    args = parser.parse_args()

    try:
        RCV_CONFIG = nni.get_next_parameter()
        for key,value in RCV_CONFIG.items():
            # args[key] = value
            if key == "model":
                args.model == value
            elif key == "optimizer":
                args.optimizer = value
            elif key == "initial_lr":
                args.initial_lr = value
            elif key == "ending_lr":
                args.ending_lr = value
            elif key == "cutout":
                args.cutout = value
            elif key == "weight_decay":
                args.weight_decay = value
            elif key == "momentum":
                args.momentum = value
            
        main(args)
    
    except Exception as exception:
        logger.exception(exception)
        raise