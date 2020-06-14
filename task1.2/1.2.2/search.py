import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from model import CNN
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from nni.nas.pytorch.darts import DartsTrainer
from utils import top_accuracy

logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument('--initial_lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--ending_lr', default=0, type=float, help='ending learning rate')
    parser.add_argument('--cutout', default=0, type=int, help='cutout length in data augmentation')
    parser.add_argument("--channels", default=16, type=int)
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    model = CNN(32, 3, args.channels, 10, args.layers)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), args.initial_lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=args.ending_lr)

    trainer = DartsTrainer(model,
                           loss=criterion,
                           metrics=lambda output, target: top_accuracy(output, target, topk=(1,)),
                           optimizer=optim,
                           num_epochs=args.epochs,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           batch_size=args.batch_size,
                           log_frequency=args.log_frequency,
                           unrolled=args.unrolled,
                           callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
    if args.visualization:
        trainer.enable_visualization()
    trainer.train()