# **Task 1.2.2: Improving Performance with NNI**

## 队伍：pyboys

## 1. 环境配置

nni==1.6

pytorch==1.2.0

## 2. Baseline

我们先直接运行main.py看看结果如何

baseline结果

```javascript
INFO (hpo) Eval Epoch [300/300] Loss = 0.448868 Acc = 0.847700
INFO (hpo) Final accuracy is: 0.847700
```

最终准确率为0.8477，因此还有很大的提升空间。

## 3. HPO

看一下main.py的参数，刚刚跑的都是默认参数，使用的是torchvision中的模型，然而torchvision中的模型都是针对ImageNet的图像尺寸来的，一般输入是224x224，然后cifar10的图像输入尺寸是32，所以torchvision中的模型的卷积核为7对于cifar10来说过大了，这可能是导致resnet18的准确率只能到0.84的原因。

```python
available_models = ['resnet18', 'resnet50', 'vgg16', 'vgg16_bn', 'densenet121', 'squeezenet1_1','shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'mnasnet1_0']
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
```

所以第一步就是将模型全部改了，将第一个卷积层的卷积核尺寸改成3

```python
self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
```

下一步就是选择需要搜索的超参了，我们选择搜索model、optimizer、initial_lr、ending_lr和cutout

search_space.json

```json
{
    "initial_lr":{"_type":"uniform", "_value":[0.001, 0.2]},
    "ending_lr":{"_type":"uniform", "_value":[0.001, 0]},
    "optimizer":{"_type":"choice", "_value":["SGD", "Adam", "RMSprop"]},
    "model":{"_type":"choice", "_value":["vgg16","vgg19", "resnet18","resnet34", "densenet121","senet18"]},
    "cutout":{"_type":"quniform","_value":[0,8,1]}
}
```

并修改main.py

```python
### 加载超参数
try:
    RCV_CONFIG = nni.get_next_parameter()
    for key,value in RCV_CONFIG.items():
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
```

```python
####模型加载
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
```

```python
### 回传结果给nni
best_top1 = 0
for epoch in range(1, args.epochs + 1):
    train(model, train_loader, criterion, optimizer, scheduler, args, epoch, device)
    top1, _ = test(model, test_loader, criterion, args, epoch, device)

    if top1 > best_top1:
        best_top1 = top1
    nni.report_intermediate_result(top1)
logger.info("Final accuracy is: %.6f", top1)
nni.report_final_result(best_top1)
```

nni的配置我们使用TPE算法，然后Assessor使用Curvefitting，能够根据预测结果来早停一些trail

config.yml

```yaml
authorName: default
experimentName: pytorch_cifar10
trialConcurrency: 2
maxExecDuration: 24h
maxTrialNum: 100
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
assessor:
  #choice: Medianstop, Curvefitting
  builtinAssessorName: Curvefitting
  classArgs:
    epoch_num: 50
    threshold: 0.95
trial:
  command: python3 trails.py
  codeDir: .
  gpuNum: 1
localConfig:
  maxTrialNumPerGpu:  1
```

WebUI

![image-20200614105546152](https://github.com/Kwongrf/NNI-Student-Program-2020-pyboys/blob/master/task1.2/1.2.2/img/image-20200614105546152.png)

从上图可以看出，搜索的超参和模型准确率能达到0.94，相比baseline的0.84有了非常大的提升。

![image-20200614105522101](https://github.com/Kwongrf/NNI-Student-Program-2020-pyboys/blob/master/task1.2/1.2.2/img/image-20200614105522101.png)

从上图我们可以看出，initial_lr对结果的影响很大，如果太大了，比如大于0.1了，准确率就无法达到很高，即使训练过程学习率会一直衰减。都没有RMSprop的结果的原因是刚开始有几个它的结果，但是准确率一直停留在0.1，所以我们就人为中止了该trail。



## 4. NAS

NAS的操作我们没有改得太复杂，基本都是将上一个阶段最好结果的参数拿来了，search.py和retrain.py都参考了nni的[示例代码](https://github.com/microsoft/nni/tree/master/examples/nas/darts)， 8层，batch-size=64，尝试了128但是显存超了，epochs我们一开始觉得会不会太少了，毕竟之前训练都是两三百个epoch起步，但是考虑到这是搜索架构，两三百epoch时间太久了，而且最后还需要retrain，所以没必要。

```shell
python search.py --batch-size 64 --epochs 50  --initial_lr 0.06 --ending_lr 0.0005 --channel 16 --weight_decay 5e-4
```

```python
parser = ArgumentParser("darts")
parser.add_argument("--layers", default=8, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--log-frequency", default=10, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--channels", default=16, type=int)
parser.add_argument("--unrolled", default=False, action="store_true")
parser.add_argument("--visualization", default=False, action="store_true")
args = parser.parse_args()

dataset_train, dataset_valid = datasets.get_dataset("cifar10")

model = CNN(32, 3, args.channels, 10, args.layers)
criterion = nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)
```

开始search

```python
Epoch 1 Training
Epoch [1/50] Step [1/391]  acc1 0.125000 (0.125000)  loss 2.423200 (2.423200)
Epoch [1/50] Step [11/391]  acc1 0.234375 (0.181818)  loss 2.432802 (2.219552)
Epoch [1/50] Step [21/391]  acc1 0.265625 (0.202381)  loss 1.975967 (2.188984)
Epoch [1/50] Step [31/391]  acc1 0.218750 (0.215222)  loss 1.976302 (2.127891)
Epoch [1/50] Step [41/391]  acc1 0.265625 (0.232851)  loss 1.843109 (2.073023)
Epoch [1/50] Step [51/391]  acc1 0.343750 (0.243873)  loss 1.859783 (2.058276)
Epoch [1/50] Step [61/391]  acc1 0.296875 (0.247951)  loss 1.990559 (2.034418)
Epoch [1/50] Step [71/391]  acc1 0.328125 (0.259683)  loss 1.879746 (1.995083)
Epoch [1/50] Step [81/391]  acc1 0.296875 (0.266782)  loss 1.790013 (1.973880)
Epoch [1/50] Step [91/391]  acc1 0.296875 (0.276786)  loss 1.782345 (1.947485)
Epoch [1/50] Step [101/391]  acc1 0.343750 (0.281714)  loss 1.793832 (1.934490)
Epoch [1/50] Step [111/391]  acc1 0.359375 (0.288851)  loss 1.643773 (1.915869)
Epoch [1/50] Step [121/391]  acc1 0.437500 (0.293518)  loss 1.780038 (1.903140)
Epoch [1/50] Step [131/391]  acc1 0.343750 (0.299857)  loss 1.910589 (1.887221)
```



搜索得到50个架构，分别保存在./checkpoint/中，我们挑选epoch_49.json进行retrain

epoch_49.json

```json
{
  "normal_n2_p0": 3,
  "normal_n2_p1": 3,
  "normal_n2_switch": [
    "normal_n2_p0",
    "normal_n2_p1"
  ],
  "normal_n3_p0": 3,
  "normal_n3_p1": 3,
  "normal_n3_p2": [],
  "normal_n3_switch": [
    "normal_n3_p0",
    "normal_n3_p1"
  ],
  "normal_n4_p0": [],
  "normal_n4_p1": 3,
  "normal_n4_p2": 3,
  "normal_n4_p3": [],
  "normal_n4_switch": [
    "normal_n4_p1",
    "normal_n4_p2"
  ],
  "normal_n5_p0": [],
  "normal_n5_p1": 3,
  "normal_n5_p2": 3,
  "normal_n5_p3": [],
  "normal_n5_p4": [],
  "normal_n5_switch": [
    "normal_n5_p1",
    "normal_n5_p2"
  ],
  "reduce_n2_p0": 0,
  "reduce_n2_p1": 3,
  "reduce_n2_switch": [
    "reduce_n2_p0",
    "reduce_n2_p1"
  ],
  "reduce_n3_p0": 0,
  "reduce_n3_p1": [],
  "reduce_n3_p2": 2,
  "reduce_n3_switch": [
    "reduce_n3_p0",
    "reduce_n3_p2"
  ],
  "reduce_n4_p0": 0,
  "reduce_n4_p1": [],
  "reduce_n4_p2": 2,
  "reduce_n4_p3": [],
  "reduce_n4_switch": [
    "reduce_n4_p0",
    "reduce_n4_p2"
  ],
  "reduce_n5_p0": [],
  "reduce_n5_p1": [],
  "reduce_n5_p2": 2,
  "reduce_n5_p3": 2,
  "reduce_n5_p4": [],
  "reduce_n5_switch": [
    "reduce_n5_p2",
    "reduce_n5_p3"
  ]
}
```



retrain参数选择的也是和HPO阶段最好模型一样的，不过batchsize还是64

```shell
python retrain.py --layers 8 --batch-size 64 --epochs 300 --arc-checkpoint "./checkpoints/epoch_49.json" --initial_lr 0.06 --ending_lr 0.0005 --channel 16 --weight_decay 5e-4
```

retrain结果

```javascript
INFO (nni) Train: [300/300] Final Prec@1 90.2880%
INFO (nni) Valid: [300/300] Step 000/156 Loss 0.214 Prec@(1,5) (92.2%, 100.0%)
INFO (nni) Valid: [300/300] Step 010/156 Loss 0.162 Prec@(1,5) (94.5%, 100.0%)
INFO (nni) Valid: [300/300] Step 020/156 Loss 0.181 Prec@(1,5) (93.9%, 99.9%)
INFO (nni) Valid: [300/300] Step 030/156 Loss 0.191 Prec@(1,5) (94.1%, 99.8%)
INFO (nni) Valid: [300/300] Step 040/156 Loss 0.211 Prec@(1,5) (94.0%, 99.7%)
INFO (nni) Valid: [300/300] Step 050/156 Loss 0.207 Prec@(1,5) (94.2%, 99.8%)
INFO (nni) Valid: [300/300] Step 060/156 Loss 0.211 Prec@(1,5) (93.9%, 99.8%)
INFO (nni) Valid: [300/300] Step 070/156 Loss 0.211 Prec@(1,5) (94.0%, 99.8%)
INFO (nni) Valid: [300/300] Step 080/156 Loss 0.212 Prec@(1,5) (93.9%, 99.7%)
INFO (nni) Valid: [300/300] Step 090/156 Loss 0.211 Prec@(1,5) (93.9%, 99.8%)
INFO (nni) Valid: [300/300] Step 100/156 Loss 0.205 Prec@(1,5) (94.1%, 99.8%)
INFO (nni) Valid: [300/300] Step 110/156 Loss 0.202 Prec@(1,5) (94.0%, 99.8%)
INFO (nni) Valid: [300/300] Step 120/156 Loss 0.201 Prec@(1,5) (94.1%, 99.8%)
INFO (nni) Valid: [300/300] Step 130/156 Loss 0.199 Prec@(1,5) (94.1%, 99.8%)
INFO (nni) Valid: [300/300] Step 140/156 Loss 0.201 Prec@(1,5) (94.0%, 99.8%)
INFO (nni) Valid: [300/300] Step 150/156 Loss 0.202 Prec@(1,5) (94.0%, 99.8%)
INFO (nni) Valid: [300/300] Step 156/156 Loss 0.201 Prec@(1,5) (94.0%, 99.8%)
INFO (nni) Valid: [300/300] Final Prec@1 94.0400%
INFO (nni) Final best Prec@1 = 94.1300%
```

NAS得到的架构retrain了300个epoch后得到的准确率为94.13%，这个分数还是有不少的提升空间的，毕竟现在最高的结果已经能达到0.98。
