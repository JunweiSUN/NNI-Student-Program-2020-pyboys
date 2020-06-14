# Task 1.1 使用体验文档
## 1. 安装及使用
### 1.1 实验环境
我们的实验环境如下：

```
Ubuntu 18.04
NVIDIA RTX 2080Ti x3
CUDA 10.2
docker 19.03.5
git 2.7.4
```
### 1.2 安装nni
我们使用anaconda来安装和配置nni<br>
首先，为项目创建一个新的虚拟环境并激活,同时为项目创建一个单独的目录：
```
conda create -n nni_program python=3.6
conda activate nni_program
cd ~ && mkdir nni_program
```
接下来使用pip安装nni:
```
pip install nni
```
### 1.3 运行Pytorch的官方MNIST样例
首先，下载pytorch官方的MNIST样例
```
git clone https://github.com/pytorch/examples.git
mv examples/mnist .
rm -rf examples  # 只保留mnist样例，删除其他样例
```
之后，安装样例的相关依赖
```
cd mnist
pip install -r requirements.txt
```
运行官方样例
```
CUDA_VISIBLE_DEVICES=2 python main.py # 卡2有人在用所以用了卡2
```
代码开始运行后，首先下载了pytorch中的MNIST数据集，然后开始训练，共训练了14个Epoch，最终准确率达到了99%
```
Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.009292
Train Epoch: 14 [51840/60000 (86%)]	Loss: 0.055482
Train Epoch: 14 [52480/60000 (87%)]	Loss: 0.021806
Train Epoch: 14 [53120/60000 (88%)]	Loss: 0.000568
Train Epoch: 14 [53760/60000 (90%)]	Loss: 0.110339
Train Epoch: 14 [54400/60000 (91%)]	Loss: 0.009167
Train Epoch: 14 [55040/60000 (92%)]	Loss: 0.000871
Train Epoch: 14 [55680/60000 (93%)]	Loss: 0.061742
Train Epoch: 14 [56320/60000 (94%)]	Loss: 0.030131
Train Epoch: 14 [56960/60000 (95%)]	Loss: 0.007577
Train Epoch: 14 [57600/60000 (96%)]	Loss: 0.007833
Train Epoch: 14 [58240/60000 (97%)]	Loss: 0.005011
Train Epoch: 14 [58880/60000 (98%)]	Loss: 0.005237
Train Epoch: 14 [59520/60000 (99%)]	Loss: 0.002461

Test set: Average loss: 0.0274, Accuracy: 9913/10000 (99%)

```
### 1.4 为MNIST样例添加nni元素
本次实验我们选定原始样例中的4个超参数进行修改，分别是batch_size, lr, 线性层的hidden_size以及dropout_rate
对于hidden_size和dropout_rate，原始的样例中是写死的：
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```
我们修改了构造函数使得这些参数可以被修改
```
# main.py line 12
class Net(nn.Module):
    def __init__(self, linear_hidden, dropout_rate_1, dropout_rate_2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout_rate_1)
        self.dropout2 = nn.Dropout2d(dropout_rate_1)
        self.fc1 = nn.Linear(9216, linear_hidden)
        self.fc2 = nn.Linear(linear_hidden, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```
同时在main函数中为parser添加这些参数
```
# main.py line 81
parser.add_argument('--linear_hidden', type=int, default=128, metavar='N',
                        help='hidden size of linear layers')
parser.add_argument('--dropout_rate_1', type=float, default=0.25, metavar='N',
                        help='dropout rate of the first dropout layer')
parser.add_argument('--dropout_rate_2', type=float, default=0.5, metavar='N',
                        help='dropout rate of the seconde dropout layer')
```
向模型中传入这些参数：
```
# main.py line 132
# model = Net().to(device)
model = Net(args.linear_hidden, args.dropout_rate_1, args.dropout_rate_2).to(device)
```
接下来，为代码添加nni元素，从而更新parser中的相关超参数（需要import nni）
```
# main.py line 104
tuner_params = nni.get_next_parameter()
args.batch_size = tuner_params['batch_size']
args.lr = tuner_params['lr']
args.linear_hidden = tuner_params['linear_hidden']
args.dropout_rate_1 = tuner_params['dropout_rate_1']
args.dropout_rate_2 = tuner_params['dropout_rate_2']
```
我们选择使用准确率作为评价指标，因此修改源代码中的test函数，使得可以返回当前epoch的准确率
```
# main.py line 53
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)
```
在main函数中上报每个epoch的准确率，并在训练完成后上报最终准确率
```
# main.py line 137
best_test_acc = 0
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    current_test_acc = test(model, device, test_loader)

    # report current result
    nni.report_intermediate_result(current_test_acc)
    if current_test_acc > best_test_acc:
        best_test_acc = current_test_acc
    scheduler.step()

# report final (best) result
nni.report_final_result(best_test_acc)
```
### 1.5 配置config.yml和search_space.json
在同目录下创建config.yml和search_space.json两个文件，内容如下

config.yml
```
authorName: pyboys
experimentName: example_mnist_pytorch_with_nni
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 100
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python main.py
  codeDir: .
  gpuNum: 1
```
search_space.json
```
{
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    "linear_hidden":{"_type":"choice","_value":[32, 64, 128, 256]},
    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
    "dropout_rate_1":{"_type":"choice","_value":[0.2, 0.25, 0.3]},
    "dropout_rate_2":{"_type":"choice","_value":[0.3, 0.4, 0.5]}
}
```
### 1.6 开始实验
在代码目录使用以下命令开始实验（使用卡2）
```
CUDA_VISIBLE_DEVICES=2 nnictl create --config config.yml
```
此时nnictl开始在后台运行，可通过8080端口访问web页面
```
INFO: expand searchSpacePath: search_space.json to /home/sjw/nni_program/mnist/search_space.json 
INFO: expand codeDir: . to /home/sjw/nni_program/mnist/. 
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
------------------------------------------------------------------------------------
The experiment id is TPXqtDgo
The Web UI urls are: http://127.0.0.1:8080   http://10.112.76.240:8080   http://172.17.0.1:8080
------------------------------------------------------------------------------------

You can use these commands to get more information about the experiment
------------------------------------------------------------------------------------
         commands                       description
1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl top                    monitor the status of running experiments
4. nnictl log stderr             show stderr log content
5. nnictl log stdout             show stdout log content
6. nnictl stop                   stop an experiment
7. nnictl trial kill             kill a trial job by id
8. nnictl --help                 get help information about nnictl
------------------------------------------------------------------------------------
Command reference document https://nni.readthedocs.io/en/latest/Tutorial/Nnictl.html
------------------------------------------------------------------------------------
```
### 1.7 实验结果总结与分析
请详见样例分析文档
# 2. 使用感受
总体来说体验比较好。安装上基本没有遇到问题，需要在原代码中做的修改也比较少。有一点建议是希望能够像NAS一样提供参数搜索模块的one-shot api，使用户可以直接在代码中获取评价结果,而不是必须要使用nnictl来观察结果。另外，我们是基于pytorch的官方mnist example进行了修改，对比nni提供的mnist样例我们发现，nni的mnist样例中将argparse的部分放到了get_params()里面，这样似乎只能使用给parser设定的默认值进行训练而不能从命令行传入（除了需要tune的参数以外的）参数；另外，nni的样例中最后report的是最后一个epoch的acc，而我们report的是最优的acc。