# Task 1.2.1: Training a classifier
## 队伍：pyboys
## 1. 环境配置
由于使用了pytorch的官方样例，只需要安装pytorch和torchvision
```
pip install torch torchvision
```
## 2. 加载数据集
本次使用了CIFAR10图片分类数据集，该数据集共包含10个类别、共60000张图片，每个类别包含6000张。其中的50000张被用作训练集，10000张用作测试集。使用torchvision的API下载该数据集，使用数据集构建dataloader，并使用transfrom对图片进行归一化等预处理：
```python
# load the CIFAR 10 dataset
# get the dataloaders for trainset and testset

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```
## 3. 定义神经网络
我们使用一个简单的卷积网络来进行实验<br>
网络结构为<br>
Conv2d -> relu -> MaxPool2d -> Conv2d -> relu -><br>
MaxPool2d -> Linear -> relu -> Linear -> relu -> Linear
```python
# define the classification network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
## 4. 使用训练集训练神经网络
我们使用SGD作为优化器，交叉熵损失作为损失函数对模型进行训练，batch-size=4，由于CIFAR10数据集没有划分验证集，因此我们将训练的epoch数定死为10个。在训练完成后，将模型参数保存为一个文件```cifar_net.pth```。为了加速训练过程，在可以使用GPU时使用GPU进行训练
```python
# Training: Use 4 batches for training and 1 batch for validation
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```
经过5个epoch的迭代，训练集损失从2.194降到1.051
```
[1,  2000] loss: 2.194
[1,  4000] loss: 1.842
[1,  6000] loss: 1.665
[1,  8000] loss: 1.570
[1, 10000] loss: 1.520
[1, 12000] loss: 1.500
[2,  2000] loss: 1.404
[2,  4000] loss: 1.395
[2,  6000] loss: 1.347
[2,  8000] loss: 1.334
[2, 10000] loss: 1.310
[2, 12000] loss: 1.301
[3,  2000] loss: 1.245
[3,  4000] loss: 1.216
[3,  6000] loss: 1.228
[3,  8000] loss: 1.215
[3, 10000] loss: 1.197
[3, 12000] loss: 1.178
[4,  2000] loss: 1.115
[4,  4000] loss: 1.111
[4,  6000] loss: 1.113
[4,  8000] loss: 1.141
[4, 10000] loss: 1.124
[4, 12000] loss: 1.130
[5,  2000] loss: 1.011
[5,  4000] loss: 1.050
[5,  6000] loss: 1.070
[5,  8000] loss: 1.063
[5, 10000] loss: 1.050
[5, 12000] loss: 1.051
```
## 5. 使用测试集测试网络性能
从```cifar_net.pth```加载模型参数，对测试集数据进行预测，并对比ground truth计算总体准确率及每个类别下的准确率
```python
# Testing
net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# print the accuracy for each class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```
测试集上的总体准确率及每个类别的准确率如下：
```
Accuracy of the network on the 10000 test images: 62 %
Accuracy of plane : 70 %
Accuracy of   car : 85 %
Accuracy of  bird : 42 %
Accuracy of   cat : 39 %
Accuracy of  deer : 63 %
Accuracy of   dog : 55 %
Accuracy of  frog : 54 %
Accuracy of horse : 69 %
Accuracy of  ship : 76 %
Accuracy of truck : 68 %
```
