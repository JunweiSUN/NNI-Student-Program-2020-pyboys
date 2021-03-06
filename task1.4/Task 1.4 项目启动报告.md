# Task 1.4 AutoML for Graphs 项目启动报告
## 队伍：pyboys
## 1. 项目概述
我们目前正在参加[KDDCup 2020 automl track](https://www.automl.ai/competitions/3),该比赛的目标是将自动机器学习应用于图相关的机器学习任务上，任务是在给定的时间内生成一个节点分类模型对图节点进行分类，评价指标为在各个图数据集上的节点分类正确率。我们希望能够借助nni来快速搜索（图神经网络）模型的架构和超参数，使得我们的模型能够在不同的数据集上均取得比较好的表现。
## 2. 项目预期目标
在KDDCup 2020 automl track的final phase中取得top10的成绩。
## 3. 项目难点
在本次nni学生项目启动之前，我们已经使用nni进行了一些尝试。目前遇到的主要问题有：<br>
1. 本次比赛每个数据集的time_budget比较短，通常只有几百秒，这对automl算法的性能提出了较高的要求
2. 图数据由于每个节点彼此之间不独立，因此通常不支持分batch训练，这就需要我们魔改原始的NAS Train来满足需求
3. 我们尝试了ENAS和DARTS两个NAS算法，性能没有明显提升，正在找bug中...
4. 不同于NAS模块，nni的超参搜索模块不支持直接使用python代码进行one-shot的搜索，必须使用nnictl，如何将nnictl结合到我们的预测接口中也是一个需要思考的问题
5. 之前尝试了nni的特征选择算法，但是报了import error，似乎是pytorch版本不匹配造成的，目前还没有解决这部分问题
## 4. 项目规划与实施方案
即日起至该比赛结束（6月4日），使用nni的各个模块，包括但不限于特征选择、超参搜索、NAS等等，优化我们的节点分类模型，争取在比赛中取得好的名次。在比赛结束后总结和整理代码，提交结项报告。
