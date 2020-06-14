# Task 1.4 自主项目结项报告
## 1. 项目进展与结果
我们的项目是借助nni参加AutoML相关的竞赛，具体来说，是借助nni在KDDCup 2020 automl track的final phase中取得top10的成绩。<br>
目前该竞赛已经结束，我们最终取得了第六名的成绩。
## 2. 实验信息
在我们的项目中，主要借助了nni的Feature Engineering功能，我们使用了其中的[Hyperopt Tuner](https://github.com/microsoft/nni/blob/98f66f76d310b0e0679823d966fdaa6adafb66c2/src/sdk/pynni/nni/hyperopt_tuner/hyperopt_tuner.py)来对图神经网络中的超参数进行调整，示例代码如下：
```python
self.tuner = HyperoptTuner(algorithm_name='tpe', optimize_mode='maximize')
search_space = {
        "dropedge_rate": {
            "_type": "choice",
            "_value": [self.info['dropedge_rate']]
        },
        "dropout_rate": {
            "_type": "choice",
            "_value": [self.info['dropout_rate']]
        },
        "num_layers": {
            "_type": "quniform",
            "_value": [1, 3, 1]
        },
        "hidden": {
            "_type": "quniform",
            "_value": [4, 7, 1]
        },
        "lr":{
            "_type": "choice",
            "_value": [0.005]
        },
        'num_stacks' : {
            "_type": "quniform",
            "_value": [1, 5, 1]
        },
        'conv_layers' : {
            "_type": "quniform",
            "_value": [1, 5, 1]
        },
        'use_linear': {
            "_type":"choice",
            "_value":[True, False]
        }
    }
self.tuner.update_search_space(search_space)


self.hyperparameters = self.tuner.generate_parameters(round_num-1)

# training......

self.tuner.receive_trial_result(round_num-1,self.hyperparameters,val_score)

```
每次从tuner中获得一组超参数，对模型进行训练后得到验证集性能并反馈给tuner以生成下一组超参数。我们使用的优化方法是TPE。<br>
通过使用nni对模型进行超参数调优，我们在短时间内就在每个基模型上取得了比较好的效果，并最终通过模型融合在B榜取得了不错的成绩。
我们也尝试了使用ENAS对模型架构进行搜索，但可能是由于ENAS本身算法的限制，实现出来的效果并不好，因此在最后的模型中只保留了参数调优的部分，NAS的部分被舍弃。
## 3. 实现代码
我们的全部代码已经开源在[这里](https://github.com/JunweiSUN/AutoGL)，欢迎使用~