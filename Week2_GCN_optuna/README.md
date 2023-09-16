# Week 2 GCN HyperParameter
## 1. 探索 GCN 超参数对准确率的影响
> 说明：统一使用 cora 数据集进行测试，epochs为200

- GCN 层数对准确率的影响(%)

    | layers |    2    |    3    |    4    |    5    |    6    |
    |:------:|:-------:|:-------:|:-------:|:-------:|:-------:|
    |accuracy| 83.47 ± 0.54  | 73.10 ± 2.05  | 57.4 ± 8.78  | 47.70 ± 3.50  | 43.87 ± 11.27  |

    ![Alt text](<Effect of GCN Layer Number on Accuracy.png>)
    可以看出，随着层数的增加，准确率逐渐下降，而且模型越来越不稳定

- 学习率对准确率的影响

    | lr     | 0.005   | 0.01    |0.015    | 0.02    | 0.025   |  0.03 | 0.04 | 0.05 | 0.1 | 0.2 |
    |:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-----:|:-:|:-:|:-:|:-:|
    |accuracy| 78.76±1.294  | 83.47 ± 0.54  | 83.4±0.30  | 83.1±0.92  | 83.8±0.44 | 83.13 ± 0.30 | 83.35 ± 1.00 | 82.95 ± 0.79 | 83.0 ± 1.5 | 82.0 ± 0.36 |

    ![Alt text](<Effect of GCN Learning Rate on Accuracy.png>)
    当学习率过小时，模型准确率较低，可能原因是学习过慢，在200个epoch内学习效率不及更高的学习率；当学习率处于0.01~0.05时，模型准确率较稳定，而后随着学习率增大呈现递减趋势，且逐渐不稳定，原因可能是学习率过大导致模型更新的步长较大，无法靠近最优解

- weight_decay对准确率的影响（5e^）

    | weight decay(指数级别) | -6 | -5 | -4 | -3 | -2 |
    |:------:|:-------:|:-------:|:-------:|:-------:|:-------:|
    |accuracy| 81.5±0.50 | 83.10 ± 1.27 | 83.47 ± 0.54 | 43.23 ± 0.37 | 30.90 ± 0.0 |

    ![Alt text](<Effect of GCN Weight Decay on Accuracy.png>)
    权重衰减较合适时，可以保证模型性能良好；如果权重衰减设置过大会导致准确率骤降，表示模型欠拟合，无法学习有效特征
    
## 2. 学习使用Optuna框架
`optunaGCN/optimizer.py`代码中使用了Optuna库提供的可视化函数来分析GCN模型的优化过程和结果。下面是每个可视化函数的作用解释：

1. `optuna.visualization.plot_contour(study)`：绘制超参数优化的轮廓图，绘制的轮廓图将显示超参数之间的相互关系和目标函数值的分布情况

2. `optuna.visualization.plot_optimization_history(study)`：绘制优化历史的目标值随着迭代次数的变化曲线图。可以帮助观察目标值在整个优化过程中的变化趋势。

3. `optuna.visualization.plot_param_importances(study)`：绘制超参数的重要性图。可以帮助了解不同超参数对优化结果的影响程度，从而更好地选择重要的超参数进行调优。

4. `optuna.visualization.plot_parallel_coordinate(study, ['layer_size', 'hidden_channels', 'lr', 'weight_decay', 'dropout_rate'])`：绘制超参数的平行坐标图。可以直观地展示不同超参数组合与目标值之间的关系，帮助观察超参数的取值范围和目标值之间的关系。

5. `optuna.visualization.plot_slice(study, ['layer_size', 'hidden_channels', 'lr', 'weight_decay', 'dropout_rate'])`：绘制超参数的切片图。可以帮助观察每个超参数与目标值之间的关系，从而更好地理解超参数对优化结果的影响。


- 结果
  - cora
    ``` python
    Best Parameters: {'hidden_channels': 76, 
                      'lr': 0.0017085304666074985, 
                      'weight_decay': 0.00020925329921608887, 
                      'dropout_rate': 0.3852306303116923}
    Best Accuracy: 0.827
    ```
  - citeseer
    ``` python
    Best Parameters: {'hidden_channels': 87, 
                      'lr': 0.008103369471824794, 
                      'weight_decay': 0.0024277062769735708, 
                      'dropout_rate': 0.3766924087636698}
    Best Accuracy: 0.726
    ```
