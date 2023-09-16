import optuna
from train import train  # 导入你的GCN模型
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice

def objective(trial):
    # 使用Optuna生成超参数的建议值
    hidden_size = trial.suggest_int('hidden_size', 16, 128)   # 范围可以根据你的需求进行调整
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.7)

    # 创建GCN模型
    accuracy = train(hidden = hidden_size, dropout = dropout_rate)

    return accuracy

study = optuna.create_study(direction='maximize')   # 根据你的优化目标选择'maximize'或'minimize'
study.optimize(objective, n_trials = 10)   # 设置适当的迭代次数

# 打印最佳超参数和对应的指标值
best_params = study.best_params
best_value = study.best_value
print('Best params:', best_params)
print('Best value:', best_value)

# 可视化优化历史
plot_optimization_history(study)

# 可视化超参数的并行坐标图
plot_parallel_coordinate(study)

# 可视化超参数的切片图
plot_slice(study)
