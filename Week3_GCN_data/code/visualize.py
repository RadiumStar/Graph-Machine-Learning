import numpy as np
import matplotlib.pyplot as plt

def get_mean(data):
    return np.mean(data)

def get_std(data):
    return np.std(data)

def draw_train(x, y, std):
    # 绘制图表
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.errorbar(x, y, yerr=std, fmt='o', color='b', ecolor='g', capsize=5)
    plt.xlabel('ratio of training data')
    plt.ylabel('accuracy')
    plt.title('Texas Dataset')
    plt.xticks(x)
    plt.grid(True)
    plt.savefig('texas_ratio_of_training_data_and_accuracy.png', dpi = 600)
    plt.show()

if __name__ == '__main__':
    x = [.1, .2, .3, .4, .5, .6, .7, .8]
    y = [48.80, 48.31, 50.77, 50.34, 50.53, 52.63, 48.81, 46.05]
    std = [3.24, 1.47, 4.74, 5.92, 5.07, 2.15, 8.1, 5.74]

    draw_train(x, y, std)
