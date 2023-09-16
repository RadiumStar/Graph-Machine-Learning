import matplotlib.pyplot as plt  

def draw(x, accuracy, error, description, xlabel, ylabel = 'Accuracy (%)'):
    plt.plot(x, accuracy, marker='o')  
    
    for i in range(len(x)):  
        plt.errorbar(x[i], accuracy[i], yerr=error[i], fmt='none')  
    
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.title(description) 
    plt.grid(True)
    
    plt.savefig(description + '.png', dpi = 600)
    plt.show()

# 数据  
layers = [2, 3, 4, 5, 6]  
accuracy = [83.47, 73.10, 57.4, 47.70, 43.87]  
error = [0.54, 2.05, 8.78, 3.50, 11.27]  

draw(layers, accuracy, error, 
     'Effect of GCN Layer Number on Accuracy', 'Number of Layers')

lrs = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.04, 0.05, 0.1, 0.2]
accuracy = [78.76, 83.47, 83.4, 83.1, 83.8, 83.13, 83.35, 82.95, 83.0, 82.0]
error = [1.294, 0.54, 0.3, 0.92, 0.44, 0.3, 1, 0.79, 1.5, 0.36]
  
draw(lrs, accuracy, error, 
     'Effect of GCN Learning Rate on Accuracy', 'learning rate')

weight_decay = [-6, -5, -4, -3, -2]
accuracy = [81.5, 83.1, 83.47, 43.23, 30.90]
error = [0.5, 1.27, 0.54, 0.37, 0]

draw(weight_decay, accuracy, error, 
     'Effect of GCN Weight Decay on Accuracy', 'weight decay(5e^)')

