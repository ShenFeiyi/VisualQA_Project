# 做结果图

import numpy as np
import csv
from matplotlib import pyplot as plt


num_epochs = 5

fig, ax = plt.subplots(1, 2, figsize=(10,5))
# 分别做训练集与验证集的线
for phase in ['train', 'valid']:
    
    epoch = []
    loss = []
    acc = []
    
    for i in range(num_epochs):
        
        with open('./logs/{}-log-epoch-{:02d}.txt'.format(phase, i+1), 'r') as f:
            df = csv.reader(f, delimiter='\t') # 只有一行
            data = list(df) # 转换为列表，[[每行元素]]

        epoch.append(float(data[0][0]))
        loss.append(float(data[0][1]))
        acc.append(float(data[0][3]))

    if phase == 'train':
        plot0 = ax[0].plot(epoch, loss, label=phase, color='red', linewidth=3)
    else:
        plot0 = ax[0].plot(epoch, loss, label=phase, color='blue', linewidth=3)
            
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    plt.tight_layout()

    if phase == 'train':
        plot1 = ax[1].plot(epoch, acc, label=phase, color='red', linewidth=3)
    else:
        plot1 = ax[1].plot(epoch, acc, label=phase, color='blue', linewidth=3)
    
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    # 线标签设置
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')

# plt.show()
plt.savefig('./png/train.png', dpi = fig.dpi)

