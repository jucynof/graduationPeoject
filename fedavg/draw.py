import pandas as pd
import matplotlib.pyplot as plt

def drawOneTest():
    # 读取数据
    accuracy = pd.read_csv('./test_accuracy.csv', header=None).squeeze()
    loss = pd.read_csv('./test_loss.csv', header=None).squeeze()
    cost = pd.read_csv('./test_cost.csv', header=None).squeeze()
    time = pd.read_csv('./test_time.csv', header=None).squeeze()
    #建立坐标系
    #横坐标
    xAccuracy=range(len(accuracy))
    xLoss=range(len(loss))
    xCost=range(len(cost))
    xTime=range(len(time))
    #纵坐标
    yAccuracy=accuracy.reindex(xAccuracy,fill_value=None)
    yLoss=loss.reindex(xLoss,fill_value=None)
    yCost=cost.reindex(xCost,fill_value=None)
    yTime=time.reindex(xTime,fill_value=None)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(xAccuracy, yAccuracy, label='accuracy', color='#1f77b4', linestyle='-', linewidth=1.5)
    plt.xlabel('comRounds', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.title('accuracy', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('accuracy.png', dpi=300)
    plt.legend(loc='lower right')
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(xLoss, yLoss, label='loss', color='#1f77b4', linestyle='-', linewidth=1.5)
    plt.xlabel('comRounds', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title('loss', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('loss.png', dpi=300)
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(xCost, yCost, label='cost', color='#1f77b4', linestyle='-', linewidth=1.5)
    plt.xlabel('comRounds', fontsize=12)
    plt.ylabel('cost', fontsize=12)
    plt.title('cost', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('cost.png', dpi=300)
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(xTime, yTime, label='time', color='#1f77b4', linestyle='-', linewidth=1.5)
    plt.xlabel('comRounds', fontsize=12)
    plt.ylabel('time', fontsize=12)
    plt.title('time', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('time.png', dpi=300)
    plt.legend(loc='lower right')
    plt.show()
if __name__ == '__main__':
    drawOneTest()