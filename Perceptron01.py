#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chapter01.Perceptron import Perceptron
from matplotlib.colors import ListedColormap

#数据下载网址：https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
def get_flowers_feature():
    '''
       iris.data中一共包含了150条数据，包含了三种鸢尾花
       这里我们一共使用100条数据，山鸢尾(setosa)和变色鸢尾(versicolor)
       每一组鸢尾花数据都包含了四个特征
       为了方便绘图只挑选其中的两个特征(花瓣的长度和萼片的长度)
       '''
    # 通过pandas读取鸢尾花数据,一定要加header=None否则不会包括第一行数据
    df = pd.read_csv("iris.data",header=None)
    #获取前100组鸢尾花数据以及对应的标签
    flowers_name = df.iloc[:100,4].values
    #将花的名字转换为标签，setosa为-1，versicolor为1
    flowers_label = np.where(flowers_name=="Iris-setosa",-1,1)
    #选取前100组鸢尾花的第一个()和第二个特征
    flowers_feature = df.iloc[0:100,[0,2]].values
    return flowers_feature,flowers_label

def plot_flowers_distribute():

    flowers_feature,flowers_label = get_flowers_feature()
    #根据山鸢尾的两个特征进行绘图
    plt.scatter(flowers_feature[:50,0],flowers_feature[:50,1],
                color="red",marker="o",label=u"山鸢尾")
    #根据变色鸢尾的两个特征进行绘图
    plt.scatter(flowers_feature[50:100,0],flowers_feature[50:100,1],
                color="blue",marker="x",label=u"变色鸢尾")
    #设置x轴的标签
    plt.xlabel(u"花瓣长度(cm)")
    #设置y轴的标签
    plt.ylabel(u"萼片长度(cm)")
    #设置显示label的位置
    plt.legend(loc="upper left")
    plt.show()

def iter_errors_num():
    #初始化感知器，设置感知器的学习率和迭代的次数
    perceptron = Perceptron(eta=0.1,n_iter=10)
    #获取花的特征和标签
    x,y = get_flowers_feature()
    #训练
    perceptron.fit(x,y)
    plt.plot(range(1,len(perceptron.errors_)+1),perceptron.errors_,marker="o")
    plt.xlabel("迭代次数")
    plt.ylabel("错误分类样本数量")
    plt.show()

def plot_decision_regions(resolution=0.02):
    #定义标记符
    markers = ('s','x','o','^','v')
    #定义颜色
    colors = ('red','blue','lightgreen','gray','cyan')
    #获取花的特征和标签
    x,y = get_flowers_feature()
    perceptron = Perceptron(eta=0.1, n_iter=10)
    perceptron.fit(x,y)
    #np.unique(y)方法获取y中不重复的元素，也就只有-1和1
    #ListedColormap方法是将标记符和颜色进行对应
    #在绘图的时候红色表示正方形而蓝色表示叉
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #获取第一个特征中最大值加1和最小值减1
    x1_min,x1_max = x[:,0].min() - 1,x[:,0].max() + 1
    #获取第二个特征中最大值加1和最小值减1
    x2_min,x2_max = x[:,1].min() - 1,x[:,1].max() + 1
    #根据特上面获取到特征的最大最小值构建一个网格坐标
    #通过模拟足够多的鸢尾花数据，来绘制出决策边界
    #resolution表示网格的大小
    '''
    如一个2*2的网格坐标,网格大小为1，网格坐标点如下
    (0,0),(0,1),(0,2)
    (1,0),(1,1),(1,2)
    (2,0),(2,1),(2,2)
    '''
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))
    z = perceptron.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    #绘制边界
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
    #设置坐标的长度
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx,cl in enumerate(np.unique(y)):
        #在图上根据鸢尾花的特征进行绘点
        plt.scatter(x=x[y == cl,0],y=x[y == cl,1],
                    alpha=0.8,c=cmap(idx),
                    marker=markers[idx],label=cl)
    plt.xlabel("花瓣长度(cm)")
    plt.ylabel("萼片长度(cm)")
    plt.show()

if __name__ == "__main__":
    # plot_flowers_distribute()
    # iter_errors_num()
    plot_decision_regions()





