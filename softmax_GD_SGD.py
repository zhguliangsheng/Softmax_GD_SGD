# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:44:00 2021

""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import accuracy_score
 
# 数据处理，x增加默认值为1的b偏量，y处理为onehot编码类型
def data_convert(x,y):
    b=np.ones(y.shape)   # 添加全1列向量代表b偏量
    x_b=np.column_stack((b,x)) # b与x矩阵拼接
    K=len(np.unique(y.tolist())) # 判断y中有几个分类
    eyes_mat=np.eye(K)           # 按分类数生成对角线为1的单位阵
    y_onehot=np.zeros((y.shape[0],K)) # 初始化y的onehot编码矩阵
    for i in range(0,y.shape[0]):
        y_onehot[i]=eyes_mat[y[i]]  # 根据每行y值，更新onehot编码矩阵
    return x_b,y,y_onehot
 
# softmax函数，将线性回归值转化为概率的激活函数。输入s要是行向量
def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=1)
 

#梯度下降算法（GD) 
def SoftmaxGD(x,y,alpha=0.05,max_loop=100):
    m=np.shape(x)[1]  # x的特征数
    n=np.shape(y)[1]  # y的分类数
    #weights=np.ones((m,n)) # 权重矩阵
    weights=np.random.randn(m,n) #初始权重为randn函数返回一个或一组样本，具有标准正态分布
    
    for k in range(max_loop):
        # k=2
        h=softmax(x*weights)        
        error=y-h
        weights=weights+alpha*x.transpose()*error # 梯度下降算法公式
    return weights.getA()

# 随机梯度下降算法(SGD) 
def SoftmaxSGD(x,y,alpha=0.05,max_loop=100):
    m=np.shape(x)[1]
    n=np.shape(y)[1]
    #weights=np.ones((m,n))
    weights=np.random.randn(m,n) #初始权重为randn函数返回一个或一组样本，具有标准正态分布
    for k in range(max_loop):
        for i  in range(0,len(x)):
            h=softmax(x[i]*weights)
            error=y[i]-h[0]
            weights=weights+alpha*x[i].T*error[0]  # 随机梯度下降算法公式
    return weights.getA()

 
# 对新对象进行预测
def predict(weights,testdata):
    y_hat=softmax(testdata*weights)
    predicted=y_hat.argmax(axis=1).getA()
    return predicted


# 多分类绘制分界区域。而不是通过分割线来可视化
def plotBestFit(dataMat,labelMat,weights):
 
    # 获取数据边界值，也就属性的取值范围。
    x1_min, x1_max = dataMat[:, 1].min() - .5, dataMat[:, 1].max() + .5
    x2_min, x2_max = dataMat[:, 2].min() - .5, dataMat[:, 2].max() + .5
    # 产生x1和x2取值范围上的网格点，并预测每个网格点上的值。
    step = 0.002
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    testMat = np.c_[xx1.ravel(), xx2.ravel()]   #形成测试特征数据集
    testMat = np.column_stack((np.ones(((testMat.shape[0]),1)),testMat))  #添加第一列为全1代表b偏量
    testMat = np.mat(testMat)
    # 预测网格点上的值
    y = softmax(testMat*weights)   #输出每个样本属于每个分类的概率
    # 判断所属的分类
    predicted = y.argmax(axis=1)                            #获取每行最大值的位置，位置索引就是分类
    predicted = predicted.reshape(xx1.shape).getA()
    # 绘制区域网格图
    plt.pcolormesh(xx1, xx2, predicted, cmap=plt.cm.Paired)
 
    # 再绘制一遍样本点，方便对比查看
    plt.scatter(dataMat[:, 1].flatten().A[0], dataMat[:, 2].flatten().A[0],
                c=labelMat.flatten().A[0],alpha=.5)  # 第一个偏量为b，第2个偏量x1，第3个偏量x2
    plt.xticks([]) 
    plt.yticks([])
    plt.title('Softmax')
    plt.show()

    
if __name__ == "__main__":
    df = pd.read_csv("ex.txt",header=None)
    x = np.array(df[[0,1]])
    y = np.array(df[2])   
    
    #0-1标准化  
    x=MinMaxScaler().fit_transform(x)
    # 转换数据为matrix类型
    x = np.mat(x)
    y = np.mat(y).T
    
    # 调用数据预处理函数
    X,Y,Y_onehot = data_convert(x,y)
    
    # 梯度下降算法
    weights1=SoftmaxGD(X, Y_onehot)    
    print('梯度下降算法')
    print("权重：")
    print(weights1)
    y_hat1 = predict(weights1,X)
    print("准确率：%f" %(accuracy_score(y, y_hat1)))
    plotBestFit(X,Y,weights1)
    
    
    
    # 随机批量梯度下降算法
    weights2 = SoftmaxSGD(X, Y_onehot)
    print('随机梯度下降算法')
    print("权重：")
    print(weights2)
    y_hat2 = predict(weights2,X)
    print("准确率：%f" %(accuracy_score(y, y_hat2)))  
    plotBestFit(X,Y,weights2)

