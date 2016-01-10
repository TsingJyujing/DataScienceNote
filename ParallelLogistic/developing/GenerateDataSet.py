# -*- coding: utf-8 -*-
"""
生成训练数据集
"""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	N1 = 1000000;
	N0 = 1000000;
	plt.figure(num=1)
	x = np.random.normal(-8,3,[N0,1])
	y = np.random.normal(-8,2,[N0,1])+0.3*x
	label0 = np.zeros([N0,1])
	u0 = np.concatenate((x,y),1)
	plt.plot(x,y,'r.')
	x = np.random.normal(3,3,[N1,1])
	y = np.random.normal(-3,3,[N1,1])-x
	u1 = np.concatenate((x,y),1)
	label1 = np.ones([N1,1],dtype="float64")
	plt.plot(x,y,'bo')
	X = np.concatenate((u0,u1))
	Y = np.concatenate((label0,label1))
	np.save('trainset_X',X)
	np.save('trainset_Y',Y)
	