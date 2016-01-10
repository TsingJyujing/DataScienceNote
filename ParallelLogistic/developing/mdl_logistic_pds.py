# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:15:17 2016

@author: TsingJyujing
"""

import numpy as np
from ctypes import *
import threading

class logistic_pds(threading.Thread):
	def __init__(self,thread_info,DataX,DataY,W):
		threading.Thread.__init__(self)
		self.thread_info = thread_info + ": "
		self.X = DataX
		self.Y = DataY
		self.W = W
		self.N,self.dim = np.shape(self.X)
		self.clk = np.ctypeslib.load_library("logistic_kernel.dll",'.')
		self.clk.logistic_hessian_matrix_sum.argtypes = [
			POINTER(c_double),
			POINTER(c_double),
			c_uint,
			c_uint,
			POINTER(c_double)]
		self.clk.logistic_hessian_matrix_sum.restype = c_void_p
		
		self.clk.logistic_result.argtypes = [
			POINTER(c_double),
			POINTER(c_double),
			c_uint,
			c_uint,
			POINTER(c_double)]
		self.clk.logistic_result.restype = c_void_p
		
		self.clk.logistic_gradient.argtypes = [ 	
			POINTER(c_double),
			POINTER(c_double),
			POINTER(c_double),
			c_uint,
			c_uint,
			POINTER(c_double)]
		self.clk.logistic_gradient.restype = c_void_p
		
	def work(self):
		Res = np.zeros([self.N,1],dtype="float64")
		self.clk.logistic_gradient(
			self.X.ctypes.data_as(POINTER(c_double)),
			self.W.ctypes.data_as(POINTER(c_double)),
			self.dim,self.N,
			Res.ctypes.data_as(POINTER(c_double))	)
		return Res
		
	def hessian_matrix(self):
		HM = np.zeros([self.dim,self.dim],dtype="float64")
		self.clk.logistic_hessian_matrix_sum(
			self.X.ctypes.data_as(POINTER(c_double)),
			self.W.ctypes.data_as(POINTER(c_double)),
			self.dim,self.N,
			HM.ctypes.data_as(POINTER(c_double))	)
		return HM/self.N
		
	def partial_diff(self):
		diff = np.zeros([1,self.dim],dtype="float64")
		self.clk.logistic_gradient(
			self.X.ctypes.data_as(POINTER(c_double)),
			self.Y.ctypes.data_as(POINTER(c_double)),
			self.W.ctypes.data_as(POINTER(c_double)),
			self.dim,self.N,
			diff.ctypes.data_as(POINTER(c_double))	)
		return diff/self.N
			
	def __run__(self):
		#TRAIN
		return self.partial_diff()

def unit_test():
	Y = np.load('trainset_Y.npy')
	X = np.load('trainset_X.npy')
	W = np.array([2.,-3.])
	lpds = logistic_pds("THREAD_1",X,Y,W)
	print "Started"
	print lpds.partial_diff()
	
if __name__ == '__main__':
	print "import mdl_logistic_pds.logistic_pds as lp"
	unit_test()