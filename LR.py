from __future__ import print_function
__docformat__ = 'restructedtext en'
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import csv as csv
import pandas as pd
import theano
import theano.tensor as T
import os
import sys




def LinearRegression():
	n_dims=2485
	
	
	
	x=Tmatrix(name='x')

	w = theano.shared(value=np.zeros((n_dims,1),
				dtype=theano.config.floatX),
				name='w',borrow=True)
	y = T.vector(name='y')
	
	err= T.mean((T.dot(x,w).T- y)**2)

	loss = T.mean((T.dot(x,w).T - y) **2)+lam*T.mean(w**2)
	
	g_loss = T.grad(loss,wrt=w)
	


	print('building Linear Regression model ...')
	
	r=1460
	train_model = function(inputs=[learning_rate,lam],
				output=loss,
				updates=[(w,w*(1-lam/r) - learning_rate * g_loss)],
				givens={
					x:train_set,
					y:train_y
	})

	y_pred = function(x, T.dot(x,w))
        
	validate_model=function(inputs =[],
				output=err,
				givens={
					x:valid_set,
					y:valid_y
	})
	test_model=function(inputs =[],
				output=err,
				givens={
					x:test_set,
					y:test_y
	})

	print('training the model ...')
	li=[100,50,10,5,1,0.6,0.3,0.1,0.06,0.03,0.01,0.006,0.001]
	lamb=[100,50,10,5,10.5,0.1]
	for xx in lamb:
		for yy in li:
			print('Learning rate = {}, and lambda = {}'.format(yy,xx))
			for i in range(1000):
				print("training error : ",train_model(yy,xx))
			
			print('new model validation error : ',validate_model())
			
	print('best model test error : ',test_model())

def make_int(opt):
	if len(train_df[opt][train_df[opt].isnull()]) > 0:
		train_df[opt][train_df[opt].isnull()]=train_df[opt].dropna().mode().values
	if len test_df[opt][test_df[opt].isnull()])>0:
		test_df[opt][test_df[opt].isnull()]=test_df[opt].dropna().mode.values

	Ports=list(enumerate(np.unique(train_df[opt]))
	Port_dict = {name:i for i,name in Ports}
	train_df[opt] =train_df[opt].map(lambda x:Ports_dict[x])
	test_df[opt] = test_df[opt].map(lambda x: Ports_dict[x])

def fill_null(opt):
	median=train_df[opt].dropna().median()
	if len(train_df[opt][train_df[opt].isnull()]) >0:
		train_df.loc[(train_df[opt].isnull(),opt] =median
	if len(test_df[opt][test_df[opt].isnull()),opt]=median


train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')












