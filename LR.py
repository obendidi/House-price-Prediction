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
					y:train_xy
	})

	y_pred = function(x, T.dot(x,w))
        
	validate_model=function(inputs =[],
				output=err,
				givens={
					x:validation_set,
					y:valide_y
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
make_int('MSZoning'),make_int('LotShape'),make_int('LandContour'),make_int('LotConfig'),make_int('Neighborhood'),make_int('Condition1')
make_int('BldgType'),make_int('HouseStyle'),make_int('RoofStyle'),make_int('Exterior1st'),make_int('Exterior1st'),make_int('MasVnrType'),make_int('BsmtQual'),make_int('Foundation'),make_int('ExterCond')
make_int('ExterQual'),make_int('BsmtExposure'),make_int('BsmtFinType1'),make_int('BsmtFinType2'),make_int('Heating'),make_int('HeatingQC'),make_int('CentralAir'),make_int('Electrical'),make_int('KitchenQual')
make_int('Functional'),make_int('FireplaceQu'),make_int('GarageType'),make_int('GarageFinish'),make_int('GarageQual'),make_int('GarageCond'),make_int('SaleType'),make_int('SaleCondition'),make_int('PavedDrive')
make_int('Exterior2nd')
fill_null('LotFrontage'),fill_null('MasVnrArea'),fill_null('GarageYrBlt'),fill_null('BsmtFinSF1'),fill_null('BsmtFinSF2'),fill_null('BsmtUnfSF'),fill_null('TotalBsmtSF'),fill_null('BsmtFullBath'),fill_null('BsmtHalfBath'),fill_null('GarageCars')
fill_null('GarageArea')
train_df.SalePrice=train_df.SalePrice.astype(float) 
train_y = train_df['SalePrice'].values
ids = test_df['Id'].values
train_df=train_df.drop(['Id','Street','RoofMatl','Alley','MiscFeature','Fence','PoolQC','Utilities','LandSlope','Condition2','BsmtCond','SalePrice'],axis=1)
test_df=test_df.drop(['Id','Street','RoofMatl','Alley','MiscFeature','Fence','PoolQC','Utilities','LandSlope','Condition2','BsmtCond'],axis=1)
train_data = train_df.values
test_data=test_df.values

for i in range(0,69):
    moy=train_data[:,i].mean()
    M=train_data[:,i].max()
    m=train_data[:,i].min()
    for j in range(0,1460):
        train_data[j,i]=(train_data[j,i]-moy)/(M-m)
    for j in range(0,1459):
        test_data[j,i]=(test_data[j,i]-moy)/(M-m)
train_set=train_data[0:876,:]
vlaidation_set=train_data[876:1168,:]
test_det=train_data[1168:1460,:]
moy_y=train_y.mean()
M_y=train_y.max()
m_y=train_y.min()
for j in range(0,1460):
    train_y[j]=(train_y[j]-moy_y)/(M_y-m_y)	       
train_xy=train_y[0:876,:]
vlaidate_y=validate_y[876:1168,:]
test_y=train_y[1168:1460,:]
LinearRegression()









