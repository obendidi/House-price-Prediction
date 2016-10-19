import numpy as np
import pandas as pd
import theano
from theano import function
from theano import pp
from theano import tensor as T
from theano.printing import min_informative_str
import matplotlib.pyplot as plt
import csv as csv
from sklearn.preprocessing import PolynomialFeatures


def make_int(opt):
    if len(train_df[opt][ train_df[opt].isnull() ]) > 0:
        train_df[opt][ train_df[opt].isnull() ] = train_df[opt].dropna().mode().values
    if len(test_df[opt][ test_df[opt].isnull() ]) > 0:
        test_df[opt][ test_df[opt].isnull() ] = test_df[opt].dropna().mode().values
    Ports = list(enumerate(np.unique(train_df[opt])))   
    Ports_dict = { name : i for i, name in Ports }              
    train_df[opt] = train_df[opt].map( lambda x: Ports_dict[x])     
    test_df[opt] = test_df[opt].map( lambda x: Ports_dict[x])

def fill_null(opt):
    median = train_df[opt].dropna().median()
    if len(train_df[opt][ train_df[opt].isnull() ]) > 0:
        train_df.loc[ (train_df[opt].isnull()), opt] = median
    if len(test_df[opt][ test_df[opt].isnull() ]) > 0:
        test_df.loc[ (test_df[opt].isnull()), opt] = median



train_df=pd.read_csv('train.csv',header=0)
test_df=pd.read_csv('test.csv',header=0)
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
learning_rate = 0.1
n_dims = 2485
n_steps = 100
lam = 1
poly = PolynomialFeatures(degree=2)
train = poly.fit_transform(train_data)
test = poly.fit_transform(test_data)
x = T.matrix(name='x')  # Input matrix with examples.
w = theano.shared(value=np.zeros((n_dims, 1),  # Parameters of the model.
        dtype=theano.config.floatX),
        name='w', borrow=True)
f = function([x], T.dot(x, w))  # Linear regression.

# Define objective function.
y = T.vector(name='y')  # Output vector with y values.
r=1460
# Define loss function.
loss = T.mean((T.dot(x, w).T - y) ** 2)+lam*T.sum(w**2)

# Build the gradient descent algorithm.
g_loss = T.grad(loss, wrt=w)

train_model = function(inputs=[],
                       outputs=loss,
                       updates=[(w, w*(1-lam/r) - learning_rate * g_loss)],
                       givens={
                           x: train,
                           y: train_y
})
for i in range(n_steps):
    print "cost", train_model()
output=[]
for xx in test:
    output.append(float((f([xx]))*(M_y-m_y))+moy_y)
output=np.asarray(output)
predictions_file = open("firsttest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","SalePrice"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
