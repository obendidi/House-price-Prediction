# House-price-Prediction
Using a dataset got from  kaggle to try and create a linear model to predict the sale prices of the houses using 79 carefully selected house feature 


The dataset can be downloaded from the Kaggle website in house price prediction compettion.
Used Theano and numpy to create a linear regression model with regularization to prevent overfitting 

Used feature engeneering to add new variables that correlate strongly with the sale price 

Language used :Python 
Librairies: Theano , Scikit-learn-preprocessing

##Scripts:
==========
1.py : Contains the main linear regression model , depends on the learning-rate(alpha) and the regularization parametre(lambda)
       to create the model.
       
       
LR.py: uses the same algorithme as '1.py' but iterated over a number of learning-rates between(10 and 0.001) and reagularization-parametres between (10 and 0.001)
to find the best combination of alpha and lambda to minimize the cost function 

The csv File contains the predicted sale price of the test-set

Got a score of 0.16786 on the kaggle competition leader board (evaluated on Root-Mean-Squared-Error)
