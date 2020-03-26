#!/usr/bin/env python
# coding: utf-8

# # Prakrati Data Analytics
# 
# # Team Name -  WAKANDA_FOR3V3R
# 
# ### Ayush Kumar              9661179290    17CS10007    IIT KHARAGPUR
# ### Anshul Choudhary          8918500798    17CS10005    IIT KHARAGPUR
# ### Prakhar Bindal          9593801201     17CS10036  IIT KHARAGPUR
# ### Shubham Raj                9733331647     17CE10054   IIT KHARAGPUR
# 

# # Importing necessary libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_selection import SelectFromModel


# # Read data from file

# In[2]:


df = pd.read_excel('Data_set.xlsx')
df=df.drop([81, 114, 116])
df=df.iloc[:,1:]
#df


# # Extract necessary data in to seperate variables

# In[3]:


column_names=df.columns
elemental_names=column_names[2:13]
spectral_names=column_names[13:]

temp_data=df.to_numpy()
TC=temp_data[:,0]
TN=temp_data[:,1]
Y=temp_data[:,0:2]


# # Normalisation using MinMaxScalar

# In[4]:


min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1))
data = min_max_scaler.fit_transform(df)

# extract spectral, elemental and Combined data
spectral_data=data[:,13:]
elemental_data=data[:,2:13]
X=data[:,2:]

#print (np.shape(elemental_data),np.shape(Y),np.shape(spectral_data),np.shape(elemental_data),elemental_names.shape,spectral_names.shape)


# # Model (Random Forest Regressor)

# In[5]:


def Random_Forest(X, Y):
    # scale input data before model training
    X=preprocessing.scale(X)
    
    # split examples into test and training randomly in 2:8 ratio
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.20, random_state = 6)
    
    # create and fit the random forest regressor with 100 decision trees
    regr = RandomForestRegressor(n_estimators=500)
    tree = regr.fit(X_Train,Y_Train)
    
    # using above trained model to extract influential variables
    model = SelectFromModel(tree, prefit=True)
    X_new = model.transform(X)
    
    print('\nNo. of influential variables - ',np.size(X_new,1),' out of ',np.size(X,1))
    
    imp_features_labels=[]
    for j in range(np.size(X_new,1)):
        for i in range(np.size(X,1)):
            f=1
            for k in range(len(X)):
                if X_new[k][j]!=X[k][i]:
                    f=0
                    break
            if f==1:
                if(np.size(X,1)<20):
                    imp_features_labels.append(elemental_names[i])
                elif(np.size(X,1)<2150):
                    imp_features_labels.append(spectral_names[i])
                else:
                    imp_features_labels.append(column_names[i+2])
                break
                
    print('\nInfluential Features are : ', imp_features_labels)
    
    # determining correlation between influential dependent variables and independent variables
    t=pd.DataFrame(X_new)
    t['Y']=Y
    t2=pd.DataFrame(data=t.to_numpy())
    t2=t2.astype('float64')
    corrmat = t2.corr()
    
    feature_corr=pd.DataFrame()
    feature_corr["feature"]=imp_features_labels
    feature_corr["corr"]=corrmat.iloc[:-1,-1]
    print('\nCorrelation between Infuential Independent variables and Dependent variables : \n', feature_corr)
    
    # Testing the trained model on test data
    Y_Pred=regr.predict(X_Test)
    RMSE=sqrt(mean_squared_error(Y_Pred,Y_Test))
    
    print('\nRoot Mean Sq Error - ', RMSE)
    print('Train Accuracy - ', regr.score(X_Train,Y_Train))
    print('Test Accuracy - ', regr.score(X_Test,Y_Test),'\n')


# # Providing Model with different sets of data

# In[7]:


print('############### Predicting TC using Spectral Data ################\n')
Random_Forest(spectral_data,TC)
print('\n\n\n\n\n############### Predicting TN using Spectral Data ################\n')
Random_Forest(spectral_data,TN)
print('\n\n\n\n\n############### Predicting TC using Elemental Data ################\n')
Random_Forest(elemental_data,TC)
print('\n\n\n\n\n############### Predicting TN using Elemental Data ################\n')
Random_Forest(elemental_data,TN)
print('\n\n\n\n\n############### Predicting TC using combination of Spectral and Elemental Data ################\n')
Random_Forest(X,TC)
print('\n\n\n\n\n############### Predicting TC using combination of Spectral and Elemental Data ################\n')
Random_Forest(X,TN)


# In[ ]:




