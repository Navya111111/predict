import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv('C:\\Users\\user\\Downloads\\amazon.csv')

x=df.drop(['Store_ID','Product_ID','Units_Sold_Next_Week'],axis=1)
y=df['Units_Sold_Next_Week']

cf=['Product_Category','Store_Location','Day_of_Week']

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),cf)],remainder='passthrough')

x_encoded=ct.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=0.20,random_state=42)

model=DecisionTreeRegressor()

model.fit(x_train,y_train)



# Save model and preprocessor together
with open('sales_model.pkl', 'wb') as f:
    pickle.dump((model, ct), f)