# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load California housing data, select three features as X and two target variables as Y, then split into train and test sets.
2. Standardize X and Y using StandardScaler for consistent scaling across features.
3. Initialize SGDRegressor and wrap it with MultiOutputRegressor to handle multiple targets.
4. Train the model on the standardized training data.
5. Predict on the test data, inverse-transform predictions, compute mean squared error, and print results.

## Program:


Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Bakkiyalakshmi E
RegisterNumber: 212223220012 

```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

df.info()

x=df.drop(columns=['AveOccup','HousingPrice'])
x.info()

y=df[['AveOccup','HousingPrice']]
x.info()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler_x =StandardScaler()
scaler_y =StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)

y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
print(y_pred)

mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ",mse)
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
# DF head:
![image](https://github.com/user-attachments/assets/81ddc403-d77a-49a8-a466-3ea4d8e8af22)

# df info:
![image](https://github.com/user-attachments/assets/bdcfb0a4-2234-4ebe-a401-e93a7d835135)

# x.info:
![image](https://github.com/user-attachments/assets/67290d5c-786f-451e-89f8-d7c4c16c5013)

# y.info:
![image](https://github.com/user-attachments/assets/41628fd7-f7e7-4b4b-b38d-c66521ab9027)

# MultioutputRegressor:
![image](https://github.com/user-attachments/assets/08dcb458-35fc-4277-9c35-e3795602ec24)

# y_pred:
![image](https://github.com/user-attachments/assets/7fd4d4e4-5b75-485e-9d97-26b10a8756b5)

# Mean Squared Error:
![image](https://github.com/user-attachments/assets/c38590f0-2092-4829-82e0-f9a4a1e8bac9)

## Result:
Thus ,the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
