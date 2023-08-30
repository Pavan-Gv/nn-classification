# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![DL EX2](https://github.com/Pavan-Gv/nn-classification/assets/94827772/daa42f24-c03f-455b-ba8c-7ea267947e77)

## DESIGN STEPS

<b>STEP 1:</b> Import the required packages

<b>STEP 2:</b> Import the dataset to manipulate on

<b>STEP 3:</b> Clean the dataset and split to training and testing data

<b>STEP 4:</b> Create the Model and pass appropriate layer values according the input and output data

<b>STEP 5:</b> Compile and fit the model

<b>STEP 6:</b> Load the dataset into the model

<b>STEP 7:</b> Test the model by predicting and output

## PROGRAM
```
Developed by: G Venkata Pavan Kumar
Registrarion Number: 212221240013
```
### Importing the require packages:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
```

### Importing the dataset:
```
customer_df = pd.read_csv('/content/drive/MyDrive/customers.csv')
customer_df_cleaned=customer_df.dropna(axis=0)
```

### Data exploration:
```
customer_df.columns
customer_df.shape
customer_df.isnull().sum()
customer_df_cleaned = customer_df.dropna(axis=0)
customer_df_cleaned.isnull().sum()
customer_df_cleaned.shape
customer_df_cleaned.dtypes
customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Graduated'].unique()
customer_df_cleaned['Profession'].unique()
customer_df_cleaned['Spending_Score'].unique()
customer_df_cleaned['Var_1'].unique()
customer_df_cleaned['Segmentation'].unique()
categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)
customers_1 = customer_df_cleaned.copy()
customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])
customers_1.dtypes
```

### Encoding of output values:
```
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes
customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)
customers_1.dtypes
customers_1['Segmentation'].unique()
X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values
y1 = customers_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray()
```

### Spliting the data :
```
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
```

### Scaling the features of input :
```
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
```
### Creation of model :
```
ai_brain = Sequential([
  Dense(units = 4, input_shape=[8]),
  Dense(units =16, activation='relu'),
  Dense(units =4, activation ='softmax')
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=500,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
```
### Ploting the metrics :
```
metrics = pd.DataFrame(ai_brain.history.history)
early_stop = EarlyStopping(monitor='val_loss', patience=2)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
```
### Making the prediction :
```
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
```
### Saving and loading the model :
```
ai_brain.save('customer_classification_model.h5')
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

ai_brain = load_model('customer_classification_model.h5')
with open('customer_data.pickle', 'rb') as fh:
     [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)
```
### Making the prediction for single input :
```
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```

## Dataset Information :

![image](https://github.com/Pavan-Gv/nn-classification/assets/94827772/4baecbeb-fa80-45ad-9b3c-9ddc0d5c6a26)

## OUTPUT :

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Pavan-Gv/nn-classification/assets/94827772/c38f064e-d3f2-4d32-a78d-2737ddc343fe)

### Classification Report

![image](https://github.com/Pavan-Gv/nn-classification/assets/94827772/2fd58df7-d638-4906-9e5a-83e00caaa736)

### Confusion Matrix

![image](https://github.com/Pavan-Gv/nn-classification/assets/94827772/ed42b6e1-9b36-4973-a246-ff6510e62071)


### New Sample Data Prediction

![image](https://github.com/Pavan-Gv/nn-classification/assets/94827772/a7e66ade-0da3-48e8-a67d-9b5a0f1c9ca2)

## RESULT :

Therefore a Neural network classification model is developed and executed successfully.

