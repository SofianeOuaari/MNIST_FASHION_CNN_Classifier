import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPool2D,Activation,Dropout


df_train=pd.read_csv("fashion-mnist_train.csv") 
df_train.head(15)
 
 label_name=["T-shirt/Top","Trouser","PullOver","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
df_train_img=df_train.drop("label",axis=1)
df_train_label=df_train["label"]

df_test=pd.read_csv("fashion-mnist_test.csv")
df_test_img=df_test.drop("label",axis=1) 
df_test_label=df_test["label"] 

def create_data(df_img,df_label):
    data=[]
    for i in range(len(df_img)):
        img_arr=np.array(df_img.iloc[i]).reshape(28,28)
        label=df_label.iloc[i] 
        data.append([img_arr,label])
    x=[]
    y=[]
    for img,label in data:
        x.append(img) 
        y.append(label) 
    x=np.array(x)
    x=x.reshape(x.shape[0],1,28,28)  
    x=x/255
    y=np.array(y) 
    return x,y
#Creating our training and testing data
x_train,y_train=create_data(df_train_img,df_train_label) 
x_test,y_test=create_data(df_test_img,df_test_label)



#Let's Build now the Model's Architecture 
model=Sequential()
for _ in range(5):
    model.add(Conv2D(64,(3,3),input_shape=x_train.shape[1:],data_format="channels_first")) 
    model.add(Activation("relu")) 
    model.add(MaxPool2D(pool_size=(1,1)))
    
model.add(Flatten())

for _ in range(0):
    model.add(Dense(64))
    model.add(Activation("relu")) 

model.add(Dense(10)) 
model.add(Activation("softmax")) 
    
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"]) 
model.fit(x_train,y_train,batch_size=32,epochs=3,validation_data=(x_test,y_test))