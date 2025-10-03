# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_data= pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
test_data= pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")
Train_data=np.array(train_data)
Train_data=Train_data.T
X_train=Train_data[1:]
X_train=X_train/255
Y_train=Train_data[0]
test_data=np.array(test_data)
Y_train.size

def init_params():
    W1=np.random.randn(40,784)*0.01
    b1=np.zeros((40,1))
    W2=np.random.randn(10,40)*0.01
    b2=np.zeros((10,1))
    return W1,b1,W2,b2

def init_adam(W1, b1, W2, b2):
    # Initialize first moment vectors (m)
    mW1 = np.zeros(W1.shape)
    mb1 = np.zeros(b1.shape)
    mW2 = np.zeros(W2.shape)
    mb2 = np.zeros(b2.shape)
    
    # Initialize second moment vectors (v)
    vW1 = np.zeros(W1.shape)
    vb1 = np.zeros(b1.shape)
    vW2 = np.zeros(W2.shape)
    vb2 = np.zeros(b2.shape)
    
    return mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2

def relu(X):
    return np.maximum(0,X)
    
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_prop(X,W1,b1,W2,b2):
    Z1=W1.dot(X)+b1
    A1= relu(Z1)
    Z2=W2.dot(A1)+b2
    A2=softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot_y(Y):
    one_hot_y=np.zeros((Y.size,Y.max()+1))
    one_hot_y[np.arange(Y.size),Y]=1
    one_hot_y=one_hot_y.T
    return one_hot_y
def deriv_relu(Z):
    return Z>0

def back_prop(Z1,A1,W1,Z2,A2,W2,X,Y):
    m=Y.size
    one_hot_Y=one_hot_y(Y)
    dZ2= A2-one_hot_Y
    dW2=(1/m)*dZ2.dot(A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=W2.T.dot(dZ2)*deriv_relu(Z1)
    dW1=(1/m)*dZ1.dot(X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
    return dW1,db1,dW2,db2



def update_params_adam(W1, b1, W2, b2, dW1, db1, dW2, db2, mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2, t, alpha, beta1, beta2, epsilon):

    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    mb1 = beta1 * mb1 + (1 - beta1) * db1
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    mb2 = beta1 * mb2 + (1 - beta1) * db2
    
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1**2)
    vb1 = beta2 * vb1 + (1 - beta2) * (db1**2)
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2**2)
    vb2 = beta2 * vb2 + (1 - beta2) * (db2**2)
    
    mW1_corr = mW1 / (1 - beta1**t)
    mb1_corr = mb1 / (1 - beta1**t)
    mW2_corr = mW2 / (1 - beta1**t)
    mb2_corr = mb2 / (1 - beta1**t)
    
    vW1_corr = vW1 / (1 - beta2**t)
    vb1_corr = vb1 / (1 - beta2**t)
    vW2_corr = vW2 / (1 - beta2**t)
    vb2_corr = vb2 / (1 - beta2**t)
    

    W1 = W1 - alpha * mW1_corr / (np.sqrt(vW1_corr) + epsilon)
    b1 = b1 - alpha * mb1_corr / (np.sqrt(vb1_corr) + epsilon)
    W2 = W2 - alpha * mW2_corr / (np.sqrt(vW2_corr) + epsilon)
    b2 = b2 - alpha * mb2_corr / (np.sqrt(vb2_corr) + epsilon)
    
    return W1, b1, W2, b2, mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2

def prediction(A2):
    return np.argmax(A2,0)
def accuracy(predictions,Y):
    return np.sum(predictions==Y)/Y.size
def gradient_descent(X,Y,iterations,alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    W1,b1,W2,b2=init_params()
    mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2 = init_adam(W1, b1, W2, b2)
    for i in range(1,iterations+1):
        Z1,A1,Z2,A2=forward_prop(X,W1,b1,W2,b2)
        dW1,db1,dW2,db2=back_prop(Z1,A1,W1,Z2,A2,W2,X,Y)
        W1, b1, W2, b2, mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2 = update_params_adam(
            W1, b1, W2, b2, dW1, db1, dW2, db2, 
            mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2, 
            i, alpha, beta1, beta2, epsilon
        )
        
        if i%50==0:
            print("Iteration:",i)
            print("Accuracy:",accuracy(prediction(A2),Y))
    return W1,b1,W2,b2

W1,b1,W2,b2=gradient_descent(X_train,Y_train,801,0.01)
