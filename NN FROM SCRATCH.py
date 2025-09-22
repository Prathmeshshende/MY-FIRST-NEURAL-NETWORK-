import numpy as np
import random
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # activation function
def sigmoid_derivative(x):
    return x * (1 - x) # derivative of activation function
x=np.array([[0,0],[0,1],[1,0],[1,1]]) # input dataset
y=np.array([[0],[0],[0],[1]]) # output dataset
np.random.seed(1) # seed for reproducibility
weights=np.random.rand(2,1) # initialize weights randomly with mean 0
bias=np.random.rand(1) # initialize bias randomly
 
learning_rate=0.1
epochs=10000
for epoch in range(epochs):
    # Forward propagation
    z=np.dot(x,weights)+bias
    output=sigmoid(z)

    # Calculate the error
    error=y-output

    # Backpropagation
    adjustments=error*sigmoid_derivative(output)
    weights+=np.dot(x.T,adjustments)*learning_rate
    bias+=np.sum(adjustments)*learning_rate
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')