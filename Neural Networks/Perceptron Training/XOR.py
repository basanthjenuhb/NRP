import numpy as np
import random

def sigmoid(z, derivative=False):
	if derivative == True:
		return sigmoid(z) * sigmoid(1 - z)
	return 1.0 / (1.0 + np.exp(-z))

def costFunction(x, y, W1, W2, b1, b2):
	z1 = np.dot(x, W1.T) + b1
	x1 = sigmoid(z1)
	z2 = np.dot(x1, W2.T) + b2
	x2 = sigmoid(z2)
	error = x2 - y
	cost = 0.5 * sum(error ** 2)
	return z1, x1, z2, x2, error, cost

x = [ [0,0],[0,1],[1,0],[1,1]]
y = [ [a ^ b] for (a,b) in x ]

x , y = np.array(x) , np.array(y)
W1 , W2 = 2 * np.random.randn(4,2) - 1 , 2 * np.random.randn(1,4) - 1
b1 , b2 = random.random() , random.random() 
epochs , alpha = 100000 , 0.5

for i in range(epochs):
	z1 , x1 , z2 , x2 , error , cost = costFunction(x, y, W1, W2, b1, b2)
	if cost < 0.001:break
	print "Epoch",i,"Error:",cost

	delta2 = error * sigmoid(z2, derivative=True)
	dw2 = np.dot(delta2.T, x1)

	delta1 = np.dot(delta2,W2) * sigmoid(z1, derivative=True)
	dw1 = np.dot(delta1.T,x)

	W1, W2 = W1 - alpha * dw1, W2 - alpha * dw2

print "Target:",y.T
print "Output:",costFunction(x, y, W1, W2, b1, b2)[3].T
