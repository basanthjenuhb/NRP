import numpy as np, matplotlib.pyplot as plt
import random

def sigmoid(z, derivative=False):
	if derivative == True:
		return sigmoid(z) * sigmoid(1 - z)
	return 1.0 / (1.0 + np.exp(-z))

def costFunction(x, y, m, W1, W2, b1, b2):
	z1 = np.dot(x, W1.T) + b1
	x1 = sigmoid(z1)
	z2 = np.dot(x1, W2.T) + b2
	x2 = sigmoid(z2)
	error = x2 - y
	cost = 0.5 / m * sum(error ** 2)
	return z1, x1, z2, x2, error, cost

m = 100
x = [ [random.random(),random.random()] for i in range(m) ]
y = [ [int(round(a)) ^ int(round(b))] for (a,b) in x ]
# np.random.seed(10)
x , y = np.array(x) , np.array(y)
W1 , W2 = 2 * np.random.randn(4,2) - 1 , 2 * np.random.randn(1,4) - 1
b1 , b2 = 2 * np.random.randn(1,4) - 1 , random.random() 
epochs , alpha , costs = 300000 , 0.001 , []
k = 0
for i in range(epochs):
	z1 , x1 , z2 , x2 , error , cost = costFunction(x, y, m, W1, W2, b1, b2)
	if cost < 0.001:break
	# print "Epoch",i,"Error:",cost
	costs.append(cost)
	if i % 10000 == 0:
		print i,cost
	if i > 200000 and k == 0:
		alpha *= 100
		k = 1
	if i > 300000 and k == 1:
		alpha *= 100
		k = 1

	delta2 = error * sigmoid(z2, derivative=True)
	dw2 = np.dot(delta2.T, x1)

	delta1 = np.dot(delta2,W2) * sigmoid(z1, derivative=True)
	dw1 = np.dot(delta1.T,x)

	W1, W2 = W1 - alpha * dw1, W2 - alpha * dw2
	b1 , b2 = b1 - alpha * sum(delta1), b2 - alpha * sum(delta2)
	# print b1,b2,"h"

print "\nFinal Error:",cost
print "Target:",y.T
print "Output:",costFunction(x, y, m, W1, W2, b1, b2)[3].T
plt.plot([j for j in range(len(costs))],costs)
plt.xlabel('Iterations -->')
plt.ylabel('Error -->')
plt.show()
while 1:
	x = [ map(float,raw_input().split()) ]
	print int(round(costFunction(x, y, m, W1, W2, b1, b2)[3].T))
