from sklearn import datasets
import matplotlib.pyplot as plt , numpy as np

def sigmoid(z, derivative=False):
	if derivative == True:
		return sigmoid(z) * sigmoid(1 - z)
	return 1.0 / (1.0 + np.exp(-z))

def costFunction(x, y, m, W1, W2, W3, b1, b2, b3):
	z1 = np.dot(x, W1.T) + b1
	x1 = sigmoid(z1)
	z2 = np.dot(x1, W2.T) + b2
	x2 = sigmoid(z2)
	z3 = np.dot(x2, W3.T) + b3
	x3 = sigmoid(z3)
	error = x3 - y
	cost = 0.5 * sum(error ** 2)
	return z1, x1, z2, x2, z3, x3, error, cost

#Loading the datasets
digits = datasets.load_digits()

#Preprocessing
x, Y = (digits.data > 0).astype(np.int32,copy=False), np.array(digits.target)
y = np.zeros((len(x),10))
y[np.arange(len(Y)),Y] = 1

#Network variables
h1, h2, m = 64, 64, x.shape[0]
W1 = 2 * np.random.randn(h1,x.shape[1]) - 1
W2 = 2 * np.random.randn(h2,h1) - 1
W3 = 2 * np.random.randn(y.shape[1],h2) - 1
b1 = 2 * np.random.randn(1,h1) - 1
b2 = 2 * np.random.randn(1,h2) - 1
b3 = 2 * np.random.randn(1,10) - 1
epochs , alpha , costs = 3000 , 0.0001 , []
print W1.shape,W2.shape
k = 0
for i in range(epochs):
	z1 , x1 , z2 , x2, z3, x3, error , cost = costFunction(x, y, m, W1, W2, W3, b1, b2, b3)
	print "Epoch",i,"Error:",sum(cost)
	delta3 = error * sigmoid(z3, derivative=True)
	dw3 = np.dot(delta3.T, x2)

	delta2 = np.dot(delta3,W3) * sigmoid(z2, derivative=True)
	dw2 = np.dot(delta2.T,x1)

	delta1 = np.dot(delta2,W2) * sigmoid(z1, derivative=True)
	dw1 = np.dot(delta1.T,x)

	W1, W2, W3 = W1 - alpha * dw1, W2 - alpha * dw2 , W3 - alpha * dw3
	b1 , b2, b3 = b1 - alpha * sum(delta1), b2 - alpha * sum(delta2), b3 - alpha * sum(delta3)
	# print sum(delta1).shape,sum(delta2).shape
	# print sum(delta1),sum(delta2)
res = np.argmax(costFunction(x, y, m, W1, W2, W3, b1, b2, b3)[3],axis=1)
count = 0
for i in range(len(Y)):
	if res[i] == Y[i]:
		count += 1
print count ,"/",x.shape[0]
# print "\nFinal Error:",cost
# print "Target:",y.T
# print "Output:",costFunction(x, y, m, W1, W2, b1, b2)[3].T
# plt.plot([j for j in range(len(costs))],costs)
# plt.xlabel('Iterations -->')
# plt.ylabel('Error -->')
# plt.show()

# i = 100
# print(y[i])
# plt.imshow(images[i],cmap='gray')
# plt.show()
