import math
bias = 0.0
def sigmoid(x):
	return 1.0 / ( 1.0 + math.exp(-x))

def hypothesis(x , weights):
	return sigmoid(x[0] * weights[0] + x[1] * weights[1] + bias)

def error(x , y , weights):
	return ( hypothesis(x , weights) - y )

def cost_function(x , y , weights):
	cost_sum = 0.0
	for i in range(len(x)):
		cost_sum += error(x[i], y[i] , weights)
	return 1.0 / (2.0 * (len(x))) * cost_sum

def calculate_delta(x , y , weights):
	delta = [0 ] * len(weights)
	for i in range(len(delta)):
		for j in range(len(x)):
			delta[i] += error(x[j] , y[j] , weights) * x[j][i]
	delta = [ val / len(x) for val in delta ]
	return delta

x = [[0,0],[0,1],[1,0],[1,1]]
y = [ a | b for (a,b) in x]
alpha , k = 0.01 , 1
weights = [0,0]
while abs(cost_function(x , y , weights)) > 0.001:
	delta = calculate_delta(x , y , weights)
	weights = [ weights[i] - alpha * delta[i] for i in range(len(delta)) ]
	bias = bias + alpha * cost_function(x , y , weights)
	print "Iteration:",k ,"Weights: ", weights, "Bias:", bias
	for i in range(len(x)):
		print x[i][0] , x[i][1] ,"Expected:", x[i][0] | x[i][1] ,"Calculated:", hypothesis(x[i] , weights)
	k+=1
