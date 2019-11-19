# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 06:44:43 2019

@author: raona
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 08:00:20 2019

@author: raona
"""

import random

# Third-party libraries
import numpy as np
from Flowers_load import load_data
from Flowers_load import vectorized_result
import pandas as pd
import matplotlib.pyplot as plt
#from Flowers_load import load_data_wrapper
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)
class FloRec(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
    
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
#            print(np.shape(a), "feedforward")
#            print(np.shape(w), "w feedforward")
#            print(np.shape(b), "b feedforward")
#            print(np.shape(a), "inp feedforward")
            a = sigmoid(np.dot(w, a.transpose())+b)
            a = a.transpose()
#            print(np.shape(a), "feedforward")
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
            test_data=None):
        Graph_test = []
        accu_cal = []
        training_cost = []
#        print(np.shape(training_data))
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        random.shuffle(training_data)
#        print(n, 'my n is ')]]]]]
        for j in range(epochs):
#            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
###            print(np.shape(mini_batches))
            for mini_batch in mini_batches:
###                print("epoch"+str(j))
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
##                
##                print("completed update mini batch")
            if test_data:
                print("entered into test data")
#                print(self.evaluate(test_data))
                Cal = float((self.evaluate(test_data))/n_test)
                print("Epoch {0}: {1} / {2} = {3}".format(
                    j, self.evaluate(test_data), n_test, Cal))
            else:
                print("Epoch {0} complete".format(j))
                
            cost = self.cost_fun(training_data, lmbda)
            training_cost.append(cost)
            print ("Cost on training data: {}".format(cost))
#            
            Graph_test.append(j)
            accu_cal.append(Cal)
#       plotting the graph for accurancy 
        plt.plot(Graph_test, accu_cal)
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.title('accurancy graph')
        plt.show()

# plotting the graph for cost funtion 
        plt.plot(Graph_test, training_cost)
        plt.xlabel('Epochs')
        plt.ylabel('training_cost')
        plt.title('Cost funtion graph')
        plt.show()
        

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
#        print(np.shape(mini_batch))
        for x, y in mini_batch:
#            print(np.shape(x) , "x is ")
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x.transpose()
#        print(np.shape(activation), "my activation ")
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
#            print(np.shape(w)," my w is" )
#            print(np.shape(activation))
            z = np.dot(w, activation)+b
#            print(w, "w is ")
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1])
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                 for (x, y) in test_data]
#        print(np.shape(test_results), "test result")
#        print(sum(int(x == y) for (x, y) in test_results))
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_fun(self, test_data, lmbda, convert=False):
        cost = 0.0
        for x, y in test_data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(test_data)
        cost += 0.5*(lmbda/len(test_data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    
if __name__ == "__main__":
    n = FloRec([256036,20,5])
    training_data, test_data = load_data()
    n.SGD(training_data, 100, 10, 0.1, test_data = test_data)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
