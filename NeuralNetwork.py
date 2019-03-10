

#
# This is a basic 2 layer neural network, and is my first proper foray
# into the world of Machine Learning. I have no idea what I want to do 
# with it yet, but we'll see how things go.
#

import numpy as np
	
class NeuralLayer:
        # This is a class representing a single neural layer
        def __init__(self,input_length,neuron_num):
                # Initialize neural layer 
                self.weights = np.random.rand(input_length,neuron_num)
                self.bias = np.random.rand(1,neuron_num)
                self.layer_mat = np.vstack((self.weights,self.bias))

        def __repr__(self):
                return 'Gotta figure out a good way to print out the weights & biases'
        
        def sigmoid(self,x):
                #
                # The sigmoid function is typically the activation function utilized
                # in neurons for neural networks.
                #
                # Basic properties:
                #     - Linear map from (-inf, inf) to [0,1]
                #
                return 1/(1+np.exp(-x))

        def feedforward(self,inputs):
                # The actual work of taking in the inputs, then computing and propagating
                # the output to the next layer
                inputs = np.hstack((np.ravel(inputs),np.array([1])))    # ravel is used for future-proofing
                total = np.matmul(inputs,self.layer_mat)                # e.g. CNNs
                return self.sigmoid(total)
		
n = NeuralLayer(2,2)

x= np.array([2,3])
print(n.feedforward(x))
