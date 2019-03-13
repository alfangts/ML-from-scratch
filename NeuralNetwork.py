

#
# This is a basic 2 layer neural network, and is my first proper foray
# into the world of Machine Learning. I have no idea what I want to do 
# with it yet, but we'll see how things go.
#

import numpy as np
	
class NeuralLayer:
        # This is a class representing a single neural layer
        def __init__(self,input_length,neuron_num):
                '''
                Initializes new neural layer
                
                - weights is a (input length x neuron number) numpy array,
                  with the columns corresponding to the weights of each neuron
                - bias is a (1 x neuron number) numpy array, with each column
                  corresponding to the bias of each neuron
                - layer_mat is a (input length+1 x neuron number) numpy array,
                  composed of the bias vector stacked horizontally below the
                  weights matrix

                  The layer_mat is composed as such to simplify operations through
                  the use of linear algebra. An input will be reshaped into
                  a row vector, with a {1} is appended to its end to make it 
                  (1 x n+1). The operation of feeding the data forward to the next
                  layer will simply be a matrix multiplication operation.
                '''
                
                self.weights = np.random.rand(input_length,neuron_num)
                self.bias = np.random.rand(1,neuron_num)
                self.layer_mat = np.vstack((self.weights,self.bias))

        def __repr__(self):
                return 'Gotta figure out a good way to print out the weights & biases'
        
        def sigmoid(self,x):
                '''
                  The sigmoid function is typically the activation function utilized
                  in neurons for neural networks. It is a linear map from (-inf, inf)
                  to [0,1]. 

                 Input:
                  - x is a (1 x n) numpy array

                 Returns a (1 x n) numpy array with the sigmoid function applied to
                 all elements of x
                '''
                return 1/(1+np.exp(-x))

        def sigmoid_prime(self,x):
                '''
                  Returns a (1 x n) numpy array containing the partial derivatives of
                  the sigmoid function with respect to each parameter
                '''
                f = sigmoid(x)		
                return np.multiply(f,(1-f_mat))

        def feedforward(self,inputs):
                # The actual work of taking in the inputs, then computing and propagating
                # the output to the next layer
                inputs = np.hstack((np.ravel(inputs),np.array([1])))    # ravel is used for future-proofing
                total = np.matmul(inputs,self.layer_mat)                # e.g. CNNs
                return self.sigmoid(total)

        def param_update(self):
                pass
        
        def mse_loss(self,output,true_val):
                # Output	: Prediction made by neural network
                # true_val	: True value from dataset
                if len(output) != len(true_val):
                        print('Error: Arrays are not of equal length')
                return ((true_val-output)**2)/len(output)


class OneLayerNN(NeuralLayer):

        def __init__(self):
                self.layer1 = NeuralLayer(2,2)
                self.output_layer = NeuralLayer(2,1)

        def train(self,data,true_values):
                learn_rate = 0.1
                epochs = 1000

                for epoch in range(epochs):
                        for x,y in zip(data,true_values):
                                h1 = self.layer1.feedforward(x)
                                output = self.output_layer.feedforward(h1)
                                print('h1:')
                                print(h1)
                                print('output:')
                                print(output)
                                break


                
                

data = np.array([
        [-2,-1],
        [25,6],
        [17,4],
        [-15,6]
        ])
true_values = np.array([
        1,
        0,
        0,
        1
        ])

network = OneLayerNN()
network.train(data,true_values)
