# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:40:36 2020
Sequential Feed Forward Deep Neural Network for Binary Classification
From Neural Networks and Deep Learning Course by Andrew Ng
@author: Shamith Achanta
"""
    
def main():
    train_x_orig, Train_y, test_x_orig, Test_y, classes = load_data()
    
    # dimensions
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    
    # Reshape the training and test examples 
    # make B.shape == (1,2,3,4) into B.shape == (2*3*4, 1)
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    
    # Standardize data to have feature values between 0 and 1.
    Train_x = train_x_flatten/255.0
    Test_x = test_x_flatten/255.0
    
    ### CONSTANTS DEFINING THE MODEL ####
    layers_dim = [12288, 20, 7, 5, 1] #  4-layer model
    activations = ['relu', 'relu', 'relu', 'sigmoid']
    
    #hyper constants
    learning_rate = 0.75
    num_iters = 2500
    
    parameters = nn_layers(layers_dim)
    cost, parameters = nn_compile(Train_x, Train_y, num_iters, parameters, activations, learning_rate)   
    plot(cost, learning_rate)
    predictions_train = predict(Train_x, Train_y, parameters)
    predictions_test = predict(Test_x, Test_y, parameters)      
    
def nn_layers(layers_dim, const=0.01):
    """
    Parameters
    ----------
    layers_dim : tuple or list
        dimensions of activation layers including input layer
    hyper_const : float
        constant for random initialization of parameters

    Returns
    -------
    parameters : dictionary
        weights and bias

    """
    parameters = {}
    L = len(layers_dim)
    
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) * const
        parameters["b" + str(l)] = np.zeros((layers_dim[l], 1))
        
    return parameters

def nn_compile(X, Y, num_iters, parameters, activation, learning_rate, hyper_const=0.01):
    """
    Parameters
    ----------
    X : Tensor
        Input
    Y : Tensor
        Output
    parameters : dictionary
        weights and bias
    activation : list
        activation function for each layer
    learning_rate : float
        learning rate
    hyper_const : float
        constant for leaky_relu
    reg_const : float
        regularization constant

    Returns
    -------
    cost : list
        list of cost functions

    """
    cost = []
    L = len(parameters) // 2
    
    for i in range(num_iters):
        caches = forward_propagate(X, parameters, activation, hyper_const)
        J = compute_cost(Y, caches["A" + str(L)])
        grads = backward_propagate(Y, parameters, caches, activation, hyper_const)
        parameters = nn_update(parameters, grads, learning_rate)
        
        # Print the cost every 100 training example
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, J))
            cost.append(J)
    
    return (cost, parameters)
        
def nn_update(parameters, grads, learning_rate):
    """
    Parameters
    ----------
    parameters : dictionary
        weights and bias
    grads : dictionary
        gradients of parameters
    learning_rate : float
        learning rate
    reg_const : float
        regularization constant
    m : int
        number of training examples
    L : int
        number of layers

    Returns
    -------
    parameters : dictionary
        weights and bias

    """
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters
    
def forward_propagate(X, parameters, activation, hyper_const):
    cache = {"A0": X}
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        assert (parameters["W" + str(l)].shape[1] == cache["A" + str(l-1)].shape[0])
        
        cache["Z" + str(l)] = np.dot(parameters["W" + str(l)], cache["A" + str(l-1)]) + parameters["b" + str(l)]
        
        Z = cache["Z" + str(l)]
        
        if activation[l-1] == 'sigmoid':
            cache["A" + str(l)] = sigmoid(Z)
            
        elif activation[l-1] == 'relu':
            cache["A" + str(l)] = relu(Z)
        
        elif activation[l-1] == 'leaky_relu':
            cache["A" + str(l)] = leaky_relu(Z, hyper_const)
        
        elif activation[l-1] == 'tanh':
            cache["A" + str(l)] = tanh(Z)
            
        else:
            raise Exception("Available Activation Functions: sigmoid, relu, leaky_relu, tanh") 
        
        assert (cache["A" + str(l)].shape == (parameters["W" + str(l)].shape[0], cache["A" + str(l-1)].shape[1]))
        
    return cache

def compute_cost(Y, AL):
    m = Y.shape[1]
    
    J = - 1 * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T)) / m
    
    J = np.squeeze(J)     # To make sure J's shape is what we expect (e.g. this turns [[17]] into 17).
        
    assert(J.shape == ())
    
    return J

def backward_propagate(Y, parameters, cache, activation, hyper_const):
    m = Y.shape[1]
    L = len(parameters) // 2
    grads = {}
    Y = Y.reshape(cache["A" + str(L)].shape) # after this line, Y is the same shape as AL
    
    grads["dA" + str(L)] = -Y/cache["A" + str(L)] + (1-Y)/(1-cache["A" + str(L)])
    
    for l in reversed(range(1, L+1)):
        grads["dZ" + str(l)] = activation_derivative(grads["dA" + str(l)], cache["Z" + str(l)], activation[l-1])
        assert (cache["Z" + str(l)].shape == grads["dZ" + str(l)].shape)
        grads["dA" + str(l-1)] = np.dot(parameters["W" + str(l)].T, grads["dZ" + str(l)])
        grads["dW" + str(l)] = np.dot(grads["dZ" + str(l)], cache["A" + str(l-1)].T) / m
        grads["db" + str(l)] = np.sum(grads["dZ" + str(l)], axis=1, keepdims=True) / m
        
        assert (grads["dA" + str(l-1)].shape == cache["A" + str(l-1)].shape)
        assert (grads["dW" + str(l)].shape == parameters["W" + str(l)].shape)
        assert (grads["db" + str(l)].shape == parameters["b" + str(l)].shape)
    
    for l in range(1, L+1):
        del grads["dZ" + str(l)]
        del grads["dA" + str(l)]
    del grads["dA0"]
    
    return grads

# activation functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(Z, 0)

def leaky_relu(Z, hyper_const):
    return np.maximum(Z, hyper_const)

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def activation_derivative(dA, Z, activation, hyper_const=0.01):
    if activation == 'sigmoid':
        activ_derivative = dA * sigmoid(Z) * (1 - sigmoid(Z))
    
    elif activation == 'relu':
        activ_derivative = np.array(dA, copy=True)
        activ_derivative[Z <= 0] = 0
        
    elif activation == 'leaky_relu':
        activ_derivative = np.array(dA, copy=True)
        activ_derivative[Z < 0] = hyper_const
        
    elif activation == 'tanh':
        activ_derivative = dA * (4 / np.power(np.exp(Z) + np.exp(-Z), 2)) 
        
    else:
        raise Exception("Available Activation Functions: sigmoid, relu, leaky_relu, tanh") 
        
    return activ_derivative  

def plot(cost, learning_rate):
    plt.plot(cost)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
main()
