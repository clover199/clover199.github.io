import numpy as np
from numpy.random import random as rand


def gradient_descent(function, gradient, initial_value=None, step=0.01, *arg, **kwarg):
    ''' Given the function and the gradient of the function
        returns the minimal value. If initial value is not given,
        '''
    error_cutoff = 1e-12
    max_iteration = 1e5
    iterations = 0

    error = 1
    x = np.array(initial_value)
    while error>error_cutoff and iterations<max_iteration:
        iterations += 1
        previous_f_value = function(x, *arg, **kwarg)
        gradient_value = gradient(x, *arg, **kwarg)
        x = x - step*gradient_value
        error = max(step*np.max(abs(gradient_value)), np.max(abs(function(x)-previous_f_value)))
    if iterations==max_iteration:
        warn("Maximum iterations reached. error={:}".format(error))
    return x

def sigmoid(x):
    x = np.array(x)
    return 1./(1+np.exp(-x))

def restricted_boltzmann_machine(visible_nodes_number, hidden_nodes_number, objective_function, parameter_space):
    ''' designed to minimize the objective_function '''

    weights = rand((visible_nodes_number, hidden_nodes_number))
    bias_visible = rand(visible_nodes_number)
    bias_hidden = rand(hidden_nodes_number)

    visible_units = rand(visible_nodes_number)
    hidden_units = rand(hidden_nodes_number)

def neural_network(hidden_node_numbers, training_X, training_Y, testing_X, lmbda=0.0, iterations=10, step=1e-3, print_cost=False):
    ''' use backpropagation algorithm to calculate the gradient
        total layer number = hidden layer number + 2
        index layer from 0 '''

    # initialioze
    mean_pixel = training_X.mean(axis=0)
    Y = training_Y
    n = training_X.shape[0]    # sample size
    node_numbers = np.append(training_X.shape[1], hidden_node_numbers)
    try:
        node_numbers = np.append(node_numbers, Y.shape[1])
    except IndexError:
        node_numbers = np.append(node_numbers, 1)
    node_numbers = node_numbers.astype(int)
    n_layer = len(node_numbers)
    theta_ = [None]*n_layer
    for i in range(1,n_layer):    # layer_i
        q = 1+node_numbers[i-1]
        p = node_numbers[i]
        theta_[i] = 1e-5*(0.5-np.random.random((q,p)))
    dtheta_ = [None]*n_layer
    a_ = [None]*n_layer
    a_[0] = training_X-mean_pixel

    def update_layer():
        z_i = None
        for i in range(1,n_layer):    # layer_i
            z_i = theta_[i][0,:] + a_[i-1].dot(theta_[i][1:,:])
            a_[i] = sigmoid(z_i)
        return z_i

    def cost(Y_hat):
        # J = -np.sum( Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat) )/n
        J = np.sum(np.log(1+np.exp(Y_hat)) - Y*Y_hat)/n
        for i in range(1,n_layer):    # layer_i
            J += lmbda*np.sum(theta_[i]**2)/2./n
        return J

    def get_gradient():
        delta_i = a_[-1]-Y
        for i in range(n_layer-1,0,-1):    # layer_i
            a_i = np.concatenate( (np.ones((n,1)), a_[i-1]), 1 )    # attach the ones
            dtheta_[i] = np.dot(a_i.T, delta_i)/n
            delta_i = np.dot(delta_i,theta_[i][1:,:].T) * a_[i-1] * (1-a_[i-1])

    for t in range(iterations):
        z_i = update_layer()
        pre_cost = cost(z_i)
        get_gradient()
        for i in range(1,n_layer):    # layer_i
            theta_[i] -= step*dtheta_[i]
        z_i = update_layer()
        aft_cost = cost(z_i)
        num_change = 0
        while aft_cost>pre_cost:
            step = step/2.
            num_change += 1
            for i in range(1,n_layer):    # layer_i
                theta_[i] += step*dtheta_[i]
            z_i = update_layer()
            aft_cost = cost(z_i)
            if num_change>100:
                raise ValueError("cannot find a good step")
        if print_cost:
            print ("iteration %d \t cost %.6f \t step %.2e" % (t, aft_cost, step))

    a_[0] = testing_X-mean_pixel
    update_layer()
    Y_hat = a_[-1]
    testing_Y = map(np.argmax, Y_hat)
    return np.array(testing_Y, dtype=int)

import read_mnist
lbl, img = read_mnist.read()
train_x = img[:100,:]
train_y = np.array([np.where(lbl[:100]==x,1,0) for x in range(10)], dtype=int).transpose()
test_x = img[100:200,:]
test_y = neural_network([100], train_x, train_y, train_x, lmbda=0.0, iterations=100, step=0.1, print_cost=True)
print "correct rate", 1.0*np.sum(lbl[:100]==test_y)/test_y.shape[0]
# print optimization(lambda x:2*(x[0]**2)+(x[1]**2), lambda x:np.array([4*x[0],2*x[1]]), np.array([4.0,0.5]))
