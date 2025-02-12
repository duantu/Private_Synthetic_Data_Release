# This code creates a (k, lambda, eta, beta)- base synopsis generator that is (epsilon,delta)-DP 

import numpy as np
import math
import sympy as sp
from itertools import combinations
import pandas as pd
import time
import copy


lower_bound = -1 # data lower bound
upper_bound = 1  # data upper bound
data_precision = 2 # data precision

n = 200 
num_points = 2*n # number of data points
m = num_points*math.ceil(math.log(((upper_bound - lower_bound)/10**(-data_precision)) +1)) # size of the domain universe

delta =  0.1 # DP parameter
eta = 0.1 # edge for boosting
beta = 0.1 # failure probability of the base synopsis
k = math.ceil(2*((math.log(2/beta)+m)/(1-2*eta))) # number of query sample as demanded by Lemma 6.5 
# Assume coefficient \in (0,1], changing one x_i can at most change 1*[(1+x_j)^2 - (-1+x_j)^2] = 4x_j <= 4.

rho = 1/num_points**2 # l_1 sensitivity of our query = 4 sub




Llambda = 0.4 # accuracy parameter lambda
epsilon = (math.log(1/beta)*rho*math.sqrt(k*math.log(1/delta)))/Llambda
epsilon = math.trunc(epsilon*100)/100+10**(-data_precision) # round epsilon up to 2 decimal points 


# decide on a set of real data 
real_X = np.random.uniform(low=lower_bound, high=upper_bound, size=num_points)
sum_real_X_squared = np.sum(np.square(real_X))

# initialize the synopsis to be some arbirary set of data, say from the standard normal
# for verification purposes, if fake_X = real_X, the initial error should be the same as the added laplace noise
# fake_X = real_X 
fake_X = np.random.randn(num_points)
fake_X_copy = copy.copy(fake_X) # save a copy of fake_X
sum_fake_X_squared = np.sum(np.square(fake_X_copy))



all_coeff = np.round(1-np.linspace(0,10000,10000, endpoint=False)/10000,4)
sampled_queries = np.random.choice(all_coeff,k)



#### BOOSTING LOOP STARTS ####

# initialize all-zero arrays to store noiselss query output, noisy output, and laplace noise  
real_output = np.zeros(k)
real_data_noisy_output = np.zeros(k)
lap_noise = np.zeros(k)
fake_output = np.zeros(k)
error = np.zeros(k) # store |q(X) - noisy_output| for each q



# for each query, compute its real output, noisy output, and initial error
for index, item in enumerate(sampled_queries):
    
    # real output
    real_output[index] = item*sum_real_X_squared/num_points**2

    
    # # compute noisy output on the real data
    lap_noise[index] = np.random.laplace(loc=0.0, scale=rho*(2*math.sqrt(2*k*math.log(1/delta))/epsilon), size=None) 
    # # lap_noise[index] = 0
    real_data_noisy_output[index] = real_output[index] + lap_noise[index]

    # # compute query output on fake data 
    fake_output[index] = item*sum_fake_X_squared/num_points**2

    # # calculate initial error
    # # notice that this is |q(X) - real_data_noisy_output|
    error[index] = abs(fake_output[index]-real_data_noisy_output[index])


#### COORDINATE DESCENT LOOP STARTS HERE ###
#### In this loop, we do coordinate descent, NOT multivariate Newton's method ####

# initialize number of coordinate descent iterations = 0
num_iter_descent = 0


# while we don't have |q(X) - noisy_output|<lambda/2 for all q, continue coordinate descent 
while not np.all(error < Llambda/2):
    
    # calculate the current total loss
    total_loss = np.sum(np.square(fake_output-real_data_noisy_output))/Llambda**2

    ### compute the partial derivative of the loss function with respect to each coordinate 
    loss_gradient = np.zeros(num_points) # initialize the partial derivative of each coordinate to be zero
    # fix xi    
    for i in range(num_points):
        xi = fake_X[i]
        # calculate the dq_j/dx_i for all j = 1, ..., k
        query_grad_wrt_xi_array = 2*xi*sampled_queries/num_points**2
        loss_gradient[i] = 2/Llambda**2*np.sum((sampled_queries/num_points**2*sum_fake_X_squared - real_data_noisy_output)*query_grad_wrt_xi_array)

    # find the coordinate with the max absolute value part_derivative 
    x_coord_descent = np.argmax(np.abs(loss_gradient))

    # update x value at the coordinate x_coord_descent 
    # First, we calculate the 2nd derivative wrt x_coord_descent
    xi = fake_X[x_coord_descent]
    loss_hessian_wrt_chosen_x_coord = 2/Llambda**2*np.sum((2*sampled_queries/num_points**2*xi)**2+2*sampled_queries/num_points**2*(sampled_queries/num_points**2*sum_fake_X_squared-real_data_noisy_output))
   
    # update fake_X at coordinate = x_coord_descent using 2nd order Newton's method 
    fake_X[x_coord_descent] = xi - loss_gradient[x_coord_descent]/loss_hessian_wrt_chosen_x_coord

    # update fake_output 
    sum_fake_X_squared = np.sum(np.square(fake_X))
    fake_output = sampled_queries/num_points**2*sum_fake_X_squared 

    # update error
    error = abs(fake_output-real_data_noisy_output)

    # update total loss
    total_loss = np.sum(np.square(fake_output-real_data_noisy_output))/Llambda**2

    # print current progress
    print(f"#iter {num_iter_descent} x_co={x_coord_descent}, 1st={loss_gradient[x_coord_descent]}, 2nd={loss_hessian_wrt_chosen_x_coord}, # queries above err={sum(error>Llambda/2)} fake x={fake_X[x_coord_descent]}")
    print('Total loss = ', total_loss)
    
    num_iter_descent += 1 

