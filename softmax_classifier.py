# Unfolding recursive auto-encoder as in Manning & Socher NIPS2011
# built for use with nltk.tree Trees
import numpy as np
import random
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as l_bfgs
from nltk.tree import Tree
import nltk
from collections import defaultdict
import math

# class initiates with list of training examples x^(i), each a Tree; also input & encoding size
class softMaxClassifier:

  def get_J_sub_gradient(self, data):
    x_i = np.concatenate((data[0].flatten(),np.array([1])))
    y_i = data[1]

    J_sum = np.zeros((self.n*self.m+1)*self.label_count)
    #error_matrix = np.multiply(x_i, self.W)
    #error = np.sum(np.sum(error_matrix))

    cumul_value = 0
    final_label_values = np.zeros(self.label_count)
    
    for index in range(self.label_count):

        label_weights = self.W[(self.n*self.m+1)*(index):(self.n*self.m+1)*(index+1)]
        label_value = pow(math.e,np.dot(label_weights, x_i))
        cumul_value += label_value
        final_label_values[index]=label_value
        

    for index in range(self.label_count):
      temp_y = 0
      if y_i - 1 == index:
        temp_y = 1
      temp_arr = x_i*(temp_y - 1.0*final_label_values[index]/cumul_value)
      J_sum[(self.n*self.m+1)*(index):(self.n*self.m+1)*(index+1)] = temp_arr

    
    
    
    #try:
     # J_sum = 1.0*y_i*math.log10(h) + (1.0-y_i)*math.log10(1.0-h)
    #except ValueError:
     # if h == 1.0:
      #  h -= 0.000001
      #elif h == 0.0:
      #  h += 0.000001
      #J_sum = 1.0*y_i*math.log10(h) + (1.0-y_i)*math.log10(1.0-h)
      
    return J_sum
  
  # return the average cost of all training data, with a wieght decay factor
  def get_J_gradient(self, x):
    self.W = x[:(self.n*self.m+1)*self.label_count]
    J_sum = np.zeros((self.n*self.m+1)*self.label_count)
    for data in self.training_data:
      J_sum+=(self.get_J_sub_gradient(data))
    return -(1.0/len(self.training_data))*J_sum + (self.wd/2.0)*x[:(self.n*self.m+1)*self.label_count]

 
  # The cost function: computed as SUM OF unfolded reconstruction error OF ALL non-term. nodes
  # return the sum of reconstruction error of root node + all immediate child nodes
  def get_J_sub(self, data):
    x_i = np.concatenate((data[0].flatten(),np.array([1])))
    y_i = data[1]

    #error_matrix = np.multiply(x_i, self.W)
    #error = np.sum(np.sum(error_matrix))

    cumul_value = 0
    final_label_value = 0
    
    for index in range(self.label_count):

        label_weights = self.W[(self.n*self.m+1)*(index):(self.n*self.m+1)*(index+1)]
        label_value = pow(math.e,np.dot(label_weights, x_i))
        cumul_value += label_value
        if y_i - 1 == index:
          final_label_value = label_value
    
    if final_label_value == 0:
      final_label_value+=0.00001
    try:
      J_sum = math.log10(1.0*final_label_value/cumul_value)
    except ValueError:
      if h == 1.0:
        h -= 0.000001
      elif h == 0.0:
        h += 0.000001
      elif cumul_value == 0.0:
        J_sum = 0
    
    #try:
     # J_sum = 1.0*y_i*math.log10(h) + (1.0-y_i)*math.log10(1.0-h)
    #except ValueError:
     # if h == 1.0:
      #  h -= 0.000001
      #elif h == 0.0:
      #  h += 0.000001
      #J_sum = 1.0*y_i*math.log10(h) + (1.0-y_i)*math.log10(1.0-h)
      
    return J_sum
  
  # return the average cost of all training data, with a wieght decay factor
  def get_J(self, x):
    self.W = x[:(self.n*self.m+1)*self.label_count]
    J_sum = 0.0
    for data in self.training_data:
      J_sum += self.get_J_sub(data)
    return -(1.0/len(self.training_data))*J_sum + (self.wd/2.0)*np.sum(x[:(self.n*self.m+1)*self.label_count]**2)
  
 
  # train the new uRAE upon initiation
  def __init__(self, training_data, weight_decay_p, epsilon, label_count, numerical=False):
    self.training_data = training_data
    self.n, self.m = training_data[0][0].shape
    self.wd = weight_decay_p  # NOTE: WEIGHT DECAY PARAMETER --> what should this be set at??
    self.label_count = label_count
    self.W = np.random.normal(0.0,epsilon**2,(self.n*self.m+1)*self.label_count)
    
    print "training model..." 
    if numerical:
      x, f, d = l_bfgs(self.get_J, self.W, None, (), True)
    else:
      x, f, d = l_bfgs(self.get_J, self.W, self.get_J_gradient)
    print "minimum found, J = " + str(f)
    #self.W = np.array(x[:self.n*self.m]).reshape((self.n,self.m))
    
    # FOR DEBUG
    self.x = x
    self.f = f
    self.d = d

training_data = []
for k in range(20000):
  arr = [random.randint(0,1), random.randint(0,1), random.randint(0,1), random.randint(0,1)]
  training_data.append([np.array(arr).reshape((2,2)),random.randint(0,5)])
#smc = softMaxClassifier(training_data,0.01,0.5,6)
