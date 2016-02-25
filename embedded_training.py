# Yanshan Guo & Lucy Lu
from scipy.stats import multivariate_normal 
from collections import defaultdict
import numpy as np
import math
import operator
import sys
from decimal import *
"""
embedded_training performs all necessary steps of the modified Baum-Welch algorithm for speech recognition
"""
class embedded_training:
	def __init__(self,observation,state_space):
		self.observation = observation
		self.state_space = state_space
		self.init_A = self.initialize_A()
		self.init_B = self.initialize_B()
		

	def global_u_and_sigma(self):
		"""
		global_u_and_sigma computes the global mean and variance of the entire training data
		"""
		ob =[]
		for observation in self.observation:
			for f in observation:
				ob.append(f)
	
		g_u = np.mean(np.array(ob),axis=0)
		g_sigma = np.cov(np.array(ob).T)
		g_sigma_dia = g_sigma.diagonal()
		
		return np.array(g_u), np.array(g_sigma_dia)

	def get_A_and_B(self):
		return self.init_A, self.init_B

	def forward_backward(self):
		"""
		forward_backward iterates while updating transition and omission matrices until they converge
		"""
		A = self.init_A
		B = self.init_B
		
		while True:

			# E-step

			converge = 0
			gamma = {}
			test = {}

			for observation in self.observation: #for every observation
				
				alpha = self.alpha(A,B,observation)
				beta = self.beta(A,B,observation)
				T = len(observation)
			
				for s in self.state_space: # for every state 
					for i in range(3):						
						for t in range(1,T+1): # at any time t
							o_t = observation[t-1]
							t_j = s + str(i) + ":"+str(t)
							state = s + str(i)
							a = alpha.get(t_j)
							b = beta.get(t_j)
							d = alpha.get("f:"+str(T))
							if d == 0.0:
								l = 0
							else:
								l = a *b/d
							if math.isnan(l) or math.isinf(l):
								l = 0
							if test.get(state) is None:
								test[state] =  [l]
							else:
								test.get(state).append(l)
							if gamma.get(state) is not None:
								gamma.get(state).append((o_t,l)) # store o_t and P(j|o_t)
							else:
								gamma[state] = [(o_t,l)]
			# M-step
			for s in self.state_space: # for every state 
				for i in range(3):
					state = s + str(i)
					sum_d = 0
					for t in gamma.get(state):
						sum_d += t[1]
					if sum_d != 0:
						mult_ = map(lambda x: np.multiply(x[1]/sum_d, x[0]),gamma.get(state))		
						sum_n = reduce(lambda x,y: x+y, mult_)				
						u = sum_n		
						mult_2 = map(lambda x: np.multiply(np.multiply(x[1], np.subtract(x[0],u)),np.subtract(x[0],u).T),gamma.get(state))
						sum_n_2 = reduce(lambda x,y: x+y, mult_2)
						sig = np.multiply(sum_n_2,1/sum_d)
						converge += abs(np.sum(np.subtract(B.get(state)[0],u)))+ abs(np.sum(np.subtract(B.get(state)[1],sig)))
						B[state] = [u,sig] # update mean and covariance matrix for each state
					else:
						break
			if converge < 0.001:
				break
		return A, B

	def initialize_A(self):
		"""
		initialize_A initializes the transition matrix A
		"""
		list_of_phones = self.state_space
		list_of_matrix_a = {}
		for phone in list_of_phones:
			matrix = [[0 for i in range(3)] for j in range(3)]
			for i in range(3):
				if i != 2:
					matrix[i][i+1] = 0.5
					matrix[i][i] = 0.5
				else:
					matrix[i][i] = 0.5
			list_of_matrix_a[phone] =  matrix
		return list_of_matrix_a	

	def alpha(self,A,B,observation):
		"""
		alpha generates the alpha matrix using forward algorithm
		"""
		T = len(observation)
		alpha = {}
		for s in self.state_space:
			for i in range(3):
				b = multivariate_normal.pdf(observation[0], B.get(s+str(i))[0], np.diag(B.get(s+str(i))[1]))
				if b != 0:
					b = abs(math.log(b))
				# prob from start state 1
				if i ==0:
					alpha[s+str(i)+":1"] = b*(1/float(len(self.state_space)))
				else:
					alpha[s+str(i)+":1"] = 0
		for t in range(2,T+1):
			for s in self.state_space:
				for i in range(3):
					b = multivariate_normal.pdf(observation[t-1], B.get(s+str(i))[0], np.diag(B.get(s+str(i))[1]))
					if b != 0:
						b = abs(math.log(b))
					prev = [alpha.get(s+str(i_)+":"+str(t-1))*A.get(s)[i_][i] for i_ in range(3)]					
					alpha[s+str(i)+":"+str(t)] = sum(prev)
		k = 0		
		for s in self.state_space:
			j = alpha.get(s+"2:"+str(T))
			if math.isnan(j):
				j = 0
			k += j * (1/float(len(self.state_space)))
		alpha["f:"+str(T)] = k
	
		return alpha
	def beta(self,A,B,observation):
		"""
		beta generates the beta matrix using backward algorithm
		"""
		T = len(observation)
		beta = {}
		for s in self.state_space:
			for i in range(3):
				# prob from start state 1
				if i ==2:
					beta[s+str(i)+":"+str(T)] = 0.5
				else:
					beta[s+str(i)+":"+str(T)] = 0
		for t in reversed(range(1,T)):
			for s in self.state_space:
				for i in range(3):
					b = multivariate_normal.pdf(observation[t], B.get(s+str(i))[0], np.diag(B.get(s+str(i))[1]))
					if b != 0:
						b = abs(math.log(b))
					prev = [beta.get(s+str(i_)+":"+str(t+1))*A.get(s)[i][i_] for i_ in range(3)]	
					beta[s+str(i)+":"+str(t)] = sum(prev)
		
		for s in self.state_space:
			b = multivariate_normal.pdf(observation[0], B.get(s+str(i))[0], np.diag(B.get(s+str(i))[1]))
			if b != 0:
				b = abs(math.log(b))
			prev = [beta.get(s+"0:1")*(1/float(len(self.state_space)))*b for i_ in range(3)]
		beta["0:1"] = sum(prev)
		return beta

	def initialize_B(self):
		"""
		initialize_B initializes the emission matrix by setting mean and covariance of every state to be the global
		mean and covariance
		"""
		g_u, g_sigma_dia = self.global_u_and_sigma()
		B = {}
		for state in self.state_space:
			begin = state + "0"
			mid = state + "1"
			end = state + "2"
			B[begin] = [g_u, g_sigma_dia]
			B[mid] = [g_u, g_sigma_dia]
			B[end] = [g_u, g_sigma_dia]
		return B
	

if __name__ == '__main__':
	#testing functions for debugging 
	observation = [[[1,2,3],[2,3,4],[3,4,5]],[[4,2,6],[3,5,6],[2,6,7]]]
	state_space = ['aa','iy']
	#sound_dict = read_in(state_space)
	train = embedded_training(observation,state_space)
	train.forward_backward()
