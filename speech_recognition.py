# Yanshan Guo & Lucy Lu
from scipy.io import wavfile
from scipy.stats import multivariate_normal 
from scikits.talkbox.features import mfcc
import numpy
import math
import operator
import sys
from embedded_training import embedded_training
from testing import testing
"""
This is a simple speech recognition program that predicts which phone a given sound input is.
After reading in training sound files, we train the acoustic model using modified Baum-Welch 
algorithm for speech recognition.Then we used the acoustic model to predict phones for a test set.

This program runs with Python2.7 and depends many packages. Please make sure you have downloaded all packages
before running the program

Edited by Yanshan Guo and Lucy Lu
"""

def main():
	filename = sys.argv[1]
	state_space,test_space = read_in_state_space_and_test_set(filename)
	observations = read_in_training(state_space)
 	A = {}
 	B = {}
	for state in state_space:
		print "Training the acoustic likelihood matrix for", state, "..."
		training = embedded_training(observations.get(state),[state]) # training
		A_,B_ = training.forward_backward()
		for key in A_.keys():
			A[key] = A_.get(key)
		for key in B_.keys():
			B[key] = B_.get(key)
	print "Finish training"
	print "Start testing"
	for test_number in test_space.keys(): #testing 
		test_set = test_space.get(test_number)
		test_set = read_in_testing(test_set)
		test = testing(A,B,test_set)
		result = test.predict()
		print "Result for test:", test_number,":"
		print result
		print "\n"

def read_in_state_space_and_test_set(filename):
	"""
	read_in_state_space_and_test_set reads in the state space and test set for speech recognition
	"""
	test_space = {}
	test = open(filename)
	state_space = test.readline().strip("\n").split(",")
	test_space[0] = state_space
	i = 1
	for line in test:
		test_space[i] = line.strip("\n").split(",")
		i+=1
	return state_space,test_space

def read_in_testing(test_space):
	"""
	read_in_testing reads in sound files for all phones in the test set
	"""
	sounds = {}
	for test in test_space:
		for i in range(4,5):
			dir_name = "train/"
			file_name = test + str(i) + ".wav"
			sample_rate,sound = wavfile.read(dir_name + file_name)
			feature = mfcc(sound, nwin=int(sample_rate * 0.01), fs=sample_rate, nceps=13)[0] # mfcc
			sounds[test] = feature
	return sounds

def read_in_training(state_space):
	"""
	read_in_training reads in sound files for all phones in the training set
	"""
	sounds = {}
	for state in state_space:
		temp = []
		for i in range(1,4):
			dir_name = "train/"
			file_name = state + str(i) + ".wav"
			sample_rate,sound = wavfile.read(dir_name + file_name)
			feature = mfcc(sound, nwin=int(sample_rate * 0.01), fs=sample_rate, nceps=13)[0] #mfcc
			temp.append(feature)
		sounds[state] = temp
	return sounds

main()