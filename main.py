'''
@authors
Tanmay Kulkarni
Ch Vishal
Aditya Srikanth

Preamble:

'''
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import pickle 
import os 

def getData():
	"""
	opens the dataset, creates numpy arrays for the points, for lda, splits the data according to 
	classes, for perceptron, splits the labels and data.
	Then it pickles the dataset

	"""
	for i in range(1,4):
		if not os.path.isfile('dataset_'+str(i)):
			data = pd.read_csv('./ML - Assignment 1 - Datasets/dataset_'+str(i)+'.csv').values # returns a numpy array
			coordinates = data[:,1:3]
			labels = data[:,-1]
			data_dict = {}
			data_dict['class_0_coordinates'] = coordinates[np.argwhere(labels == 0).flatten()]
			data_dict['class_1_coordinates'] = coordinates[np.argwhere(labels == 1).flatten()]
			data_dict['coordinates'] = coordinates
			data_dict['labels'] = labels
			with open('dataset_'+str(i),'wb') as f:
				pickle.dump(data_dict,f)
		else:
			continue
	print('dataset processed, temporaries generated.\nSuccess')

def perceptron(dataset_index):
	"""
		FILL YOUR CODE HERE
	"""
	pass

def lda(dataset_index,plot=True):
	with open('dataset_'+str(dataset_index),'rb') as f:
		data_dict = pickle.loads(f.read())
	# get the statistics
	data_class_0 = data_dict['class_0_coordinates']
	mean_0 = np.mean(data_class_0,axis=0)
	covar_0 = np.cov(data_class_0.T,bias=True)
	
	data_class_1 = data_dict['class_1_coordinates']
	mean_1 = np.mean(data_class_1,axis=0)
	covar_1 = np.cov(data_class_1.T,bias=True)

	Sw = covar_0*data_class_0.shape[0] + covar_1*data_class_1.shape[0]
	delta_m = mean_0 - mean_1

	w = np.linalg.inv(Sw)
	w = w.dot(delta_m)
	w = w/np.sum(w**2)**0.5 					# making into a unit vector
	threshold = mean_0 + mean_1
	threshold /= 2
	print(w,np.sum(threshold*w))
	"""
	alternative solution, N1(mu1,sigma1) = N2(mu2,sigma2)

	"""
	plt.scatter(data_dict['coordinates'][:,0],data_dict['coordinates'][:,1])
	plt.show()
	
	if plot:
		plt.cla()
		plt.scatter(data_class_0[:,0],data_class_0[:,1],c='blue')
		plt.scatter(data_class_1[:,0],data_class_1[:,1],c='red')
		plt.scatter(threshold[0]*w[0],threshold[1]*w[1],c='yellow')
		x = np.arange(-1,1,.1)
		y = -w[1]*x/w[0]
		print(x,y)
		plt.plot(x,y)
		plt.xlim(-3,3)
		plt.ylim(-3,3)
		plt.show()


def main():
	getData()
	
	to_run_program = int(input("enter 1 for perceptron, 2 for lda\n"))
	on_which_dataset = int(input("enter 1 for dataset-1, 2 for dataset-2, 3 for dataset-3\n"))
	if to_run_program == 1:
		perceptron(on_which_dataset)
	elif to_run_program == 2:
		lda(on_which_dataset)

if __name__ == "__main__":
	main()