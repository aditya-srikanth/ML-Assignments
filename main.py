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
import matplotlib.animation as animation
import pickle 
import os 

fig,ax = plt.subplots()
plot, = plt.plot([],[],animated=True)
x = np.arange(-3,3,0.1)
line = None

w_list = []

def init():
	return plot,

def animate(i):
	y =  -w_list[i][0]/w_list[i][2] - w_list[i][1]/w_list[i][2]*x
	plot.set_data(x,y) 
	return plot,

def plot_data(list_of_weights,class_0_coordinates,class_1_coordinates,labels):
	ax.scatter(class_0_coordinates[:,0],class_0_coordinates[:,1],c='blue')
	ax.scatter(class_1_coordinates[:,0],class_1_coordinates[:,1],c='red')
	global w_list
	w_list = list_of_weights
	ani = animation.FuncAnimation(fig, animate, init_func=init, frames=np.arange(0,len(w_list),1), blit=True)
	plt.show()
	
def get_hard_coded_params():
	
	## Hardcoded parameters
	number_of_classes=2
	pickle_file_prefix='dataset_'
	number_of_pickle_files=3
	datasets_path='./ML - Assignment 1 - Datasets'
	dataset_filename_prefix='dataset_'

	return number_of_classes,pickle_file_prefix,number_of_pickle_files,datasets_path,dataset_filename_prefix

def getData():
	"""
	opens the dataset, creates numpy arrays for the points, for lda, splits the data according to 
	classes, for perceptron, splits the labels and data.
	Then it pickles the dataset

	"""
 
	number_of_classes,pickle_file_prefix,number_of_pickle_files,datasets_path,dataset_filename_prefix=get_hard_coded_params()

	for i in range(1,number_of_pickle_files+1):
		if not os.path.isfile(pickle_file_prefix+str(i)): # Check for the presence of pickled files
			data = pd.read_csv(datasets_path+'/'+dataset_filename_prefix+str(i)+'.csv',header=0).values # returns a numpy array
			coordinates = data[:,1:3] # The first column contains the index the next two contain X and Y coordinates
			labels = data[:,-1] # The last column contans the class the particular data point belongs to
			
			data_dict = {} # Dictionary that contains the whole data as well as the data separated by class
			for j in range(0,number_of_classes): # Store the coordinates for each class
				key='class_{0}_coordinates'.format(j)
				data_dict[key] = coordinates[np.argwhere(labels == j).flatten()] # Finds all points where labels is j
				
			data_dict['coordinates'] = coordinates
			data_dict['labels'] = labels

			with open(pickle_file_prefix+str(i),'wb') as f:
				pickle.dump(data_dict,f)
		
		else:
			continue

	print('dataset processed, temporaries generated.\nSuccess')

def perceptron(dataset_index):
	
	
	number_of_classes,pickle_file_prefix=get_hard_coded_params()[0:2]

	with open(pickle_file_prefix+str(dataset_index),'rb') as f:
		data_dict = pickle.loads(f.read())
 
	## Hardcoded for perceptron
	number_of_basis_funcs = 2 
	learn_rate = 0.1	
	number_of_iter = 1000
	w = np.random.rand(number_of_basis_funcs+1,1) # As we assume a 2 input + 1 bias
	coordinates = data_dict['coordinates']
	labels = data_dict['labels']
	perceptron_labels = 2*labels-1
	'''
		Transfromation if any should happen here.
	'''

	## Assuming no transformations
	transformed = coordinates
	n,m = transformed.shape
	onz = np.ones((n,1))
	transformed = np.hstack( (onz,transformed) )

	w_list = []
	w_list.append(w)

	for i in range(0,number_of_iter): 
		val=np.dot(transformed,w).flatten()

		indices1 = np.argwhere( (perceptron_labels==-1) & (val>=0) ).flatten()
		indices2 = np.argwhere( (perceptron_labels==1) & (val<0) ).flatten()
		error = -1 * np.sum (perceptron_labels[ indices1 ] * val[ indices1 ]  )
		error = error + -1 * np.sum ( perceptron_labels[ indices2 ] * val[ indices2 ] ) 
		print('loss: ',error)
		if error == 0.:
			break
		grad =  -1*( np.dot(perceptron_labels[ indices1 ],transformed[ indices1 ]) )
		# print(grad.shape)
		grad =  grad + -1*np.dot(perceptron_labels[ indices2 ] , transformed[ indices2 ])
		# print('grad: ',grad)
		grad=grad.T
		temp = w.T - learn_rate*grad  
		w = temp.T
		w_list.append(w)
	
	print('The value for weight ',w)
	
	print('showing animation')
	plot_data(w_list,data_dict['class_0_coordinates'],data_dict['class_1_coordinates'],labels)

def lda(dataset_index,plot=True):

	number_of_classes,pickle_file_prefix=get_hard_coded_params()[0:2]

	with open(pickle_file_prefix+str(dataset_index),'rb') as f:
		data_dict = pickle.loads(f.read())
	
	# get the statistics
	data={}
	mean={}
	covar={}
	for i in range(0,number_of_classes):
		data[i] = data_dict['class_{0}_coordinates'.format(i)]
		mean[i] = np.mean(data[i],axis=0) # Along the columns
		covar[i] = np.cov(data[i].T,bias=False)

	## Currently we only differente between two classes
	Sw = covar[0]*data[0].shape[0] + covar[1]*data[1].shape[0] #Finding the weighed sum
	delta_m = mean[0] - mean[1]

	w = np.linalg.inv(Sw)
	w = w.dot(delta_m)
	w = w/np.sum(w**2)**0.5 					# making into a unit vector
	threshold = mean[0] + mean[1]
	threshold /= 2 # Using the average as the threshold
	print('The value for the weight is {0} and the threshold is {1}'.format( w, np.sum(threshold*w) ) )
	"""
	alternative solution, N1(mu1,sigma1) = N2(mu2,sigma2)

	"""
	plt.scatter(data_dict['coordinates'][:,0],data_dict['coordinates'][:,1])
	plt.show()
	
	if plot:
		plt.cla()
		plt.scatter(data[0][:,0],data[0][:,1],c='blue')
		plt.scatter(data[1][:,0],data[1][:,1],c='red')
		plt.scatter(threshold[0]*w[0],threshold[1]*w[1],c='green')
		x = np.arange(-1,1,.1)
		y = -w[1]*x/w[0]
		plt.plot(x,y)
		plt.xlim(-3,3)
		plt.ylim(-3,3)
		plt.show()


def main():
	getData()
	
	to_run_program = int(input("enter 1 for perceptron, 2 for lda\n"))
	on_which_dataset = int(input("enter 1\n for dataset->1\n 2 for dataset->2\n 3 for dataset->3\n"))
	if to_run_program == 1:
		perceptron(on_which_dataset)
	elif to_run_program == 2:
		lda(on_which_dataset)

if __name__ == "__main__":
	main()