'''
@authors
Tanmay Kulkarni
Ch Vishal
Aditya Srikanth

Preamble:
This code contains the implementation for Fisher's Linear Discriminent and the Perceptron Algorithm to help solve a classification problem.
Currently, this program has the capability to solve only for two classes.
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import pickle
import os
import math

fig,ax = plt.subplots()
plot, = plt.plot([],[],animated=True)
x = np.arange(-3,3,0.1) # We deal in the domain from -3 to 3 separated by 0.1
line = None

w_list = [] ## list to store the weights obtained during the training of the perceptron algorithm

def init():
	return plot,

def animate(i):
	y =  -w_list[i][0]/w_list[i][2] - w_list[i][1]/w_list[i][2]*x #w0 + w1x + w2y => Simiplifies to the expression used here
	plot.set_data(x,y)
	return plot,

def plot_data(list_of_weights,class_0_coordinates,class_1_coordinates,labels):
	'''
		This contains the animation for the perceptron algorithm.
	'''
	## the class points are static in nature
	ax.scatter(class_0_coordinates[:,0],class_0_coordinates[:,1],c='blue')
	ax.scatter(class_1_coordinates[:,0],class_1_coordinates[:,1],c='red')
	global w_list
	w_list = list_of_weights # assign the value to the global list
	ani = animation.FuncAnimation(fig, animate, init_func=init, frames=np.arange(0,len(w_list),1), blit=True)
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
	ani.save('perceptron3.mp4',writer = writer)
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
	perceptron_labels = 2*labels-1 # 2-1=1 and 0-1=-1 Thus, all the labels will now be 1 and -1
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
		val=np.dot(transformed,w).flatten() # the predicted value

		indices1 = np.argwhere( (perceptron_labels==-1) & (val>=0) ).flatten() # posive prediction and negative label
		indices2 = np.argwhere( (perceptron_labels==1) & (val<0) ).flatten() # negative prediction and positive label
		error = -1 * np.sum (perceptron_labels[ indices1 ] * val[ indices1 ]  )
		error = error + -1 * np.sum ( perceptron_labels[ indices2 ] * val[ indices2 ] )
		print('loss: ',error)

		if error == 0.: # we can't do any better than thatfalse poistive
			break

		grad =  -1*( np.dot(perceptron_labels[ indices1 ],transformed[ indices1 ]) )
		grad =  grad + -1*np.dot(perceptron_labels[ indices2 ] , transformed[ indices2 ])
		grad =  np.reshape(grad,(grad.shape[0],1))

		grad=grad
		w = w - learn_rate*grad  # Update for the gradient descent
		w_list.append(w)

	print('The value for weight ',w)

	## The code for the animation
	print('showing animation')
	# (weights from training,class 0, class 1, labels)
	plot_data(w_list,data_dict['class_0_coordinates'],data_dict['class_1_coordinates'],labels) ## Hardcoded for two classes

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
		# var.append(np.var(data))

	# Currently we only differente between two classes
	Sw = covar[0]*data[0].shape[0] + covar[1]*data[1].shape[0] #Finding the weighed sum
	delta_m = mean[1] - mean[0]
	w = np.linalg.inv(Sw)
	w = w.dot(delta_m)
	print(w)
	w = w/(np.sum(w**2)**0.5) 					# making into a unit vector
	print(w)
	# finding the projections
	projections = [[],[]]
	mu = [0,0]
	var = [0,0]
	#plot the points of two classes
	plt.scatter(data[0][:,0],data[0][:,1],c='blue',alpha=0.2)
	plt.scatter(data[1][:,0],data[1][:,1],c='red',alpha=0.2)
	plt.show()
	#calculate the point projections on the w vector
	for i in range(2):
		# projections[i].append(np.matmul(np.transpose(w).reshape(1,2),data[i].T))
		projections[i] = np.dot(data[i],w)
		mu[i] = np.mean(projections[i])
		var[i] = np.var(projections[i])
	a = 1/(2*math.sqrt(var[0])**2) - 1/(2*math.sqrt(var[1])**2)
	b = mu[1]/(math.sqrt(var[1])**2) - mu[0]/(math.sqrt(var[0])**2)
	c = mu[0]**2 /(2*math.sqrt(var[0])**2) - mu[1]**2 / (2*math.sqrt(var[1])**2) - np.log(math.sqrt(var[1])/math.sqrt(var[0]))
	# sol[0] = (-b + math.sqrt(b**2 - 4*a*c))/2*a
	# sol[1] = (-b - math.sqrt(b**2 - 4*a*c))/2*a
	sol = np.roots([a,b,c])
	#this is the solution to the intersection of two gaussians for the threshold
	x = sol[0] if sol[0] <= mu[0] and sol[0] >= mu[1] else sol[1]
	threshold = x*w
	print(threshold)
	# plt.plot((projections[i]*w.reshape(2,1))[::1,0],(projections[i]*w.reshape(2,1))[::1,0],'*',color='b',alpha=0.5)
	projection_vector_1 = np.stack([projections[0],projections[0]],axis=1)*w
	projection_vector_2 = np.stack([projections[1],projections[1]],axis=1)*w
	# print(projection_vector_1.shape**

	#these are the projected points on w vector
	plt.plot(projection_vector_1[::1,0],projection_vector_1[::1,1],'o',color='blue',alpha=0.5,zorder=1)
	plt.plot(projection_vector_2[::1,0],projection_vector_2[::1,1],'x',color='red',alpha=0.5,zorder=1)

	# plt.show()
	#creating the normal distributions of the projected points
	W_perpendicular = np.array([-w[1],w[0]])
	z=np.linspace(-3,3,1000)
	points=np.column_stack([z,z])

	projection_points=points*w
	normal_point_class1 = norm.pdf(z,mu[0],math.sqrt(var[0]))
	normal_point_class2 = norm.pdf(z,mu[1],math.sqrt(var[1]))
	normal_projection_vec_class1 = np.column_stack([normal_point_class1,normal_point_class1])*W_perpendicular
	normal_projection_vec_class2 = np.column_stack([normal_point_class2,normal_point_class2])*W_perpendicular
	normal_class1 = projection_points+normal_projection_vec_class1
	normal_class2 = projection_points+normal_projection_vec_class2
	plt.plot(normal_class1[::1,0],normal_class1[::1,1],'-',color='black',alpha=0.7,zorder=1)
	plt.plot(normal_class2[::1,0],normal_class2[::1,1],'-',color='black',alpha=0.7,zorder=1)
	# plt.show()

	# threshold = mean[0] + mean[1]
	# threshold /= 2 # Using the average as the threshold
	print('The value for the weight is {0} and the threshold is {1}'.format( w, threshold ) )
	# plt.scatter(data_dict['coordinates'][:,0],data_dict['coordinates'][:,1])
	# plt.show()
	
	#plotting the points of class 1 and class 2 along with the normal distributions and the threshold point
	if plot:
		# plt.cla()
		plt.scatter(data[0][:,0],data[0][:,1],c='blue',alpha=0.2)
		plt.scatter(data[1][:,0],data[1][:,1],c='red',alpha=0.2)
		plt.scatter(threshold[0]*w[0],threshold[1]*w[1],c='green',zorder=2)
		# x_vals = np.array(plt.gca().get_xlim())
		# y_vals = -(w[0]/w[1])*x_vals + (threshold[1]*w[1] + (w[0]/w[1])*threshold[0]*w[0])
		# plt.plot(x_vals,y_vals)
		x = np.arange(-1,1,.1)
		y = w[1]*x/w[0]
		plt.plot(x,y,zorder=1.5)
		plt.xlim(-3,3)
		plt.ylim(-3,3)
	plt.show()


def main():
	## Obtain the data from the data sets
	getData()

	## taking the user input
	to_run_program = int(input("enter: \n1 for perceptron, \n2 for lda\n"))
	on_which_dataset = int(input("enter 1\n for dataset->1\n 2 for dataset->2\n 3 for dataset->3\n"))

	if to_run_program == 1:
		perceptron(on_which_dataset)
	elif to_run_program == 2:
		lda(on_which_dataset)

if __name__ == "__main__":
	main()
