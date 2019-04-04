import hardcoded_params
import os
import pandas as pd
import pickle
import numpy as np

def get_logistic_data():


	""" Description
	:raises:

	:rtype:
	"""
	
	print('Searching for the data')
	if  not os.path.isfile(hardcoded_params.PATH_TO_PICKLE_LOGISTIC):
		data = pd.read_csv(hardcoded_params.PATH_LOGISTIC_DATASET,header=0).values
		np.random.shuffle(data)

		train_size = int(0.8*data.shape[0])
		test_size = data.shape[0] - train_size

		explanatory_variables = data[:,:-1]
		class_lables = data[:,-1]
		
		# feature scaling
		explanatory_variables = (explanatory_variables - np.mean(explanatory_variables,axis=0))/np.std(explanatory_variables,axis=0)

		train_data = explanatory_variables[:train_size]
		test_data = explanatory_variables[train_size:]
		train_labels = class_lables[:train_size]
		test_labels = class_lables[train_size:]

		train_dict = {}
		test_dict = {}
		train_dict['explanatory_variables'] = train_data
		train_dict['class_labels'] = train_labels

		test_dict['explanatory_variables'] = test_data
		test_dict['class_labels'] = test_labels

		data_dict = {}
		data_dict ['train_data'] = train_dict
		data_dict ['test_data'] = test_dict

		if not os.path.isdir(hardcoded_params.PATH_TO_PICKLE):
			os.mkdir(hardcoded_params.PATH_TO_PICKLE)

		with open(hardcoded_params.PATH_TO_PICKLE_LOGISTIC,'wb') as file_handle:
			pickle.dump(data_dict,file_handle)

		print('Data extracted successfully')

	else:
		print('pickle file for logistic already present')


if __name__ == '__main__':
	get_logistic_data()