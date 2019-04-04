import hardcoded_params
import os
import pickle
import numpy as np

def get_bayes_data():
	print('Searching for the data')
	data = []
	if  not os.path.isfile(hardcoded_params.PATH_TO_PICKLE_BAYES):
		with open(hardcoded_params.PATH_BAYES_DATASET,'r') as file_handle:
			for line in file_handle:
				line = line.strip()
				parts = line.split(" ")
				data.append(np.array([parts[1],np.array(list(set(parts[3:])))]))

		permutation = np.random.permutation(len(data)).tolist()

		data = [data[permutation_element] for permutation_element in permutation]
		train_size = int(0.8*len(data)) # 80 - 20 train test split
		test_size = len(data) - train_size

		explanatory_variables = [variable[-1] for variable in data]
		class_lables = [variable[0] for variable in data]

		train_data = {}
		test_data = {}

		train_data['train_size'] = train_size
		train_data['explanatory_variables'] = explanatory_variables[:train_size]
		train_data['class_labels'] = class_lables[:train_size]

		test_data['test_size'] = test_size
		test_data['explanatory_variables'] = explanatory_variables[train_size:]
		test_data['class_labels'] = class_lables[train_size:]

		data_dict = {}
		data_dict ['train_data'] = train_data
		data_dict ['test_data'] = test_data

		if not os.path.isdir(hardcoded_params.PATH_TO_PICKLE):
			os.mkdir(hardcoded_params.PATH_TO_PICKLE)

		with open(hardcoded_params.PATH_TO_PICKLE_BAYES,'wb') as file_handle:
			pickle.dump(data_dict,file_handle)

		print('Data extracted successfully')

	else:
		print('pickle file for bayes already present')


if __name__ == '__main__':
	get_bayes_data()