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
				data.append(np.array([parts[1],np.array(parts[3:])]))
				break

		# print(data)
		# np.array(data)
		# print(data)
		# print()
		# print()
		# print(data[0][0])
		# class_lables = data[:,0]
		print(class_lables)
		explanatory_variables = data[:][-1]
		print(explanatory_variables)
		input('stop')
		data_dict = {}
		data_dict ['explanatory_variables'] = explanatory_variables
		data_dict ['class_lables'] = class_lables

		with open(hardcoded_params.PATH_TO_PICKLE_BAYES,'wb') as file_handle:
			pickle.dump(data_dict,file_handle)

		print('Data extracted successfully')

	else:
		print('pickle file for logistic already present')


if __name__ == '__main__':
	get_bayes_data()