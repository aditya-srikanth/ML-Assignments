import hardcoded_params
import os
import pandas as pd
import pickle
def get_logistic_data():
	print('Searching for the data')
	if  not os.path.isfile(hardcoded_params.PATH_TO_PICKLE_LOGISTIC):
		data = pd.read_csv(hardcoded_params.PATH_LOGISTIC_DATASET,header=0).values
		explanatory_variables = data[:,:4]
		class_lables = data[:,-1]
		data_dict = {}
		data_dict ['explanatory_variables'] = explanatory_variables
		data_dict ['class_lables'] = class_lables

		with open(hardcoded_params.PATH_TO_PICKLE_LOGISTIC,'wb') as file_handle:
			pickle.dump(data_dict,file_handle)

		print('Data extracted successfully')

	else:
		print('pickle file for logistic already present')


if __name__ == '__main__':
	get_logistic_data()