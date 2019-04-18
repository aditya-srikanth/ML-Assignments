import matplotlib.image as mpimg
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import PIL, pickle
import hard_code_params

def reduce_data():
    # dirpath, dirnames, filenames = os.walk('./data')
    if False :# os.path.isfile(hard_code_params.path_pickle_test) and os.path.isfile(hard_code_params.path_pickle_train):
        print('reduced data already present')
    else:
        testfiles = [f for f in listdir(hard_code_params.path_test) if isfile(join(hard_code_params.path_test, f))]
        trainfiles = [f for f in listdir(hard_code_params.path_train) if isfile(join(hard_code_params.path_train, f))]
        testing_data = np.empty((hard_code_params.dim*hard_code_params.dim,1))
        
        data_dict = {}
        labels = []
        print('testing_data')
        print('max for test',len(testfiles))
        for testfile in testfiles:
            
            img = PIL.Image.open(hard_code_params.path_test+'/'+testfile).convert('L')
            img = img.resize((hard_code_params.dim,hard_code_params.dim))
            # img.save('test.png')
            # print(img.size)
            pix = np.reshape(np.array(img).flatten(),(hard_code_params.dim*hard_code_params.dim,1))
            testing_data = np.hstack( (testing_data,pix) )
            print(testing_data.shape)

        data_dict['test'] = testing_data
        with open(hard_code_params.path_pickle_test,'wb') as f:
            pickle.dump(data_dict,f)
        
        training_data = np.empty((hard_code_params.dim*hard_code_params.dim,1))
        data_dict = {}
        print('training_data')
        print('max in train is ',len(trainfiles))
        for trainfile in trainfiles:
            labels += [trainfile.split(hard_code_params.delim)[0]]
            img = PIL.Image.open(hard_code_params.path_train+'/'+trainfile).convert('L')
            img = img.resize((hard_code_params.dim,hard_code_params.dim))
            # img.save('test.png')
            # print(img.size)
            pix = np.reshape(np.array(img).flatten(),(hard_code_params.dim*hard_code_params.dim,1))
            training_data = np.hstack((training_data,pix))
            print(training_data.shape)

        
        data_dict['train'] = training_data
        data_dict['labels'] = labels
        with open(hard_code_params.path_pickle_train,'wb') as f:
            pickle.dump(data_dict,f)
            # print(pix.shape)
        # os.mkdir('./reduced/test')
        # os.mkdir('./reduced/train')




    # print(len(filenames[2]))
    # for filename in filenames[2]:
    #     print('./data/')
def main():
    print(os.path.isdir('./data'))
    if os.path.isdir('./data') and os.path.isdir(hard_code_params.path_test) and os.path.isdir(hard_code_params.path_train):
        reduce_data()
    else:
        print("No data present")

if __name__ == '__main__':
    main()
