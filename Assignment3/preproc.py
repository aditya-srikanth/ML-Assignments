import matplotlib.image as mpimg
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import PIL, pickle
import hard_code_params

def reduce_data():
    # dirpath, dirnames, filenames = os.walk('./data')
    if os.path.isfile(hard_code_params.path_pickle_test) and os.path.isfile(hard_code_params.path_pickle_train):
        print('reduced data already present')
    else:
        os.mkdir('./reduced/')
        os.mkdir('./reduced/train/')
        os.mkdir('./reduced/test/')
        testfiles = [f for f in listdir(hard_code_params.path_train) if isfile(join(hard_code_params.path_train, f))][20000:25000]
        # testfiles = [f for f in listdir(hard_code_params.path_test) if isfile(join(hard_code_params.path_test, f))][10000:15000]
        trainfiles = [f for f in listdir(hard_code_params.path_train) if isfile(join(hard_code_params.path_train, f))][:20000]
        
        
        testing_data = np.zeros((hard_code_params.dim*hard_code_params.dim,1))
        data_dict = {}
        labels = []
        print('testing_data')
        # print('max for test',len(testfiles))
  
        testing_data = np.zeros((hard_code_params.dim*hard_code_params.dim,1))
        i = 0
        for testfile in testfiles:
            i += 1
            if i%100 == 0:
                print('testing ',i)
            labels += [testfile.split(hard_code_params.delim)[0]]
            # img = PIL.Image.open(hard_code_params.path_test+'/'+testfile).convert('L')
            img = PIL.Image.open(hard_code_params.path_train+'/'+testfile).convert('L')
            img = img.resize((hard_code_params.dim,hard_code_params.dim))
            # img.save('test.png')
            # print(img.size)
            pix = np.reshape(np.array(img).flatten(),(hard_code_params.dim*hard_code_params.dim,1))
            testing_data = np.hstack( (testing_data,pix) )
        testing_data = testing_data[:,1:]
        print(testing_data.shape)
        # print(testing_data.shape)
        testing_data =testing_data.T
        testing_data = (testing_data - np.mean(testing_data,axis=0))/np.std(testing_data,axis=0)
        testing_data =testing_data.T
        data_dict['test'] = testing_data
        data_dict['labels'] = labels
        with open(hard_code_params.path_pickle_test,'wb') as f:
            pickle.dump(data_dict,f)
        
        training_data = np.zeros((hard_code_params.dim*hard_code_params.dim,1))
        data_dict = {}
        labels = []
        print('training_data')
        print('max in train is ',len(trainfiles))
        i = 0
        for trainfile in trainfiles:
            i += 1
            if i%100 == 0:
                print('training ',i)
            labels += [trainfile.split(hard_code_params.delim)[0]]
            img = PIL.Image.open(hard_code_params.path_train+'/'+trainfile).convert('L')
            img = img.resize((hard_code_params.dim,hard_code_params.dim))
            # img.save('test.png')
            # print(img.size)
            pix = np.reshape(np.array(img).flatten(),(hard_code_params.dim*hard_code_params.dim,1))
            training_data = np.hstack((training_data,pix))
        training_data = training_data[:,1:]
        print(training_data.shape)
        training_data = training_data.T
        training_data = (training_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
        training_data = training_data.T
        data_dict['train'] = training_data
        data_dict['labels'] = labels
        with open(hard_code_params.path_pickle_train,'wb') as f:
            pickle.dump(data_dict,f)
            # print(pix.shape)





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
