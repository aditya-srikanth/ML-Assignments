import matplotlib.image as mpimg
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import PIL, pickle

def reduce_data():
    # dirpath, dirnames, filenames = os.walk('./data')
    if os.path.isfile('./reduced/test.pickle') and os.path.isfile('./reduced/train.pickle'):
        print('reduced data already present')
    else:
        testfiles = [f for f in listdir('./data/test') if isfile(join('./data/test', f))]
        trainfiles = [f for f in listdir('./data/train') if isfile(join('./data/train', f))]
        training_data = np.empty((2500,1))
        print('testing_data')
        for testfile in testfiles:
            img = PIL.Image.open('./data/test/'+testfile).convert('L')
            img = img.resize((50,50))
            # img.save('test.png')
            # print(img.size)
            pix = np.reshape(np.array(img).flatten(),(2500,1))
            training_data = np.hstack( (training_data,pix) )
            print(training_data.shape)
        with open('./test.pickle') as f:
            pickle.dump(training_data,f)
        
        testing_data = np.empty((2500,1))
        print('training_data')
        for trainfile in trainfiles:
            img = PIL.Image.open('./data/train/'+trainfile).convert('L')
            img = img.resize((50,50))
            # img.save('test.png')
            # print(img.size)
            pix = np.reshape(np.array(img).flatten(),(2500,1))
            testing_data = np.hstack((testing_data,pix))
            print(training_data.shape)
        with open('./train.pickle') as f:
            pickle.dump(training_data,f)
            # print(pix.shape)
        # os.mkdir('./reduced/test')
        # os.mkdir('./reduced/train')




    # print(len(filenames[2]))
    # for filename in filenames[2]:
    #     print('./data/')
def main():
    print(os.path.isdir('./data'))
    if os.path.isdir('./data') and os.path.isdir('./data/test') and os.path.isdir('./data/train'):
        reduce_data()
    else:
        print("No data present")

if __name__ == '__main__':
    main()
