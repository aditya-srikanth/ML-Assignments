import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
import hardcoded_params

def sigmoid(X,deriv = False):
    if deriv:
        return X*(1-X)
    return 1/(1 + np.exp(-X))

def logistic_regression():
    
    """ Description
        Following bishop conventions, actual labels = T and predicted labels = Y
    :raises:

    :rtype:
    """

    if not os.path.isfile(hardcoded_params.PATH_TO_PICKLE_LOGISTIC):
        raise FileNotFoundError
        
    else:
        with open(hardcoded_params.PATH_TO_PICKLE_LOGISTIC,'rb') as f:
            data_dict= pickle.load(f)

        train_X = (data_dict['train_data'])['explanatory_variables'].T
        train_T = (data_dict['train_data'])['class_labels'].T

        test_X = (data_dict['test_data'])['explanatory_variables'].T
        test_T = (data_dict['test_data'])['class_labels'].T


        N = train_X.shape[1]

        # reshaping for matrix multiplication
        train_T = np.reshape(train_T,(1,train_T.shape[0]))
        test_T = np.reshape(test_T,(1,test_T.shape[0]))
        
        # get parameters    
        W = np.random.randn(train_X.shape[0],1)
        b = np.random.randn()

        # FOR PLOTS
        results_list = []

        print('for Non-Regularized Logistic Regression after ',hardcoded_params.NUMBER_OF_EPOCHS,' epochs')
        print('output format: (learning_rate, accuracy)')
        for learning_rate in np.arange(0.001,0.01,0.0001):
            for epoch in range(hardcoded_params.NUMBER_OF_EPOCHS):
                Y = sigmoid(W.T.dot(train_X) + b)
                # loss = -np.sum(train_T*np.log(Y) + (1-train_T)*np.log(1-Y))/N
                grad_W = train_X.dot(((Y - train_T)).T)
                grad_b = np.sum(((Y - train_T)))
                W = W - learning_rate/N*grad_W
                b = b - learning_rate/N*grad_b

            # testing
            Y = sigmoid(W.T.dot(test_X) + b)
            Y[Y >= 0.5] = 1
            Y[Y != 1] = 0
            
            res = np.where(Y == test_T)
            
            print(learning_rate,'\t',len(res[0])/test_T.shape[1]*100)

        print('for Regularized Logistic Regression with Regularization param: ',hardcoded_params.REGULARIZATION_PARAM,' and number of epochs ',hardcoded_params.NUMBER_OF_EPOCHS)
        print('output format: (learning_rate, accuracy)')
        for learning_rate in np.arange(0.001,0.01,0.0001):
            for epoch in range(hardcoded_params.NUMBER_OF_EPOCHS):
                Y = sigmoid(W.T.dot(train_X) + b)
                # loss = -np.sum(train_T*np.log(Y) + (1-train_T)*np.log(1-Y))/N + (hardcoded_params.REGULARIZATION_PARAM/(2)) * W.T.dot(W)
                grad_W = train_X.dot(((Y - train_T)).T) + (hardcoded_params.REGULARIZATION_PARAM) * W
                W = W - learning_rate/N*grad_W
                grad_b = np.sum(((Y - train_T)))
                b = b - learning_rate/N*grad_b
            
            # testing
            Y = sigmoid(W.T.dot(test_X) + b)
            Y[Y >= 0.5] = 1
            Y[Y != 1] = 0
            
            res = np.where(Y == test_T)
            
            print(learning_rate,'\t',len(res[0])/Y.shape[1]*100)

def naive_bayes():
    
    """ Description
    :raises:

    :rtype:
    """
    if not os.path.isfile(hardcoded_params.PATH_TO_PICKLE_BAYES):
        raise FileNotFoundError

    else:
        with open(hardcoded_params.PATH_TO_PICKLE_BAYES,'rb') as f:
            data_dict= pickle.load(f)

        train_size = (data_dict['train_data'])['train_size']
        train_features = (data_dict['train_data'])['explanatory_variables'] 
        train_labels = (data_dict['train_data'])['class_labels']

        test_size = (data_dict['test_data'])['test_size']
        test_features = (data_dict['test_data'])['explanatory_variables']
        test_labels = (data_dict['test_data'])['class_labels']

        conditional_probability_table = {}
        prior ={}
        prior[hardcoded_params.POSITIVE_CLASS] = 0
        prior[hardcoded_params.NEGATIVE_CLASS] = 0
        N = len(train_features)

        for train_data_index in range(train_size):
            datapoint_class = train_labels[train_data_index]
            prior[datapoint_class] += 1/N
            for word in train_features[train_data_index]:
                if datapoint_class == hardcoded_params.POSITIVE_CLASS:
                    if word in conditional_probability_table:
                        conditional_probability_table[word][0] += 1/N
                    else:
                        conditional_probability_table[word] = [1/N,0]
                elif datapoint_class == hardcoded_params.NEGATIVE_CLASS:
                    if word in conditional_probability_table:
                        conditional_probability_table[word][1] += 1/N
                    else:
                        conditional_probability_table[word] = [0,1/N]
        
        acc = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for test_data_index in range(test_size):
            datapoint_class = test_labels[test_data_index]
            pos_res = prior[hardcoded_params.POSITIVE_CLASS]
            neg_res = prior[hardcoded_params.NEGATIVE_CLASS]
            for word in test_features[test_data_index]:
                if word not in conditional_probability_table:
                    continue 
                pos_res *= conditional_probability_table[word][0]
                neg_res *= conditional_probability_table[word][1]
            if pos_res >= neg_res and datapoint_class == hardcoded_params.POSITIVE_CLASS:
                acc += 1
                tp += 1

            elif pos_res < neg_res and datapoint_class == hardcoded_params.NEGATIVE_CLASS:
                acc += 1
                tn += 1
            
            elif pos_res >= neg_res and datapoint_class == hardcoded_params.NEGATIVE_CLASS:
                fp += 1
            elif pos_res < neg_res and datapoint_class == hardcoded_params.POSITIVE_CLASS:
                fn += 1

        print('tp: %s fp: %s tn: %s fn: %s'%(tp,fp,tn,fn))
        print('accuracy',acc/len(test_features)*100)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        print('precision: ',precision)
        print('recall: ',recall)
        print('f1 measure: ',(2*precision*recall)/(precision + recall))
        # print(precision_recall_fscore_support(actual,predicted))
        # print(confusion_matrix(actual,predicted))

def main():
    algorithm = int(input('enter 0 for logistic regression\nenter 1 for naive bayes\n'))
    if algorithm == 0:
        logistic_regression()
    elif algorithm == 1:
        naive_bayes()


if __name__ == "__main__":
    main()
