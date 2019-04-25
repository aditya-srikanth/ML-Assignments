import hard_code_params
import pickle
import numpy as np
from layer import Layer
from keras import models,layers
from sklearn.metrics import precision_recall_fscore_support

def read_data():
    with open(hard_code_params.path_pickle_train,'rb') as file_handle:
        train_dict = pickle.load(file_handle)
        train = train_dict['train']
        labels = train_dict['labels']

    with open(hard_code_params.path_pickle_test,'rb') as file_handle:
        test_dict = pickle.load(file_handle)
        test = test_dict['test']
        test_labels = test_dict['labels']

    return train,labels,test,test_labels

# def sanitycheck(weights_list,train,labels):
#     num_output = len ( set(labels) )
#     if(weights_list[0].shape[0] != train.shape[0]):
#         print('Initial layer has the wrong dimensions')
#         exit()
#     elif(num_output != weights_list[-1].shape[1]):
#         print('The final layer has the wrong dimensions')
#         exit()
#     else:
#         for i in range(0,len(weights_list)-1):
#             if(weights_list[i].shape[1] != weights_list[i].shape[0] ):
#                 print('There is a mismatch in the layer dimensions at ',i,' ',i+1)
#                 exit()
    
#     print('Everything in order')

def main():
    Layer.number_of_layers = int ( input('Enter the number of layers\n') )
    weights_list = []
    print('Enter the number of hidden units as a tuple\n\n')

    for i in range(0,Layer.number_of_layers):
        print("Enter the data for layer ",i+1)
        x = int ( input('enter dimension one\n') )
        y = int ( input('enter dimension two\n') )

        hidden_units = (x,y)
        weights_list.append(Layer(hidden_units))

    train,labels,test,test_labels = read_data()

    labels = np.array([0 if x == 'cat' else 1 for x in labels])
    test_labels = np.array([0 if x == 'cat' else 1 for x in test_labels])

    ## Hack
    test_labels = labels
    test = train
    # print(labels)
    
    for epoch in range(hard_code_params.epochs):
        input_for_layers = train
        for layer in weights_list:    
            input_for_layers = layer.forward(input_for_layers)

        print(epoch,'loss: ',np.sum((input_for_layers - labels)**2)/train.shape[1])
        
        delta = input_for_layers - labels                                                                              
        delta,weights = weights_list[-1].backward(delta,None,last_layer=True)
        # print('\nlooping: \n')
        for layer in reversed(weights_list[:-1]):    
            delta,weights = layer.backward(delta,weights)

        for layer in weights_list:
            layer.update()
    

    input_for_layers = test
    for layer in weights_list:    
        input_for_layers = layer.forward(input_for_layers)

    input_for_layers[input_for_layers >= 0.5] = 1
    input_for_layers[input_for_layers != 1] = 0
    input_for_layers = input_for_layers.flatten()
    accuracy = 0
    ## TODO: CHANGE THE INDICES FOR TESTING ACCORDINGLY
    for test_point_index in range(test_labels.shape[0]):
        print(test_point_index,input_for_layers[test_point_index],test_labels[test_point_index])
        if input_for_layers[test_point_index] == test_labels[test_point_index]:
            accuracy += 1


    # input_for_layers = train
    # for layer in weights_list:    
    #     input_for_layers = layer.forward(input_for_layers)

    # input_for_layers[input_for_layers >= 0.5] = 1
    # input_for_layers[input_for_layers != 1] = 0
    # input_for_layers = input_for_layers.flatten( )
    # accuracy = 0


    
    # ## TODO: CHANGE THE INDICES FOR TESTING ACCORDINGLY
    # for test_point_index in range(labels.shape[0]):
    #     print(test_point_index,input_for_layers[test_point_index],labels[test_point_index])
    #     if input_for_layers[test_point_index] == labels[test_point_index]:
    #         accuracy += 1

    # print(train)
    print('accuracy: ',accuracy)
    print('train stats',' num cats: ',labels[labels == 0].shape,' num dogs: ',labels[labels == 1].shape)
    print('accuracy: ',accuracy/test.shape[1],' num cats: ',test_labels[test_labels == 0].shape,' num dogs: ',test_labels[test_labels == 1].shape)
    # sanitycheck(weights_list,train,labels)
    print(precision_recall_fscore_support(test_labels,input_for_layers))


    print("BASELINES USING KERAS")
    model = models.Sequential()

    model.add(
                layers.Dense(2500,activation='sigmoid',input_dim=train.shape[0])
            )
    model.add(
                layers.Dense(1,activation='sigmoid')
            )
                
    model.compile('SGD',loss='binary_crossentropy',metrics=['accuracy'])

    print(train.shape,labels.shape)

    results = model.fit(
    train.T, labels.T,
    epochs= 100 ,
    validation_data = (test.T, test_labels.T)
    )
    print(results)


if __name__ == '__main__':
    main()