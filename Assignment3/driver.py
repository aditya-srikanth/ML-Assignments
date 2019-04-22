import hard_code_params
import pickle
import numpy as np
from layer import Layer

def read_data():
    with open(hard_code_params.path_pickle_train,'rb') as file_handle:
        train_dict = pickle.load(file_handle)
        train = train_dict['train']
        labels = train_dict['labels']

    with open(hard_code_params.path_pickle_test,'rb') as file_handle:
        test_dict = pickle.load(file_handle)
        test = test_dict['test']

    return train,labels,test

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
        x = int ( input('enter dimension one\n') )
        y = int ( input('enter dimension two\n') )

        hidden_units = (x,y)
        weights_list.append(Layer(hidden_units))

    train,labels,test = read_data()
    
    labels = np.array([0 if x == 'cat' else 1 for x in labels])
    # print(labels)
    
    for epoch in range(hard_code_params.epochs):
        input_for_layers = train
        for layer in weights_list:    
            input_for_layers = layer.forward(input_for_layers)
        

        grad = labels - input_for_layers
        for layer in weights_list:    
            grad = layer.backward(grad)

        for layer in weights_list:
            layer.update()
    
    input_for_layers = train
    for layer in weights_list:    
        input_for_layers = layer.forward(input_for_layers)

    input_for_layers[input_for_layers > 0.5] = 1
    input_for_layers[input_for_layers != 1] = 0
    input_for_layers = input_for_layers.flatten()
    accuracy = 0

    for train_point_index in range(labels.shape[0]):
        input_for_layers[train_point_index] == labels[train_point_index]
        accuracy += 1
    
    print(accuracy/train.shape[0])

    # sanitycheck(weights_list,train,labels)

if __name__ == '__main__':
    main()