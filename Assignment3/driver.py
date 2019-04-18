import hard_code_params
import pickle
import numpy as np
class Layer:

    def __init__(self,hidden_units):
        self.weights = np.random.normal(hard_code_params.mu,hard_code_params.sigma,hidden_units)
        self.bias = np.random.normal(hard_code_params.mu,hard_code_params.sigma,(hidden_units[1],1)) # hidden_units[0] input
        self.gradient = None
        self.output = None

    def forward(self,input_of_prev):
        y = np.dot( weights.transpose() , input_of_prev )
        y = y + bias
        return y

    def backward(self,grad_of_next):
        pass
        ## TODO: Lookup Adam optimizer
        # grad = next_grad X W dot deriv at output using this: * TODO fit the dimensions
    
    def update(self):
        pass

def read_data():
    with open(hard_code_params.path_pickle_train,'rb') as file_handle:
        train_dict = pickle.load(file_handle)
        train = train_dict['train']
        labels = train_dict['labels']

    with open(hard_code_params.path_pickle_test,'rb') as file_handle:
        test_dict = pickle.load(file_handle)
        test = test_dict['test']

    return train,labels,test

def sanitycheck(weights_list,train,labels):
    num_output = len ( set(labels) )
    if(weights_list[0].shape[0] != train.shape[0]):
        print('Initial layer has the wrong dimensions')
        exit()
    elif(num_output != weights_list[-1].shape[1]):
        print('The final layer has the wrong dimensions')
        exit()
    else:
        for i in range(0,len(weights_list)-1):
            if(weights_list[i].shape[1] != weights_list[i].shape[0] ):
                print('There is a mismatch in the layer dimensions at ',i,' ',i+1)
                exit()
    
    print('Everything in order')

def main():
    Layer.number_of_layers = int ( input('Enter the number of layers\n') )
    weights_list = []
    print('Enter the number of hidden units as a tuple\n\n')

    for i in range(0,Layer.number_of_layers):
        x = int ( input('enter dimension one') )
        y = int ( input('enter dimension two') )

        hidden_units = (x,y)
        weights_list.append(Layer(hidden_units))

    train,labels,test = read_data()
    
    sanitycheck(weights_list,train,labels)

if __name__ == '__main__':
    main()