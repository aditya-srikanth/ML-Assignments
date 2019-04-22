import numpy as np
import hard_code_params

class Layer:

    def __init__(self,hidden_units):
    
        """ Description
        :type self:
        :param self:
    
        :type hidden_units:
        :param hidden_units:
    
        :raises:
    
        :rtype:
        """    
        self.weights = np.random.normal(hard_code_params.mu,hard_code_params.sigma,hidden_units)
        self.bias = np.random.normal(hard_code_params.mu,hard_code_params.sigma,(hidden_units[1],1)) # hidden_units[0] input
        self.gradient_w = None
        self.gradient_b = None
        self.output = None
        self.input = None

    def activation(self,X,deriv = False):
        # calculate activation
        if not deriv:
            return 1/(1+np.exp(-X))
        else:
            return (X)*(1-X)

    def forward(self,input_of_prev):
    
        """ Description
        :type self:
        :param self:
    
        :type input_of_prev:
        :param input_of_prev:
    
        :raises:
    
        :rtype:
        """
        self.input = input_of_prev
        y = np.dot( self.weights.transpose() , input_of_prev )
        y = y + self.bias
        print(y.shape)
        y = self.activation(y)
        self.output = y
        return y

    def backward(self,grad_of_next):
    
        """ Description
        :type self:
        :param self:
    
        :type grad_of_next:
        :param grad_of_next:
    
        :raises:
    
        :rtype:
        """    
        pass
        ## TODO: Lookup Adam optimizer
        # grad = next_grad X W dot deriv at output using this: * TODO fit the dimensions
        self.gradient_w = (grad_of_next*self.activation(self.output)).dot(self.input.T)
        
        print('aws',(grad_of_next*self.activation(self.output)).shape)
        print('asfa',self.input.T.shape)
        self.gradient_b = np.sum((grad_of_next*self.activation(self.output)),axis=1)
        print(self.gradient_w.shape)
        print(self.gradient_b.shape)
        return self.gradient_w

    def update(self):
        self.weights -= self.gradient_w*hard_code_params.learning_rate
        self.bias -= self.gradient_b*hard_code_params.learning_rate