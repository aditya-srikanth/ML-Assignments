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

    def forward(self,output_of_prev):
    
        """ Description
        :type self:
        :param self:
    
        :type input_of_prev:
        :param input_of_prev:
    
        :raises:
    
        :rtype:
        """
        self.input = output_of_prev
        y = np.dot( self.weights.T , output_of_prev )
        y = y + self.bias
        # print('output shape: ',y.shape)
        y = self.activation(y)
        self.output = y
        return np.array(y)

    def backward(self,delta,next_layer_weights,last_layer=False):
    
        """ Description
        :type self:
        :param self:
    
        :type grad_of_next:
        :param grad_of_next:
    
        :raises:
    
        :rtype:
        """    
        pass
        ## TODO: DEBUG
        
        if last_layer:
            self.gradient_w = self.input.dot(delta.T)
            self.gradient_b = np.sum(delta,axis=1)
        else:
            delta = (next_layer_weights.dot(delta ))* self.activation(self.output,deriv=True)
            self.gradient_w = self.input.dot(delta.T)
            self.gradient_b = np.reshape(np.sum(delta,axis=1),(self.weights.shape[1],1))
        
        return delta,np.array(self.weights)

    def update(self):
        self.weights -= self.gradient_w*hard_code_params.learning_rate
        self.bias -= self.gradient_b*hard_code_params.learning_rate
        
