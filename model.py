import torch
import torch.nn as nn 
#making a feed forward nn model with two hidden layers

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) #(input, output)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes) #num_classes has to be fixed
        self.relu = nn.ReLU() #activation functions

    def forward(self, x): #forward pass
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        #no activation and no softmax
        return out
    

        

